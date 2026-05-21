#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

TOKENS_PER_SECOND_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:tokens/s|tok/s)", re.IGNORECASE)
EPSILON_DISTANCE = 1e-6
EPSILON_VARIANCE = 1e-12
MIN_ELAPSED_TIME = 1e-6
QUICK_SAMPLING_ATTEMPTS = 300
MAX_CANDIDATE_POOL = 128
MAX_POOL_ATTEMPTS = 2000
EXPLORATION_WEIGHT = 1.4
DISTANCE_WEIGHT = 0.15
K_NEIGHBORS = 6
MIN_WARMUP_ITERATIONS = 5
WARMUP_MULTIPLIER = 2

DEFAULT_SEARCH_SPACE = {
    "prompt_argument": "--prompt",
    "static_arguments": [],
    "common_parameters": {},
    "spec_methods": [
        {
            "name": "k",
            "spec_type": "k",
            "drafter_argument": "--spec-drafters",
            "drafter_chains": [["k"]],
            "parameters": {
                "--spec-draft-type-k": {
                    "type": "categorical",
                    "options": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
                },
                "--spec-draft-max-k": {"type": "int", "min": 2, "max": 16},
            },
        },
        {
            "name": "k+eagle",
            "spec_type": "combo",
            "drafter_argument": "--spec-drafters",
            "drafter_chains": [["k", "eagle"], ["eagle", "k"]],
            "parameters": {
                "--spec-draft-type-k": {
                    "type": "categorical",
                    "options": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
                },
                "--spec-accept-threshold": {"type": "float", "min": 0.1, "max": 0.9},
            },
        },
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    options: Optional[Tuple[Any, ...]] = None

    @staticmethod
    def from_json(name: str, raw: Dict[str, Any]) -> "ParameterSpec":
        kind = raw.get("type")
        if kind == "int":
            minimum = int(raw["min"])
            maximum = int(raw["max"])
            step = int(raw.get("step", 1))
            if maximum < minimum:
                raise ValueError(f"int parameter '{name}' has max < min")
            if step <= 0:
                raise ValueError(f"int parameter '{name}' step must be > 0")
            return ParameterSpec(
                name=name,
                kind=kind,
                minimum=minimum,
                maximum=maximum,
                step=step,
            )
        if kind == "float":
            minimum = float(raw["min"])
            maximum = float(raw["max"])
            if maximum < minimum:
                raise ValueError(f"float parameter '{name}' has max < min")
            return ParameterSpec(name=name, kind=kind, minimum=minimum, maximum=maximum)
        if kind == "categorical":
            options = tuple(raw["options"])
            if not options:
                raise ValueError(f"categorical parameter '{name}' has no options")
            return ParameterSpec(name=name, kind=kind, options=options)
        if kind == "flag":
            return ParameterSpec(name=name, kind=kind)
        raise ValueError(f"unsupported parameter type '{kind}' for '{name}'")

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "int":
            assert self.minimum is not None and self.maximum is not None and self.step is not None
            choices = int((self.maximum - self.minimum) / self.step) + 1
            idx = rng.randrange(choices)
            return int(self.minimum + idx * self.step)
        if self.kind == "float":
            assert self.minimum is not None and self.maximum is not None
            return self.minimum + (self.maximum - self.minimum) * rng.random()
        if self.kind == "categorical":
            assert self.options is not None
            return rng.choice(self.options)
        if self.kind == "flag":
            return bool(rng.getrandbits(1))
        raise ValueError(f"unsupported kind: {self.kind}")

    def mutate(self, value: Any, rng: random.Random) -> Any:
        if self.kind == "int":
            assert self.minimum is not None and self.maximum is not None and self.step is not None
            width = max(self.step, (self.maximum - self.minimum) / 8)
            delta = int(round(rng.gauss(0, width) / self.step)) * self.step
            candidate = int(value) + delta
            candidate = max(int(self.minimum), min(int(self.maximum), candidate))
            snapped = int(round((candidate - self.minimum) / self.step) * self.step + self.minimum)
            return max(int(self.minimum), min(int(self.maximum), snapped))
        if self.kind == "float":
            assert self.minimum is not None and self.maximum is not None
            width = (self.maximum - self.minimum) / 6.0
            candidate = float(value) + rng.gauss(0, width)
            return max(self.minimum, min(self.maximum, candidate))
        if self.kind == "categorical":
            assert self.options is not None
            if len(self.options) == 1:
                return self.options[0]
            others = [o for o in self.options if o != value]
            return rng.choice(others)
        if self.kind == "flag":
            return not bool(value)
        raise ValueError(f"unsupported kind: {self.kind}")

    def normalize(self, value: Any) -> float:
        if self.kind == "int":
            assert self.minimum is not None and self.maximum is not None
            if self.maximum == self.minimum:
                return 0.0
            return (float(value) - self.minimum) / (self.maximum - self.minimum)
        if self.kind == "float":
            assert self.minimum is not None and self.maximum is not None
            if self.maximum == self.minimum:
                return 0.0
            return (float(value) - self.minimum) / (self.maximum - self.minimum)
        if self.kind == "categorical":
            assert self.options is not None
            if len(self.options) == 1:
                return 0.0
            return self.options.index(value) / (len(self.options) - 1)
        if self.kind == "flag":
            return 1.0 if bool(value) else 0.0
        raise ValueError(f"unsupported kind: {self.kind}")


@dataclass(frozen=True)
class MethodSpec:
    name: str
    spec_type: str
    parameters: Tuple[ParameterSpec, ...]
    drafter_argument: str
    drafter_chains: Tuple[Tuple[str, ...], ...]

    @staticmethod
    def from_json(raw: Dict[str, Any]) -> "MethodSpec":
        param_items = sorted((raw.get("parameters") or {}).items(), key=lambda item: item[0])
        params = tuple(ParameterSpec.from_json(name, definition) for name, definition in param_items)
        chains = tuple(tuple(chain) for chain in raw.get("drafter_chains", []))
        return MethodSpec(
            name=raw["name"],
            spec_type=raw["spec_type"],
            parameters=params,
            drafter_argument=raw.get("drafter_argument", "--spec-drafters"),
            drafter_chains=chains,
        )


@dataclass(frozen=True)
class Candidate:
    method_name: str
    spec_type: str
    parameters: Dict[str, Any]
    drafter_chain: Optional[Tuple[str, ...]]

    def key(self) -> str:
        return json.dumps(
            {
                "method_name": self.method_name,
                "spec_type": self.spec_type,
                "parameters": {k: self.parameters[k] for k in sorted(self.parameters)},
                "drafter_chain": list(self.drafter_chain) if self.drafter_chain else None,
            },
            sort_keys=True,
        )


class SearchSpace:
    def __init__(self, raw: Dict[str, Any]):
        methods = raw.get("spec_methods", [])
        if not methods:
            raise ValueError("search space must contain at least one spec method")
        self.prompt_argument = raw.get("prompt_argument", "--prompt")
        self.static_arguments = tuple(raw.get("static_arguments", []))
        common_items = sorted((raw.get("common_parameters") or {}).items(), key=lambda item: item[0])
        self.common_parameters = tuple(ParameterSpec.from_json(name, definition) for name, definition in common_items)
        self.methods = tuple(MethodSpec.from_json(method) for method in methods)
        self._method_by_name = {method.name: method for method in self.methods}
        if len(self._method_by_name) != len(self.methods):
            raise ValueError("spec method names must be unique")
        self._feature_schema = self._build_feature_schema()

    def _build_feature_schema(self) -> List[Tuple[str, ParameterSpec]]:
        schema: List[Tuple[str, ParameterSpec]] = []
        for param in self.common_parameters:
            schema.append(("common", param))
        for method in self.methods:
            for param in method.parameters:
                schema.append((method.name, param))
        return schema

    @property
    def parameter_count(self) -> int:
        return len(self.common_parameters) + max(len(m.parameters) for m in self.methods) + 2

    def random_candidate(self, rng: random.Random) -> Candidate:
        method = rng.choice(self.methods)
        return self._sample_for_method(method, rng)

    def mutate_candidate(self, base: Candidate, rng: random.Random, mutate_rate: float = 0.35) -> Candidate:
        method = self._method_by_name[base.method_name]
        if rng.random() < 0.15:
            method = rng.choice(self.methods)
            return self._sample_for_method(method, rng)
        params: Dict[str, Any] = {}
        for param in self.common_parameters:
            current = base.parameters.get(param.name, param.sample(rng))
            params[param.name] = param.mutate(current, rng) if rng.random() < mutate_rate else current
        for param in method.parameters:
            current = base.parameters.get(param.name, param.sample(rng))
            params[param.name] = param.mutate(current, rng) if rng.random() < mutate_rate else current
        drafter_chain = base.drafter_chain
        if method.drafter_chains:
            if drafter_chain not in method.drafter_chains or rng.random() < mutate_rate:
                drafter_chain = rng.choice(method.drafter_chains)
        else:
            drafter_chain = None
        return Candidate(method_name=method.name, spec_type=method.spec_type, parameters=params, drafter_chain=drafter_chain)

    def _sample_for_method(self, method: MethodSpec, rng: random.Random) -> Candidate:
        parameters: Dict[str, Any] = {}
        for param in self.common_parameters:
            parameters[param.name] = param.sample(rng)
        for param in method.parameters:
            parameters[param.name] = param.sample(rng)
        chain = rng.choice(method.drafter_chains) if method.drafter_chains else None
        return Candidate(method_name=method.name, spec_type=method.spec_type, parameters=parameters, drafter_chain=chain)

    def to_cli_arguments(self, candidate: Candidate) -> List[str]:
        method = self._method_by_name[candidate.method_name]
        args = list(self.static_arguments)
        args.extend(["--spec-type", candidate.spec_type])
        for param in self.common_parameters:
            args.extend(self._render_argument(param, candidate.parameters[param.name]))
        for param in method.parameters:
            value = candidate.parameters[param.name]
            args.extend(self._render_argument(param, value))
        if candidate.drafter_chain:
            args.extend([method.drafter_argument, ",".join(candidate.drafter_chain)])
        return args

    def candidate_features(self, candidate: Candidate) -> List[float]:
        features: List[float] = []
        method_index = self.methods.index(self._method_by_name[candidate.method_name])
        if len(self.methods) == 1:
            features.append(0.0)
        else:
            features.append(method_index / (len(self.methods) - 1))

        method = self._method_by_name[candidate.method_name]
        if method.drafter_chains:
            if candidate.drafter_chain in method.drafter_chains:
                chain_index = method.drafter_chains.index(candidate.drafter_chain)
                if len(method.drafter_chains) == 1:
                    features.append(0.0)
                else:
                    features.append(chain_index / (len(method.drafter_chains) - 1))
            else:
                features.append(-1.0)
        else:
            features.append(0.0)

        active = {param.name: param for param in self.common_parameters}
        for param in method.parameters:
            active[param.name] = param
        for namespace, param in self._feature_schema:
            if namespace not in ("common", method.name):
                features.append(-1.0)
                continue
            if param.name not in active:
                features.append(-1.0)
            else:
                features.append(param.normalize(candidate.parameters[param.name]))
        return features

    @staticmethod
    def _render_argument(param: ParameterSpec, value: Any) -> List[str]:
        if param.kind == "flag":
            return [param.name] if bool(value) else []
        if param.kind == "float":
            return [param.name, f"{float(value):.6g}"]
        return [param.name, str(value)]

    def to_progress_dict(self) -> Dict[str, Any]:
        return {
            "prompt_argument": self.prompt_argument,
            "static_arguments": list(self.static_arguments),
            "common_parameters": {param.name: _param_to_dict(param) for param in self.common_parameters},
            "spec_methods": [
                {
                    "name": method.name,
                    "spec_type": method.spec_type,
                    "drafter_argument": method.drafter_argument,
                    "drafter_chains": [list(chain) for chain in method.drafter_chains],
                    "parameters": {param.name: _param_to_dict(param) for param in method.parameters},
                }
                for method in self.methods
            ],
        }


def _param_to_dict(param: ParameterSpec) -> Dict[str, Any]:
    if param.kind == "int":
        return {"type": "int", "min": int(param.minimum), "max": int(param.maximum), "step": int(param.step or 1)}
    if param.kind == "float":
        return {"type": "float", "min": param.minimum, "max": param.maximum}
    if param.kind == "categorical":
        return {"type": "categorical", "options": list(param.options or ())}
    return {"type": "flag"}


def load_prompts(path: Path) -> List[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError("prompt file contains no prompts")
    return prompts


def load_search_space(path: Optional[Path]) -> SearchSpace:
    if path is None:
        return SearchSpace(DEFAULT_SEARCH_SPACE)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return SearchSpace(raw)


def parse_tokens_per_second(output: str) -> Optional[float]:
    matches = TOKENS_PER_SECOND_RE.findall(output)
    if not matches:
        return None
    values = [float(value) for value in matches]
    return statistics.mean(values)


def subprocess_isolation_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"stdin": subprocess.DEVNULL}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    return kwargs


def evaluate_candidate(
    llama_cli: Path,
    search_space: SearchSpace,
    candidate: Candidate,
    prompts: Sequence[str],
    trials_per_setting: int,
) -> Dict[str, Any]:
    cli_prefix = [str(llama_cli)] + search_space.to_cli_arguments(candidate)
    prompt_argument = search_space.prompt_argument
    tokens_scores: List[float] = []
    inverse_elapsed_scores: List[float] = []
    runs = 0
    for _ in range(trials_per_setting):
        for prompt in prompts:
            command = cli_prefix + [prompt_argument, prompt]
            started = time.perf_counter()
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                **subprocess_isolation_kwargs(),
            )
            elapsed = max(time.perf_counter() - started, MIN_ELAPSED_TIME)
            runs += 1
            merged_output = (process.stdout or "") + "\n" + (process.stderr or "")
            if process.returncode != 0:
                raise RuntimeError(
                    f"llama-cli exited with code {process.returncode}\n"
                    f"command: {' '.join(command)}\n"
                    f"stdout:\n{process.stdout}\n"
                    f"stderr:\n{process.stderr}"
                )
            parsed = parse_tokens_per_second(merged_output)
            if parsed is not None:
                tokens_scores.append(parsed)
            inverse_elapsed_scores.append(1.0 / elapsed)
    if tokens_scores:
        score = statistics.mean(tokens_scores)
        metric_name = "tokens_per_second"
    else:
        score = statistics.mean(inverse_elapsed_scores)
        metric_name = "inverse_elapsed_seconds"
    return {
        "score": score,
        "metric_name": metric_name,
        "runs": runs,
        "tokens_scores": tokens_scores,
    }


def predict_candidate_score(candidate_features: List[float], history: Sequence[Dict[str, Any]]) -> Tuple[float, float, float]:
    if not history:
        return 0.0, 1.0, 1.0
    distances: List[Tuple[float, float]] = []
    for item in history:
        features = item["features"]
        score = item["score"]
        squared = 0.0
        for lhs, rhs in zip(candidate_features, features):
            if lhs == -1.0 or rhs == -1.0:
                continue
            squared += (lhs - rhs) ** 2
        distance = math.sqrt(squared)
        distances.append((distance, score))
    distances.sort(key=lambda row: row[0])
    neighbours = distances[: min(K_NEIGHBORS, len(distances))]
    weights = [1.0 / (row[0] + EPSILON_DISTANCE) for row in neighbours]
    weight_sum = sum(weights)
    mean = sum(weight * score for weight, (_, score) in zip(weights, neighbours)) / weight_sum
    variance = sum(weight * ((score - mean) ** 2) for weight, (_, score) in zip(weights, neighbours)) / weight_sum
    stddev = math.sqrt(max(variance, EPSILON_VARIANCE))
    min_distance = neighbours[0][0]
    return mean, stddev, min_distance


def choose_candidate(
    rng: random.Random,
    search_space: SearchSpace,
    history: Sequence[Dict[str, Any]],
    evaluated_keys: set,
    best_candidate: Optional[Candidate],
) -> Candidate:
    for _ in range(QUICK_SAMPLING_ATTEMPTS):
        if best_candidate and rng.random() < 0.65:
            candidate = search_space.mutate_candidate(best_candidate, rng)
        else:
            candidate = search_space.random_candidate(rng)
        if candidate.key() not in evaluated_keys:
            return candidate

    sampled: List[Candidate] = []
    attempts = 0
    while len(sampled) < MAX_CANDIDATE_POOL and attempts < MAX_POOL_ATTEMPTS:
        attempts += 1
        if best_candidate and rng.random() < 0.8:
            candidate = search_space.mutate_candidate(best_candidate, rng)
        else:
            candidate = search_space.random_candidate(rng)
        if candidate.key() in evaluated_keys:
            continue
        sampled.append(candidate)

    if not sampled:
        return search_space.random_candidate(rng)

    best_acquisition = None
    best = sampled[0]
    for candidate in sampled:
        features = search_space.candidate_features(candidate)
        mean, stddev, distance = predict_candidate_score(features, history)
        acquisition = mean + EXPLORATION_WEIGHT * stddev + DISTANCE_WEIGHT * distance
        if best_acquisition is None or acquisition > best_acquisition:
            best_acquisition = acquisition
            best = candidate
    return best


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_progress(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_from_dict(data: Dict[str, Any]) -> Candidate:
    chain = tuple(data["drafter_chain"]) if data.get("drafter_chain") else None
    return Candidate(
        method_name=data["method_name"],
        spec_type=data["spec_type"],
        parameters=dict(data["parameters"]),
        drafter_chain=chain,
    )


def candidate_to_dict(candidate: Candidate) -> Dict[str, Any]:
    return {
        "method_name": candidate.method_name,
        "spec_type": candidate.spec_type,
        "parameters": {k: candidate.parameters[k] for k in sorted(candidate.parameters)},
        "drafter_chain": list(candidate.drafter_chain) if candidate.drafter_chain else None,
    }


def print_live_stats(record: Dict[str, Any], best_score: float, best_candidate: Candidate) -> None:
    method = record["candidate"]["method_name"]
    score = record["score"]
    runs = record["evaluation"]["runs"]
    print(
        f"[{record['iteration']:04d}] method={method} score={score:.6f} "
        f"metric={record['evaluation']['metric_name']} runs={runs}"
    )
    print(f"       best={best_score:.6f} method={best_candidate.method_name} key={best_candidate.key()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimise llama.cpp speculative decoding parameters")
    parser.add_argument("--llama-cli", required=True, type=Path, help="Path to llama-cli executable")
    parser.add_argument("--trials-per-setting", required=True, type=int, help="Number of trials per candidate setting")
    parser.add_argument("--prompts", required=True, type=Path, help="Path to newline-separated prompts file")
    parser.add_argument("--search-space", type=Path, default=None, help="Path to JSON search space definition")
    parser.add_argument("--progress-file", type=Path, default=Path("spec_opt_progress.json"), help="Progress JSON file")
    parser.add_argument("--max-iterations", type=int, default=40, help="Maximum number of candidates to evaluate")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    if args.trials_per_setting <= 0:
        raise ValueError("--trials-per-setting must be > 0")
    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be > 0")
    if not args.llama_cli.exists():
        raise FileNotFoundError(f"llama-cli not found: {args.llama_cli}")

    prompts = load_prompts(args.prompts)
    search_space = load_search_space(args.search_space)
    rng = random.Random(args.seed)

    progress_path = args.progress_file
    progress = load_progress(progress_path)
    history: List[Dict[str, Any]] = []
    evaluated_keys = set()
    best_score = float("-inf")
    best_candidate: Optional[Candidate] = None

    warmup = min(args.max_iterations, max(MIN_WARMUP_ITERATIONS, search_space.parameter_count * WARMUP_MULTIPLIER))

    if progress:
        print(f"Resuming from progress file: {progress_path}")
        for item in progress.get("history", []):
            history.append(item)
            evaluated_keys.add(item["candidate_key"])
            if item["score"] > best_score:
                best_score = item["score"]
                best_candidate = candidate_from_dict(item["candidate"])

    start_iteration = len(history)
    if not progress:
        print(f"Starting optimization. warmup={warmup} max_iterations={args.max_iterations}")
    else:
        print(f"Continuing optimization from iteration {start_iteration}, best score={best_score:.6f}")

    for iteration in range(start_iteration, args.max_iterations):
        if best_candidate is None or iteration < warmup:
            candidate = search_space.random_candidate(rng)
            while candidate.key() in evaluated_keys:
                candidate = search_space.random_candidate(rng)
        else:
            candidate = choose_candidate(rng, search_space, history, evaluated_keys, best_candidate)

        evaluation = evaluate_candidate(args.llama_cli, search_space, candidate, prompts, args.trials_per_setting)
        score = evaluation["score"]
        features = search_space.candidate_features(candidate)
        key = candidate.key()
        evaluated_keys.add(key)

        record = {
            "iteration": iteration + 1,
            "timestamp": utc_now(),
            "candidate": candidate_to_dict(candidate),
            "candidate_key": key,
            "features": features,
            "score": score,
            "evaluation": evaluation,
            "cli_arguments": search_space.to_cli_arguments(candidate),
        }
        history.append(record)

        if score > best_score:
            best_score = score
            best_candidate = candidate

        if best_candidate is None:
            raise RuntimeError("internal error: best candidate missing after evaluation")
        print_live_stats(record, best_score, best_candidate)

        payload = {
            "version": 1,
            "updated_at": utc_now(),
            "settings": {
                "llama_cli": str(args.llama_cli),
                "trials_per_setting": args.trials_per_setting,
                "prompts": str(args.prompts),
                "max_iterations": args.max_iterations,
                "seed": args.seed,
            },
            "search_space": search_space.to_progress_dict(),
            "history": history,
            "best": {
                "score": best_score,
                "candidate": candidate_to_dict(best_candidate),
                "cli_arguments": search_space.to_cli_arguments(best_candidate),
            },
        }
        atomic_write_json(progress_path, payload)

    if best_candidate is None:
        raise RuntimeError("internal error: no candidate was evaluated")
    print("\nOptimization complete.")
    print(f"Best score: {best_score:.6f}")
    print(f"Best method: {best_candidate.method_name}")
    print("Best command arguments:")
    print(" ".join(search_space.to_cli_arguments(best_candidate)))


if __name__ == "__main__":
    main()
