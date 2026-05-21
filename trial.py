import subprocess
import re
import optuna
import statistics
import random

def run_llama_cli(args_list):
    try:
        result = subprocess.run(
            args_list,
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8"
        )

        if result.returncode != 0:
            return float("-inf")

        # Regex to find generation speed, e.g., "Generation: 322.8 t/s"
        match = re.search(r"Generation:\s+([\d.]+)\s+t/s", str(result.stdout))
        if match:
            return float(match.group(1))

        return float("-inf")

    except subprocess.TimeoutExpired:
        return float("-inf")

def score_func(scores):
    return statistics.harmonic_mean(scores)

def objective(trial, llama_cli_path, model_path, prompts, repeats=10, n_tokens=1024):

    spec_type = trial.suggest_categorical(
        "spec_type",
        [ "none", "ngram-mod" ]
    )

    prompts = prompts.copy()
    random.shuffle(prompts)

    # Build CLI args
    args = [
        llama_cli_path,
        "--model", str(model_path),
        "-n", str(n_tokens),
        "-c", str(int(n_tokens * 1.5)),
        "--single-turn",
        "--no-display-prompt",

        "--spec-type", str(spec_type),
    ]

    if spec_type in [ "ngram-mod" ]:
        spec_ngram_mod_n_match = trial.suggest_int(
            "spec_ngram_mod_n_match",
            1, 64
        )

        spec_ngram_mod_n_min = trial.suggest_int(
            "spec_ngram_mod_n_min",
            1, 64
        )

        spec_ngram_mod_n_max = trial.suggest_int(
            "spec_ngram_mod_n_max",
            1, 64
        )

        args.extend([
            "--spec-ngram-mod-n-match", str(spec_ngram_mod_n_match),
            "--spec-ngram-mod-n-min", str(spec_ngram_mod_n_min),
            "--spec-ngram-mod-n-max", str(spec_ngram_mod_n_max),
        ])
        
    
    if spec_type in [ "draft-mtp" ]:
        spec_draft_type_k = trial.suggest_categorical(
            "spec_draft_type_k",
            [ "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1" ]
        )

        spec_draft_type_v = trial.suggest_categorical(
            "spec_draft_type_v",
            [ "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1" ]
        )

        spec_draft_n_max = trial.suggest_int(
            "spec_draft_n_max",
            1, 32
        )

        args.extend([
            "--spec-draft-type-k", str(spec_draft_type_k),
            "--spec-draft-type-v", str(spec_draft_type_v),
            "--spec-draft-n-max", str(spec_draft_n_max),
        ])

    if trial.should_prune():
        raise optuna.TrialPruned()

    trial_count = 0
    scores = []
    for step in range(repeats):
        for prompt in prompts:
            trial_args = args.copy()
            trial_args.extend([
                "--prompt", str(prompt),
            ])

            scores.append(max(0, run_llama_cli(trial_args)))

            trial.report(score_func(scores), trial_count)
            trial_count += 1

            if trial.should_prune():
                raise optuna.TrialPruned()

    return score_func(scores)