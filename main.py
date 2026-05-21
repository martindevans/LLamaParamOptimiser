import argparse
import optuna
import trial
from functools import partial

def main():
    parser = argparse.ArgumentParser(description="LLamaParamOptimiser CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start command
    parser_start = subparsers.add_parser("run", help="Run an optimize study")
    parser_start.add_argument("--name", required=True, help="Name of this study")
    parser_start.add_argument("--state", required=True, help="Path to save the state")
    parser_start.add_argument("--llama_cli_path", required=True, help="Path to llama_cli")
    parser_start.add_argument("--model_path", required=True, help="Path to a gguf model")
    parser_start.add_argument("--prompts_file", required=True, help="Path to file of prompts (one per line)")
    parser_start.add_argument("--trials", required=True, help="How many trials to run right now")
    parser_start.add_argument("--repeats", required=True, help="How many times to repeat each trial")
    parser_start.set_defaults(func=run_optimization)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

def run_optimization(args):
    """
    Initializes a new study (or loads it if it exists) and runs some trials. prints the best result at the end
    """

    # Load prompts
    prompts = open(args.prompts_file, "r").readlines()

    study = optuna.create_study(
        study_name=args.name,
        storage=f"sqlite:///{args.state}.db",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            group=True,
        ),
        pruner=optuna.pruners.HyperbandPruner()
    )

    wrapped = partial(
        trial.objective,
        llama_cli_path=args.llama_cli_path,
        model_path=args.model_path,
        prompts=prompts,
        repeats=int(args.repeats)
    )

    study.optimize(
        wrapped,
        int(args.trials),
        show_progress_bar=True,
        n_jobs=1,
    )

    print(study.best_params)
    print(study.best_value)
    print(len(study.trials))

if __name__ == "__main__":
    main()
