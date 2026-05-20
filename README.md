# LLamaParamOptimiser

`optimise_speculative.py` discovers good speculative decoding command line settings for `llama.cpp` (`llama-cli`) while minimizing expensive trials.

## Features

- Optimizes numeric and categorical parameters (for example `--spec-draft-type-k` values).
- Supports multiple speculative method definitions, each with its own parameter set.
- Optimizes drafter combinations and order (`drafter_chains`) as another search dimension.
- Writes live progress to a JSON file and resumes from it automatically.
- Prints live best-score updates as optimization runs.

## Usage

```bash
python optimise_speculative.py \
  --llama-cli /path/to/llama-cli \
  --trials-per-setting 2 \
  --prompts /path/to/prompts.txt \
  --max-iterations 40 \
  --progress-file progress.json \
  --search-space /path/to/search_space.json
```

Required inputs:

- `--llama-cli`: path to `llama-cli`
- `--trials-per-setting`: number of repeated trials per candidate setting
- `--prompts`: file containing one test prompt per non-empty line

`--search-space` is optional; if omitted, a built-in example search space is used.

## Search space format

Define speculative method names, their `--spec-type`, method-specific parameters, and drafter combinations:

```json
{
  "prompt_argument": "--prompt",
  "static_arguments": ["-n", "128"],
  "common_parameters": {
    "--threads": { "type": "int", "min": 4, "max": 16, "step": 2 }
  },
  "spec_methods": [
    {
      "name": "k-only",
      "spec_type": "k",
      "drafter_argument": "--spec-drafters",
      "drafter_chains": [["k"]],
      "parameters": {
        "--spec-draft-type-k": {
          "type": "categorical",
          "options": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
        },
        "--spec-draft-max-k": { "type": "int", "min": 2, "max": 16 }
      }
    },
    {
      "name": "k-then-eagle",
      "spec_type": "combo",
      "drafter_argument": "--spec-drafters",
      "drafter_chains": [["k", "eagle"], ["eagle", "k"]],
      "parameters": {
        "--spec-draft-type-k": {
          "type": "categorical",
          "options": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
        },
        "--spec-accept-threshold": { "type": "float", "min": 0.1, "max": 0.9 }
      }
    }
  ]
}
```

Parameter types:

- `int`: `min`, `max`, optional `step`
- `float`: `min`, `max`
- `categorical`: `options`
- `flag`: boolean (present/absent switch)
