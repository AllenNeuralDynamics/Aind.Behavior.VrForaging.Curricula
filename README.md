# Aind.Behavior.VrForaging.Curricula

![CI](https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging.Curricula/actions/workflows/cicd.yml/badge.svg)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)


A repository of curricula for [VR foraging task](https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging).

## How to run a specific curriculum?

Curricula are modules of the main package: `aind_behavior_vr_foraging_curricula.<curriculum_name>`. Each curriculum implements a uniform command-line interface:

The interface can be access by running the module or python script directly (I will consider the `template` curriculum as an example):

```bash
uv run python -m aind_behavior_vr_foraging_curricula.template -h
```

This command will print out the available interface for the CLI. To run a curriculum iteration, you can use the `run` subcommand:

```bash
uv run python -m aind_behavior_vr_foraging_curricula.template run -h
```

The following arguments are available for the `run` subcommand:

* `--data-directory`: Path to the session data directory that will be used to calculate metrics (required)
* `--input-trainer-state`: Path to a deserialized json file with the current trainer state (required)
* `--mute-suggestion`: Disables the suggestion output (optional)
* `--output-suggestion`: A path to save the serialized suggestion (optional)

For a quick "demo" to ensure everything is working, you can run:

```bash
uv run python -m aind_behavior_vr_foraging_curricula.template run --data-directory "demo" --input-trainer-state "foo.json"
```

## Style guide

To keep things clear, I suggest the following naming convention:

* **Policies** should start with `p_` (e.g., `p_identity_policy`)
* **Policy transitions** should start with `pt_`
* **Stages** should start with `s_` (e.g., `s_stage1`)
* **Stage transitions** should start with `st_` and should be named after the stages they transition between (e.g., `st_s_stage1_s_stage2`)

Define the following modules:

* **metrics**: Defines (or imports) metrics classes and how to calculate them from data
* **stages**: Defines the different stages of the VR foraging task. This includes task settings and, optionally, policies
* **curriculum**: Defines the transitions between the stages and generate entry point to the application

## Contributors

Contributions to this repository are welcome! However, please ensure that your code adheres to the recommended DevOps practices below:

### Linting

We use [ruff](https://docs.astral.sh/ruff/) as our primary linting tool.

### Testing

Attempt to add tests when new features are added.
To run the currently available tests, run `uv run pytest` from the root of the repository.

### Lock files

We use [uv](https://docs.astral.sh/uv/) to manage our lock files and therefore encourage everyone to use uv as a package manager as well.