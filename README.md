# Aind.Behavior.VrForaging.Curricula

A repository of curricula for VR foraging task.

The main repository of the platform is located [here](https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging).

## Style guide

To keep things clear, I suggest the following naming convention:

- **Policies** should start with `p_` (e.g., `p_identity_policy`)
- **Policy transitions** should start with `pt_`
- **Stages** should start with `s_` (e.g., `s_stage1`)
- **Stage transitions** should start with `st_` and should be named after the stages they transition between (e.g., `st_s_stage1_s_stage2`)

Define the following modules:
 - **metrics**: Defines (or imports) metrics classes and how to calculate them from data
 - **stages**: Defines the different stages of the VR foraging task. This includes task settings and, optionally, policies
 - **curriculum** Defines the transitions between the stages and generate entry point to the application