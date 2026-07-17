# HighwayEnv-NGSIM

Independent simulator and imitation-learning fork used by the
validation-first interpretability project. This repository owns `highway_env`,
`scripts_gail`, dataset preparation, simulator diagnostics, tests, and Slurm
launchers. It remains a separate Git repository when checked out below
`components/HighwayEnv-NGSIM` in the parent project.

## Environment

Use the existing `ngsim_env` environment and install this checkout in editable
mode:

```bash
conda activate ngsim_env
python -m pip install -e '.[training,testing]'
python -m pytest
```

Supported Python versions are 3.10 through 3.12. Entrypoints use installed
modules and do not require `PYTHONPATH` changes.

## Layout

- `highway_env/`: simulator, replay environment, and NGSIM data utilities.
- `scripts_gail/`: BC, GAIL, AIRL, and IQ-Learn training code.
- `scripts_setup/`: dataset preparation commands.
- `scripts_env_test/`: simulator and training diagnostics.
- `hpc/slurm/`: Linux HPC launchers.
- `tests/`: fork-owned unit and integration tests.

The validation-first recurrent IQ-Learn workflow, evidence, and promotion gates
are documented in the study memory
[`IQ_LEARN_VALIDATION.md`](../../memory/autoregressive_policy_comparison/IQ_LEARN_VALIDATION.md).

## Data setup

### NGSIM data
1. Download the raw NGSIM data from the [link](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj/about_data), the download process may take some time. 
2. Place the csv trajectory file inside the `raw_data` folder. If you do not change the name of the raw Data, you can just run:
```
python -m scripts_setup.dump_data_ngsim raw_data/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv
```
in development:
```
python -m scripts_setup.dump_data_time_ngsim raw_data/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv
```
#### 2.3.2. Morinomiya Datasetup Data setup
