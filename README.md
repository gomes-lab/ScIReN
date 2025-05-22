# BINNS

This code implements the method proposed in "Biogeochemistry-Informed Neural Network (BINN) for Improving Accuracy of Model
Prediction and Scientific Understanding of Soil Organic Carbon" (Xu et al. 2025). 

## Installation Instructions

The key packages to install are PyTorch, PyTorch Geometric, Numpy, Scipy, Pandas, matplotlib, scikit-learn, geopandas, mat73, and netCDF4. Here are instructions to install the necessary packages:

Create a virtual env called ".venv", and activate it
```
cd src_binns
python -m venv .venv
source .venv/bin/activate
```

Install pip (upgrade if needed)
```
python3 -m pip install --upgrade pip
```

Install required packages
```
pip install -r requirements.txt
```

NOTE: If this did not work, you can try installing packages manually, e.g.
```
pip install numpy scipy pandas matplotlib scikit-learn geopandas mat73 xarray netCDF4 alibi joblib
pip install torch torchvision torchaudio 
pip install torch_geometric
cd src_binns/pykan
pip install -e .
cd ../q10hybrid
pip install -e .
```

To use graph neural network variant (work in progress), you may need to install these libraries as well. These are not needed for normal BINN.
```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

The input data can be downloaded [here](https://drive.google.com/file/d/1fQYA2xxeSdu4MLOU25Ej-bQ6eUiFWvXK/view?usp=sharing). You can unzip the file in the `BINNS` root directory. One way to download the data from the commandline is using `gdown`:

```
pip install gdown
gdown 1fQYA2xxeSdu4MLOU25Ej-bQ6eUiFWvXK
unzip BINN_input_data.zip
```
### KAN

If you want to use the KAN model, please cd into the subfolder and install its requirements and the package itself.
```
cd src_binns/pykan
pip install -r requirements.txt
pip install -e .
```

## Git submodules

The KAN and q10hybrid folders are submodules. To clone them use

```
git submodule update --init --recursive
```

To get changes from the remote submodules, cd to the submodule directory and run
```
git fetch
git merge origin/main
```
or
```
git submodule update --remote
```

To push changes, first push changes to the submodule, then do `git add <submodule>` in the main directory, commit it, then run
```
git push --recurse-submodules=check
```


## Running Instructions

### Table 3: Experiments with synthetic labels (predicting 4 most sensitive parameters)

These commands will run the experiments on Slurm. If you do not have slurm, you can replace `sbatch` with `bash`, or run them as
`./slurm_scripts/3a_synthetic_purenn.sh`. (Also, make sure execute permissions are granted, e.g. `chmod +x slurm_scripts/3a_synthetic_purenn.sh`).

```
sbatch slurm_scripts/3a_synthetic_purenn.sh
sbatch slurm_scripts/3b_synthetic_blackboxhybrid.sh
sbatch slurm_scripts/3c_synthetic_blackboxhybrid_hardsigmoid.sh
sbatch slurm_scripts/3d_synthetic_sciren_1layer.sh
```

### Table 4: Experiments with real labels 

The following commands run the experiments on Slurm; see the above section for notes.
```
sbatch slurm_scripts/4a_real_purenn.sh
sbatch slurm_scripts/4b_real_blackboxhybrid.sh
sbatch slurm_scripts/4c_real_blackboxhybrid_hardsigmoid.sh
sbatch slurm_scripts/4d_sciren_1layer.sh
sbatch slurm_scripts/4e_sciren_2layer.sh

```


The script `src_binns/run_binn.sh` contains an example of how to train on 4 GPUs from command-line (interactively). Run like this:
```
cd src_binns
chmod +x run_binn.sh  # If execute permission not enabled
./run_binn.sh
```

If running locally on Mac:
```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

The script `src_binns/run_binn_slurm.sh` contains an example of how to train on 4 GPUs on a server with the Slurm scheduler.

The script `src_binns/run_retrieval.sh` runs the retrieval test described in the BINN paper. NOT FULLY TESTED YET.


## Code summary

* `src_binns/run_binn_interactive.sh`: contains command to run BINN training. Change `--num_CPU` to the number of GPUs, or CPUs if no GPUs are available.
    - `--representative_sample` restricts the data to a "representative sample" of ~1000 sites. Useful for initial testing.
    - You can also choose a random subsample with `--n_datapoints 1000`
* `src_binns/binns_DDP.py`: main train script
    - Set `data_dir_input`, `data_dir_output`
    - `current_data_x[:, :, 0, 0]`: input features, shape `[batch, num_features]`
        - Other stuff in this tensor is forcing (e.g. weather) variables
        - `0:12` - each month (averaged across 20 yrs)
        - `0:20` - each soil layer
        - `current_data_*` includes train/val/test. The individual splits are `train_*`, `val_*`, `test_*`.
    - `y`: SOC (soil organic carbon) observations, shape `[batch, 200]`
        - Each site may have up to 200 observations, but usually much less. Extra entries are nan.
    - `z`: depth of each SOC observation, same format as y. `z[i, j]` is the depth of observation `y[i, j]`. Example:
        - `y: [ 50000 20000 10000 nan nan nan ...]`
        - `z: [ 0.1  0.3  0.75 nan nan nan ...]`
    - Model predicts SOC at 20 predefined depths (`zsoi`). For each site, we then linearly interpolate to predict at the specific depths in “z”
    - `c`: longitude/latitude coordinates, shape `[batch, 2]`
    - `current_proda_para`: biogeochemical parameters predicted by the previous PRODA method, shape `[batch, num_parameters]`
    - Search `args.split` for the code that splits data into train/val/test.
    - `predict_data_*` is grid data, where we have input features (x), coordinates (c), and PRODA parameters, but no SOC observations. We use this to make nationwide predictions using a trained model.
* `src_binns/mlp.py`: deep learning models
    - `mlp_wrapper` is "baseline BINN". Usage: `--model new_mlp`
        - MLP: maps input features → biogeochemical parameters
        - Process-based model: maps biogeochemical parameters + forcing → SOC predictions at 20 depths
    - `nn_only` is pure-neural network without the process-based model. Usage: `--model nn_only`
        - Directly maps input features → SOC predictions at 20 depths
* `src_binns/fun_matrix_clm5_vectorized.py`: process-based model
    - Estimates amount of carbon in 140 pools (20 depths * 7 pools per layer)
    - `a_matrix`: 140x140 matrix, containing horizontal transfers between pools of the same layer. `A[i, j]` (if `i != j`) is the flux from pool j to i. `A[i, i]` is the total flux leaving pool i.
    - `kk_matrix`: 140x140 matrix. `KK[i, i]` is the decomposition rate for pool i. Nondiagonal entries are zero.
    - `tri_matrix`: 140x140 matrix, containing vertical transfers. `Tri[i, j]` (if `i != j`) is the flux from pool j to i. `Tri[i, i]` is the total flux leaving pool i. Only the three middle diagonals contain nonzero entries, meaning that there is only transfer between adjacent layers of the same pool type.
    - Each of these has a vectorized and non-vectorized implementation. They should produce the same result, vectorized is faster.
* `src_binns/fun_matrix_clm5_vectorized_bulk_converge.py` is similar to above, but also outputs additional quantities (various combinations of parameters) that are used in final visualizations
* `losses.py`: code for loss functions

## Data Notes

The covariates and biogeochemical parameters are listed in [this document](https://docs.google.com/document/d/1dAlGbuwKkIg7-ai9ZPGSKIP7rKdKj8mUQi29TObQlUI/edit?usp=sharing).

## Additional tips / notes

Do this to avoid commiting images in Jupyter Notebooks in git: https://stackoverflow.com/a/74753885

Save pip environment:
```
pip freeze > requirements.txt
```

## Git submodule notes

Source: https://git-scm.com/book/en/v2/Git-Tools-Submodules

Add existing Git repository as submodule 
```
git submodule add https://github.com/chaconinc/DbConnector
```

Creates a new file `.gitmodules` and a new file for the submodule.

Clone project with submodules:
```
git clone https://github.com/chaconinc/MainProject  # Main project
git submodule update --init --recursive
```

Pull upstream changes from submodule remote: go to submodule directory
```
git fetch
git merge origin/master (or origin/main)
```

Alternative to the above: from the MAIN directory, this command goes into submodules and fetches/updates.
```
git submodule update --remote DbConnector
```

Automatically show submodule changes when you run `git diff`:
```
git config --global diff.submodule log
```

Pulling upstream changes from project remote: suppose a collaborator made changes to a submodule and I need to pull them.
```
git pull
git submodule update --init --recursive
```

Alternative:
```
git pull --recurse-submodules
```

If the other person changed the url of the submodule:
```
# copy the new URL to your local config
$ git submodule sync --recursive
# update the submodule from the new URL
$ git submodule update --init --recursive
```

Pushing: when pushing the main repository, tell Git to always check if submodules have been pushed properly.
```
git config push.recurseSubmodules check
```
This means that if I run `git push`, it actually runs `git push --recurse-submodules=check`

If the submodule wasn't pushed, go into the submodule and commit/push your local changes.






### (OLD STUFF, CAN PROBABLY IGNORE) Alternate commands to install



You need to download the PRODA parameters from [this link](https://drive.google.com/file/d/1AGDlybz35n3gHqyNilVthVzOaBkNZAXr/view?usp=sharing), place in the `ENSEMBLE/INPUT_DATA` directory, and unzip. One way to download this is using `gdown`:
```
pip install gdown
cd ENSEMBLE/INPUT_DATA
gdown 1AGDlybz35n3gHqyNilVthVzOaBkNZAXr
unzip PRODA_Results_Subset.zip
```


