# ScIReN

This code implements the method and experiments in "Scientifically-Interpretable Reasoning 
Network (ScIReN): Uncovering the Black Box of Nature".


## Dataset download

The input data can be downloaded [here](https://osf.io/a643m/?view_only=f1682a62cdf84900a57b6130174ec22e). Navigate to `files`, download the file, and unzip it in the `BINNS` root directory.

Here are commands that will download and unzip:
```
wget https://osf.io/download/682ed7b80f8ae3415deac68b/?view_only=f1682a62cdf84900a57b6130174ec22e
unzip 'index.html?view_only=f1682a62cdf84900a57b6130174ec22e'
```

## Git submodules

NOTE: Reviewers - ignore this section, since the submodules are already included in the zip.

The KAN and q10hybrid folders are submodules. To clone them, after cloning the main repo, use

```
git checkout sciren
git submodule update --init --recursive
```

To keep submodules up-to-date with their remote versions, cd to the submodule directory and run
```
git fetch
git merge origin/main
```
or, from the main directory:
```
git submodule update --remote
```

Pulling upstream changes from project remote: suppose a collaborator made changes to a submodule and I need to pull them.
```
git pull
git submodule update --init --recursive
```
or
```
git pull --recurse-submodules
```

To push changes, first push changes to the submodule, then do `git add <submodule>` in the main directory, commit it, then run
```
git push --recurse-submodules=check
```


## Installation Instructions

First ensure that submodules have been downloaded (previous section): `src_binns/q10hybrid` and `src_binns/pykan` should not be empty.

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

Install required packages.
```
pip install -r requirements.txt
```

ALTERNATIVE: If installing `requirements.txt`, you can try installing packages manually. The key packages are  PyTorch, Pytorch Lightning, Numpy, Scipy, Pandas, matplotlib, scikit-learn, geopandas, mat73, netCDF4, and alibi. For example, you can try these commands:

```
pip install numpy scipy pandas matplotlib scikit-learn geopandas mat73 xarray netCDF4 alibi joblib
pip install torch torchvision torchaudio 
pip install lightning
cd src_binns/pykan
pip install -e .
cd ../q10hybrid
pip install -e .
```


## Running Instructions

### Tables 1-2: Ecosystem Respiration

See `src_binns/q10hybrid/README.md` for the commands to reproduce Table 1-2.

### Table 3: Experiments with synthetic labels (predicting 4 most sensitive parameters)

These commands will run the experiments on Slurm. If you do not have slurm, you can replace `sbatch` with `bash`, or run them as
`./slurm_scripts/3a_synthetic_purenn.sh`. (Also, make sure execute permissions are granted, e.g. `chmod +x slurm_scripts/3a_synthetic_purenn.sh`).

```
sbatch slurm_scripts/3a_synthetic_purenn.sh
sbatch slurm_scripts/3b_synthetic_blackboxhybrid.sh
sbatch slurm_scripts/3c_synthetic_blackboxhybrid_hardsigmoid.sh
sbatch slurm_scripts/3d_synthetic_sciren_1layer.sh
```

Note that you should change `--num_CPU` to the number of CPUs you want to use (or GPUs if available).

Each run creates a folder inside `OUTPUT_DATA/neural_network`. Inside `OUTPUT_DATA/neural_network`, there will also be a file
called  `results_summary_{NOTE}.csv`, which contains a row for each run with that note.


### Table 4: Experiments with real labels 

The following commands run the experiments on Slurm; see the above section for notes.
```
sbatch slurm_scripts/4a_real_purenn.sh
sbatch slurm_scripts/4b_real_blackboxhybrid.sh
sbatch slurm_scripts/4c_real_blackboxhybrid_hardsigmoid.sh
sbatch slurm_scripts/4d_real_sciren_1layer.sh
sbatch slurm_scripts/4e_real_sciren_2layer.sh
```



## Code summary

* `slurm_scripts/*.sh`: contains scripts that run each method. Change `--num_CPU` to the number of GPUs, or CPUs if no GPUs are available.
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
* `src_binns/fun_matrix_clm5_experimental.py`: process-based model
    - Estimates amount of carbon in 140 pools (20 depths * 7 pools per layer)
    - `a_matrix`: 140x140 matrix, containing horizontal transfers between pools of the same layer. `A[i, j]` (if `i != j`) is the flux from pool j to i. `A[i, i]` is the total flux leaving pool i.
    - `kk_matrix`: 140x140 matrix. `KK[i, i]` is the decomposition rate for pool i. Nondiagonal entries are zero.
    - `tri_matrix`: 140x140 matrix, containing vertical transfers. `Tri[i, j]` (if `i != j`) is the flux from pool j to i. `Tri[i, i]` is the total flux leaving pool i. Only the three middle diagonals contain nonzero entries, meaning that there is only transfer between adjacent layers of the same pool type.
    - Each of these has a vectorized and non-vectorized implementation. They should produce the same result, vectorized is faster.
* `src_binns/fun_matrix_clm5_vectorized_bulk_converge.py` is similar to above, but also outputs additional quantities (various combinations of parameters) that are used in final visualizations
* `losses.py`: code for loss functions

## Data Notes

The covariates and biogeochemical parameters are described in the Appendix.


## Licenses

This codebase is built on the following public repositories:

Q10Hybrid (Apache License): https://github.com/bask0/q10hybrid
pykan (MIT License): https://github.com/KindXiaoming/pykan

In addition, the datasets used in the CLM5 experiments are drawn from this paper, which cites the original data sources (such as WoSIS and MODIS NPP):

Tao F, Zhou Z, Huang Y, Li Q, Lu X, Ma S, Huang X, Liang Y, Hugelius G, Jiang L, Doughty R, Ren Z and Luo Y (2020) Deep Learning Optimizes Data-Driven Representation of Soil Organic Carbon in Earth System Model Over the Conterminous United States. Front. Big Data 3:17. doi: 10.3389/fdata.2020.00017
Link: https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2020.00017/full




## FEEL FREE TO IGNORE: Additional tips / notes

Do this to avoid commiting images in Jupyter Notebooks in git: https://stackoverflow.com/a/74753885

Save pip environment:
```
pip freeze > requirements.txt
```

### Git submodule notes

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


### Alternate commands to install


You need to download the PRODA parameters from [this link](https://drive.google.com/file/d/1AGDlybz35n3gHqyNilVthVzOaBkNZAXr/view?usp=sharing), place in the `ENSEMBLE/INPUT_DATA` directory, and unzip. One way to download this is using `gdown`:
```
pip install gdown
cd ENSEMBLE/INPUT_DATA
gdown 1AGDlybz35n3gHqyNilVthVzOaBkNZAXr
unzip PRODA_Results_Subset.zip
```


