

Installation

conda install cvxpy

Installing pytorch-geometric

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyg=*=*cu* -c pyg
conda install pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv -c pyg








Install PyTorch

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyg=*=*cu* -c pyg
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
conda install pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv -c pyg

First check version of torch, cuda:

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"

pip install torch_geometric==2.3.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html



conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia




If using PyTorch 2.0.0 and cuda 11.7, the following command works:

`conda install pyg -c pyg`

otherwise find the right command here: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+cu.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html