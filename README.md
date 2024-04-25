# DSM-GNN
A PyTorch implementation of DSM-GNN "Dual-Space MLPs for Modeling Both Homophily and Heterophily via Distillation". <br>
code is coming soon
# Environment Settings
This implementation is based on Python3. To run the code, you need the following dependencies: <br>
* torch==1.8.1
* torch-geometric==1.7.2
* scipy==1.2.1
* numpy==1.19.5
* tqdm==4.59.0
* seaborn==0.11.2
* scikit-learn==0.24.2
* CUDA Version: 11.0
# Datasets
The data folder contains four homophilic benchmark datasets(Cora, Citeseer, Pubmed, Computers) from [CPF](https://github.com/BUPT-GAMMA/CPF/tree/master) and  four heterophilic datasets(Chameleon, Squirrel, Cornell, Texas) from [T2-GNN](https://github.com/jindi-tju/T2-GNN). 
# Run an experiment:
    $ python train_teacher.py --model GCN --dataset cora
    $ python train_student.py --teacher GCN --dataset cora
# Examples
 Training a model on the default dataset.  
![image](https://github.com/GGA23/DSM-GNN/blob/main/KD-demo-cora.gif)
# Experiments on supplementary datasets
Due to the limited space of the original manuscript, some experimental results on other datasets are presented here.
![image](https://github.com/GGA23/DSM-GNN/blob/main/experiment.JPG)
# Baselines links
* [H2GCN](https://github.com/GitEventhandler/H2GCN-PyTorch)
* [FAGCN](https://github.com/bdy9527/FAGCN)
* [HopGNN](https://github.com/JC-202/HopGNN)
* [GPRGNN](https://github.com/jianhao2016/GPRGNN)
* [BernNet](https://github.com/ivam-he/BernNet)
* [CPF](https://github.com/BUPT-GAMMA/CPF/tree/master)
* [FF-G2M](https://github.com/LirongWu/FF-G2M)
* [NOSMOG](https://github.com/meettyj/NOSMOG)
* The implementations of others are taken from the Pytorch Geometric library
# Acknowledgements
The code is implemented based on [Extracting Low-/High- Frequency Knowledge from Graph Neural Networks and
Injecting it into MLPs: An Effective GNN-to-MLP Distillation Framework](https://github.com/LirongWu/FF-G2M).
