# Installation Guide

1. Create conda env 
```
conda create -n EPQ python=3.7.16
```
2. Activate conda env
```
conda activate EPQ
```
3. Install pytorch (with GPU)
```
conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
or (with CPU, Not Recommend)
```
conda install pytorch==1.12.1 torchvision=0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
```
4. Install gym
```
pip install gym[all]==0.17.2
```
5. Install d4rl from https://github.com/rail-berkeley/d4rl

6. Install other packages
```
pip install h5py tqdm pyyaml python-dateutil matplotlib gtimer scikit-learn
  numba==0.56.2 path.py==12.5.0 patchelf==0.15.0.0 joblib==1.2.0 gtimer python-dateutil matplotlib scikit-learn wandb
```

# Run EPQ

This codebase is built on rlkit (https://github.com/vitchyr/rlkit/), and implements CQL (https://github.com/aviralkumar2907/CQL). To run our code, follow the installation instructions for rlkit as shown below, then install D4RL(https://github.com/rail-berkeley/d4rl).

Then we can run EPQ with an example as follow :

1. First, train VAE for the behavior policy
```
python behavior_cloning.py --env=halfcheetah-medium-v2
```

2. After training the behavior model, we can run EPQ by executing :
```
python EPQ_main.py --env=halfcheetah-medium-v2
```


# Detailed Installation Guide for d4rl
1. Clone the git repository
``
git clone https://github.com/rail-berkeley/d4rl.git
``
2. Move to d4rl directory
``
cd d4rl
``
3. Install d4rl 
``
pip install -e .
``





