## Environmental configuration:
install miniconda (https://www.anaconda.com/docs/getting-started/miniconda/main)  
`conda create -n magdrone python=3.10`
`conda activate magdrone`
install requirements:  
`pip install gymnasium`
`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
`pip install numpy matplotlib pandas scipy`
`pip install stable-baselines3[extra]`
`pip install control`
`pip install tensorboard`
## Train:
```
cd train
python train_ppo_lateral.py
```
## Test:
```
cd test
python test_lateral.py
```
