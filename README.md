<img width="1280" height="1398" alt="fig1" src="https://github.com/user-attachments/assets/0f05c5c0-1bc6-4278-bcdb-b864c0cff1a1" />  <br />

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

<img width="3001" height="1801" alt="fig12" src="https://github.com/user-attachments/assets/0ad24acd-1fea-4a0e-a1b7-76b09944f146" />  <br />

## Test:
```
cd test
python test_lateral.py
```
