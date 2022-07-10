# Helpful-or-Harmful:Inter-Task Association for Continual Learning

The official code for ECCV22: Helpful or Harmful: Inter-Task Association for Continual Learning

<div align="center">
  
![ECCV](https://img.shields.io/badge/ECCV-2022-blue)

![h2](images/H_2_ECCV_2022.png)

</div>


## Run the train code 
The code supports the Split CIFAR-10 experiment.

```bash
python3 main.py
```

## Requirements 
```bash
tqdm 4.62.3
pytorch 1.8.0 
```

## Code list 

```bash
config
  ㄴ config.py
data_loader
  ㄴ split_cifar10_data.py
model
  ㄴ resnet18.py
src 
  ㄴ main.py
  ㄴ manager.py
  ㄴ pruner.py
```
