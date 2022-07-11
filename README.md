# Helpful or Harmful: Inter-Task Association in Continual Learning

The official code for Helpful or Harmful: Inter-Task Association in Continual Learning ![ECCV](https://img.shields.io/badge/ECCV-2022-blue) [![PyTorch](https://img.shields.io/badge/pytorch-1.8.0-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

Author: Hyundong Jin and Eunwoo Kim 

In Proc. of the European Conference on Computer Vision (ECCV), 2022 

<div align="center">

![h2](images/H_2_ECCV_2022.png)

</div>

|               |   T1   |   T2   |   T3   |   T4   |   T5   |  Avg acc  |
|:-------------:|:------:|:------:|:------:|:------:|:------:|:---------:|
| H2            |  98.9  |  94.1  |  96.4  | 98.90  |  98.12 |   97.27   |

## Run the code

This repository currently supports the Split CIFAR-10 experiment in the original paper.
  
You can change the hyper-parmeters in the corresponding file (config/CONFIG.py) if needed.
  
```bash
python3 main.py
```
  
## Requirements 
  
Please, find a list with required packages and versions in requirements.txt

## Code list 

```bash
config
  ㄴ CONFIG.py
data_loader
  ㄴ split_cifar10_data.py
model
  ㄴ resnet18.py
src 
  ㄴ main.py
  ㄴ manager.py
  ㄴ pruner.py
```
