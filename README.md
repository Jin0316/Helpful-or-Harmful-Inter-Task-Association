# Helpful or Harmful: Inter-Task Association in Continual Learning

The official code for Helpful or Harmful: Inter-Task Association in Continual Learning [ECCV 22]

<div align="center">
  
![ECCV](https://img.shields.io/badge/ECCV-2022-blue)

![h2](images/H_2_ECCV_2022.png)

</div>

## Description
A continual learning framework described in the following paper [Link]. 

Note, this code will be dynamically updated.

This repository currently contains code to run experiments of H<sup>2.

## Run the code

This repository currently supports Split CIFAR-10 experiment in original paper.
  
Please, change the hyper-parmeters in the corresponding file (config/CONFIG.py) if needed.
  
```bash
python3 main.py
```
  
## Requirements 
  
Please, find a list with requiered packages and versions in requirements.txt

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
