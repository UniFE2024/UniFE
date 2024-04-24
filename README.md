# Unaligned Federated Knowledge Graph Embedding	
Source Code for the Check of Reviewers [Dual anonymous version]. 

More files are being organized, and we will provide a more detailed version after acceptance.
## Requirements
```
torch==1.13.1+cu117
torch-scatter==2.1.1+pt113cu117
torch-sparse==0.6.17+pt113cu117
torchaudio==0.13.1+cu117
torchmetrics==0.11.4
torchtext==0.14.1
torchvision==0.14.1+cu117
```
## Datasets
The datasets FB15k-237-C3,-C5,-C10 are available at https://doi.org/10.5281/zenodo.7601676.
The dataset NELL-995-Fed3 is available at https://github.com/AnselCmy/FedE.
Download and put in ./data folder.

## Run
```
python main.py --data_group C3FL --setting unife --local_epoch 3 --nembds_mode single --lr 0.001 --dim 128 --k_w 8 --k_h 16 --score_func conve --local_batch_size 1024 --use_structure
python main.py --data_group C5FL --setting unife --local_epoch 3 --nembds_mode single --lr 0.001 --dim 128 --k_w 8 --k_h 16 --score_func conve --local_batch_size 1024 --use_structure
python main.py --data_group C10FL --setting unife --local_epoch 3 --nembds_mode single --lr 0.001 --dim 128 --k_w 8 --k_h 16 --score_func conve --local_batch_size 1024 --use_structure
python main.py --data_group NELL3 --setting unife --local_epoch 3 --nembds_mode single --lr 0.001 --dim 128 --k_w 8 --k_h 16 --score_func conve --local_batch_size 1024 --use_structure
```

## More Parameters
More parameters can see in main.py .

