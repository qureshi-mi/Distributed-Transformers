# P2P-FT and Local-FT

The code implements the experiments performed in [paper]. It demonstrates the performance comparison of P2P-FT with locally trained Local-FT at each node. We include the experiments for the classification of Flowers, Pets, CIFAR-10, and CIFAR100 datasets.


## Dependencies and Setup
All code runs on Python 3.6.7 with Pytorch and timm libraries installed.


## Running Experiments
There are three main folders:
1) vit:  for distributed fine-tuning of vit_small_patch16_224 model.
2) deit: for distributed fine-tuning of deit_small_patch16_224 model.
3) swin: for distributed fine-tuning of swin_small_patch4_window7_224.ms_in22k model.


In each folder, there are four different datasets:
1) Flowers:  	Oxford-Flowers dataset comprises images belonging to 102 flower categories.
(This dataset can be downloaded from: robots.ox.ac.uk/~vgg/data/flowers/102/)

2) Pets:  	Oxford-Pets dataset consists of 37 categories of dogs and cats with roughly 200 images for each class.
(This dataset can be downloaded from: robots.ox.ac.uk/~vgg/data/pets/)

3) CIFAR-10:  	CIFAR-10 dataset contains 60,000 images belonging to 10 different categories. 	(This dataset is automatically downloaded in code)

4) CIFAR-100:  	CIFAR-100 dataset consists of 100 classes, each containing 600 images.		(This dataset is automatically downloaded in code)


In each of the above, there are two main files:
1) Local_FT.py:   Fine-tune the transformer model available at each node independently using the local dataset
2) P2P_FT.py:     Fine-tune the transformer model available at each node using P2P-FT method using the local dataset

The above files generate the accuracy of each method for every node and save it in ".txt" file in the "results" folder.

The user can run "Local_FT.py" and "P2P_FT.py" from any of the sub-folders and then check the accuracy.
