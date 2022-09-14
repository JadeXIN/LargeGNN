# LargeGNN
Implementation of paper "Large-scale Entity Alignment via Knowledge Graph Merging, Partitioning and Embedding", accepted by CIKM 2022.

# [Ensemble Semi-supervised Entity Alignment via Cycle-teaching (2022 AAAI)](https://arxiv.org/pdf/2208.11125.pdf)

> Entity alignment is a crucial task in knowledge graph fusion. However, most entity alignment approaches have the scalability problem. Recent methods address this issue by dividing large KGs into small blocks for embedding and alignment learning in each. However, such a partitioning and learning process results in an excessive loss of structure and alignment. Therefore, in this work, we propose a scalable GNN-based entity alignment approach to reduce the structure and alignment loss from three perspectives. First, we propose a centrality-based subgraph generation algorithm to recall some landmark entities serving as the bridges between different subgraphs. Second, we introduce self-supervised entity reconstruction to recover entity representations from incomplete neighborhood subgraphs, and design cross-subgraph negative sampling to incorporate entities from other subgraphs in alignment learning. Third,
during the inference process, we merge the embeddings of subgraphs to make a single space for alignment search. Experimental results on the benchmark OpenEA dataset and the proposed large DBpedia1M dataset verify the effectiveness of our approach.


## Overview

We build our model based on [Python](https://www.python.org/) and [Tensorflow](https://www.tensorflow.org/). 

### Getting Started
Please unzip the src.zip and place it in your project folder. You can test our implementation by runing main_ea.py or .sh files. Please modify the path to your configuration at config folder.

#### Dependencies
You can refer to the requirements of [Dual-AMN](https://github.com/MaoXinn/Dual-AMN) to start the implementation of ours.

Python 3.x (tested on Python 3.8)
matplotlib==3.1.1
networkx_metis==1.0
scipy==1.3.1
tqdm==4.60.0
tensorflow==2.0.0
networkx==2.3
Keras==2.4.3
dgl==0.5.3
numpy==1.16.2
torch==1.5.0
torch_sparse==0.6.3
xgboost==1.0.1
torch_scatter==2.0.4
faiss==1.5.3
PYNVML==11.4.1
scikit_learn==1.0.2
torch_geometric==2.0.3

#### Large-scale dataset


## Acknowledgement
We refer to the codes of these repos: [OpenEA](https://github.com/nju-websoft/OpenEA), Dual-AMN(https://github.com/MaoXinn/Dual-AMN), [LargeEA](https://github.com/ZJU-DAILY/LargeEA), [LIME](https://github.com/DexterZeng/LIME), [ClusterGCN](https://github.com/benedekrozemberczki/ClusterGCN).
Thanks for their great contributions!

