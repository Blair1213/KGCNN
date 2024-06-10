<h1 align="center">
  [KGCNN] Knowledge Graph Neural Network with Spatial-Aware Capsules for Drug-Drug Interactions Prediction
</h1>

## 👀 Overview of KGCNN

Uncovering novel drug-drug interactions (DDIs) plays a pivotal role in advancing drug development and improving clinical treatment. The outstanding effectiveness of graph neural networks (GNNs) has garnered significant interest in the field of DDI prediction. Consequently, there has been a notable surge in the development of network-based computational approaches for predicting DDIs. However, current approaches face limitations in capturing the spatial relationships between neighboring nodes and their higher-level features during the aggregation of neighbor representations. To address this issue, this study introduces a novel model, KGCNN, designed to comprehensively tackle DDI prediction tasks by considering spatial relationships between molecules within the biomedical knowledge graph (BKG). KGCNN is built upon a message-passing GNN framework, consisting of propagation and aggregation. In the context of the BKG, KGCNN governs the propagation of information based on semantic relationships, which determine the flow and exchange of information between different molecules. In contrast to traditional linear aggregators, KGCNN introduces a spatial-aware capsule aggregator, which effectively captures the spatial relationships among neighboring molecules and their higher-level features within the graph structure. The ultimate goal is to leverage these learned drug representations to predict potential DDIs. To evaluate the effectiveness of KGCNN, it undergoes testing on two datasets. Extensive experimental results demonstrate its superiority in DDI predictions and quantified performance. 

## 🚀 Installation

1⃣️ First, clone the Github repository:

```bash
$ git clone https://github.com/blair1213/KGCNN
$ cd KGCNN
```

2⃣️ Then, set up the environment. This codebase leverages Python, Pytorch, Pytorch Geometric, etc. To create an environment with all of the required packages, please ensure that conda is installed and then execute the commands:

```bash
$ conda env create -f kgcnn.yaml
$ conda activate kgcnn
```
3⃣️ Download Datasets

The dataset used in this work can be accessed at [Dataset(google drive)](https://drive.google.com/file/d/1zrMvnvbG2Ln6kfsVY47HPDJl1Iu0fWGC/view?usp=share_link) or [Dataset (zenodo)](https://zenodo.org/records/11529900). After download the kgcnn_raw_data.zip, please unzip it to raw_data.

### 🛠️ Training and Testing

After cloning the repository and installing all dependencies. You can run the following command to train our model:

```
$ python run.py
```
The trained model will be saved under :
```
$ ./ckpt/KGCNN_{dataset_name}_neigh_{neighbor_number}_embed_{embedding_dimension}_depth_{network_layer}_optimizer_adam_lr_{lr}_batch_size_{bz}_epoch_{epoch_num}.hdf5
```
For testing model, just load it and test it on testing dataset.

### 🏙️ Figure 

To recover each figure presented in this paper, please download the code and data at [Fig.zip(google drive)](https://drive.google.com/file/d/1sCM8xh9tdyhAU0fHUPiyVbJwQeKvBwu2/view?usp=sharing) or [Fig.zip (zenodo)](https://zenodo.org/records/11554803). You could find all code and initial data in this folder to recover each figure. By the way, some figures are painted by GraphPad, so we also uploaded the initial file of GraphPad.

### 🌟 Personalize based on your own dataset

If you want to benchmark KGCNN with your own dataset. You are kindly requested to perpare the follwing three files, including approved_example.txt, entity2id.txt, and train2id.txt.

- The approved_example.txt stores all DDI samples, including positive samples and negative samples. The format is [drug_id, drug_id, rel_id, label].
- The entity2id.txt stores the mapping between entity name and entity index in corresponding graph. The format is [entity source, entity_id].
- The train2id.txt stores all triples in biomedical knowledge graph. The format is [head_id, tail_id, rel_id].


### ⚖️ License

The code in this package is licensed under the MIT License.

</details>
