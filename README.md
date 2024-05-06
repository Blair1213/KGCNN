<h1 align="center">
  [KGCNN] Knowledge Graph Neural Network with Spatial-Aware Capsules for Drug-Drug Interactions Prediction
</h1>

<p align="center">
    <a href="https://pypi.org/project/autoreviewer">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/autoreviewer" />
    </a>
    <a href="https://pypi.org/project/autoreviewer">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/autoreviewer" />
    </a>
    <a href="https://github.com/cthoyt/autoreviewer/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/autoreviewer" />
    </a>
    <a href='https://autoreviewer.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/autoreviewer/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://codecov.io/gh/cthoyt/autoreviewer/branch/main">
        <img src="https://codecov.io/gh/cthoyt/autoreviewer/branch/main/graph/badge.svg" alt="Codecov status" />
    </a>  
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /> 
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
    <a href="https://github.com/cthoyt/autoreviewer/blob/main/.github/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/>
    </a>
</p>

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

The dataset used in this work can be accessed at [Dataset](https://github.com/cthoyt/autoreviewer/actions?query=workflow%3ATests).

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
### ⚖️ License

The code in this package is licensed under the MIT License.

</details>
