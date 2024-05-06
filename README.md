<h1 align="center">
  Knowledge Graph Neural Network with Spatial-Aware Capsules for Drug-Drug Interactions Prediction
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

Desired interface:

Run on the command line with:

```shell
$ autoreviewer https://github.com/rs-costa/sbml2hyb
```

## 🚀 Installation

First, clone the Github repository:

```bash
$ git clone https://github.com/blair1213/KGCNN
$ cd KGCNN
```

Then, set up the environment. This codebase leverages Python, Pytorch, Pytorch Geometric, etc. To create an environment with all of the required packages, please ensure that conda is installed and then execute the commands:

```bash
$ conda env create -f kgcnn.yaml
$ conda activate kgcnn
```
Download Datasets

The dataset used in this work can be accessed at [GitHub Action](https://github.com/cthoyt/autoreviewer/actions?query=workflow%3ATests).

### ⚖️ License

The code in this package is licensed under the MIT License.

### 🛠️ Testing

After cloning the repository and installing all dependencies. You can run following command to train our model:

```
$ python run.py
```


</details>
