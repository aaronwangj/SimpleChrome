## SimpleChrome
With the breakthrough of recent state-of-the-art sequencing technology, genomics datasets have become ubiquitous. The emergence of large-scale datasets provides great opportunities for better understanding of genomics and one fundamental task is to understand gene regulation. Although each cell in the human body contains the same set of DNA information, gene expression controls the functions of these cells. There are two important factors that control the expression level of each gene: (1). Gene regulation such as histone modification can directly regulate gene expression. (2). Neighbor genes that are functional related or interact to each other can effect the gene expression level. Previous efforts tried to address the former using Attention-based model. To address the second problem, it requires to incorporate all the potential related gene information into the model. Though modern machine learning and deep learning models were shown to be able capture signals when applied to moderate size of the data, they can struggle to recover the underlying signals due to the high dimensionality. To remedy this issue, we present SimpleChrome, a deep learning model that learns the latent histone modification representations of genes. The features learned from the model allow us to better understand the combinatorial effects of cross-gene interactions and direct gene regulation on the target gene expression. The results of this paper have immediate downstream effects in epigenomics research and drug development. <br/>
By Wei Cheng, Ghulam Murtaza, and Aaron Wang

## Setting up the requirements
To run the code first of all setup a python environment and install required python dependencies
Step 1: Create a python virtual environment.
```bash

    python3 -m venv ./deepneighbors
```

Step 2: Activate the virtual environment.
```bash
    source ./deepneighbors/bin/activate
```
Step 3: Install the dependencies. 
```bash
    pip install -r requirements.txt
```
## Directory Structure
- **src**: directory contains all the scripts that abstract away the lower level implementation details. The current version handles: 
    - Downloading the dataset from the source repository
    ```python
        check_if_dataset_exists()
    ```
    This function downloads the dataset if its not downloaded already. **dataset requires 3GBs of freespace** 

    - Parsing the dataset and provide a neat set of function to get the gene data and neighboring gene data in form of 
    ```python
        get_gene_data()
        get_neighbors_data()
    ```

- **deepneighbors**: Contains the python virtual environment
- **dataset**: Contains the dataset in ./dataset/data/E{Cell_ID} format. For example the gene expression data of E003 is in ./dataset/data/E003/ directory

## Whats Done:
- Data parsing for all cells 
- Implement VAE module
- Translate DeepChrome to tensorflow so that it can be integrated with our eventual VAE module
- Evaluation

## TODOs:
- Translate AttentiveChrome to tensorflow so that it can be integrated with our eventual VAE module
- Final Write-up

