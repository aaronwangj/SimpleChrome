## DeepNeighbors
Final Project for CSCI 2952G: Deep Learning in Genomics. <br/>
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

