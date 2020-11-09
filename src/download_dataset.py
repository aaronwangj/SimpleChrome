"""
    This script file handles that all data is present in the dataset directory
"""
BASE_DATA_SET_LINK = 'https://zenodo.org/record/2652278/files/data.tar.gz'

import requests
from tqdm.auto import tqdm
import tarfile
import sys

from src.util import exists, mkdir, delete


def download_dataset(download_path='dataset/'):
    filename = '{}{}'.format(download_path, BASE_DATA_SET_LINK.split('/')[-1])


    response = requests.get(BASE_DATA_SET_LINK, stream=True)

    with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                total=int(response.headers.get('content-length', 0)),
                desc=filename) as fout:
        
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)
    
    print("Download Completed!")
    return filename


def decompress_dataset(dataset):
    if exists(dataset):
        decompression_directory = '/'.join(dataset.split('/')[:-1])
        tar = tarfile.open(dataset, "r:gz")
        tar.extractall(decompression_directory)
        tar.close()
        delete(dataset)
    else:
        return

def check_if_dataset_exists(dataset_path='dataset/'):
    if not exists(dataset_path):
        print("Downloading Dataset!")
        mkdir(dataset_path)
        download_path = download_dataset()
        print("Decompressing Dataset!")
        decompress_dataset(download_path)
        print("Decompression Complete!")
    else:
        print("Dataset already exists in directory {}".format(dataset_path))
    




















