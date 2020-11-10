from src import download_dataset, parse



download_dataset.check_if_dataset_exists()

gene_data, gene_ids = parse.parse_all_cell_files('dataset/data/E003')

#hm_matrix, expression = parse.get_gene_data(gene_data, gene_ids[0])
for x, gene_id in enumerate(gene_ids):
    
    i,j = parse.get_neighbors_data(gene_data, gene_id, gene_ids).shape

    if i != 2100:
        print(gene_id, i, j)
    
