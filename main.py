from src import download_dataset, parse
from models import MLP_Simple, DeepChrome, AttentiveChrome
import numpy as np
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense



download_dataset.check_if_dataset_exists()

gene_data, gene_ids = parse.parse_all_cell_files('dataset/data/E100')

x_data = np.zeros((len(gene_ids), 100, 5), dtype='float32')
y_data = np.zeros((len(gene_ids), 1), dtype='float32')


for x, gene_id in enumerate(gene_ids):
    hm_matrix, expression = parse.get_gene_data(gene_data, gene_id)
    x_data[x] = np.array(hm_matrix) 
    y_data[x] = np.array(expression) 


X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.33)

print(X_train.shape, Y_train.shape)

# mlp_model = DeepChrome.DeepChrome()
# mlp_model.compile()
# mlp_model.fit(X_train, Y_train)

# print(mlp_model.evaluate(X_test, Y_test))

attentive_chrome = AttentiveChrome.AttentiveChrome()



# model = Sequential()

# model.add(Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(500,)))
# model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
# loss, acc = model.evaluate(X_test, Y_test, verbose=1)
# print('Test Accuracy: %.3f' % acc)



# for x, gene_id in enumerate(gene_ids):
    
#     i,j = parse.get_neighbors_data(gene_data, gene_id, gene_ids).shape

#     if i != 2100:
#         print(gene_id, i, j)
    
