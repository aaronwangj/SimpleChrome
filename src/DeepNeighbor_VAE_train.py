from DeepNeighbor_VAE import *
from math import *
import tensorflow_hub as hub
# import tensorflow_gan as tfgan
from imageio import imwrite
import os
import argparse
import sys
import pickle



#load in data
train_data = np.loadtxt('C:/Users/aaronjw/Desktop/GitHub/DeepNeighbors/dataset/data/E003/classification/train.csv', delimiter = ',')
train_x = []
train_y = []
for i in range(6601):
	temp_id = np.asarray([j for j in range(100)])+i*100
	train_x.append(np.transpose(train_data[temp_id,2:7]))
	train_y.append(train_data[temp_id[0], 7])


#data normalization
max_val = np.max(np.max(train_x, axis = 2), axis = 0)
train_x = np.asarray(train_x, dtype = 'float32')
for i in range(5):
	for j in range(train_x.shape[0]):
		train_x[j][i] /= max_val[i]

train_x*=2
train_x-=1
train_x = tf.convert_to_tensor(train_x)
train_x = tf.reshape(train_x, (6601, 5, 100, 1))
train_y = np.asarray(train_y)

#define hyperparameters 
#dimension of latent variable
z_dim = 2
#penalty parameter for KL divergence
lamb = 0.00005
#batch size
batch_size= 100
#prior mean u and v
u = tf.convert_to_tensor(np.zeros(z_dim), dtype = 'float32')
temp = np.zeros((z_dim, z_dim))
for i in range(z_dim):
    temp[i,i] = 1.0
#
v = tf.convert_to_tensor(temp, dtype = 'float32')
#initializae model
vae = VAE(z_dim, u, v, lamb)


#train function
def train(vae, train_x, batch_size, epoch):
    nbatch = round(train_x.shape[0]/batch_size)
    for i in range(nbatch):
        temp_id = batch_size*i + np.array(range(batch_size))
        batch = train_x[np.min(temp_id):(np.max(temp_id)+1), :]
        with tf.GradientTape() as tape:
            means, logvar, outputs = vae.call(batch)
            rec, kl, loss = vae.loss(means, logvar, batch, outputs)
        gradients = tape.gradient(loss, vae.trainable_variables)
        vae.optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        tf.cast(loss, tf.float32)
        print("Current Loss: {0:.2f}".format(loss))
        # all_loss.append([rec, kl, loss])
        # #save trained model
        # if iteration % 1000 == 0:
        #     vae.save_weights(filepath = 'trained_models/vae_'+str(vae.lamb)+'_'+str(epoch)+'_'+str(iteration)+'.h5')
        #     print('**** LOSS: %g ****' % loss)
for epoch in range(70):
    vae.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001*0.995**(epoch), beta_1=0.9, beta_2 = 0.999)
    print('========================== EPOCH %d  ==========================' % epoch)
    train(vae, train_x, batch_size, epoch)






# import matplotlib.pyplot as plt

# z, _ = vae.encoder(train_x)
# z = np.asarray(z)

# plt.scatter(z[np.where(train_y == 0)[0], 0], z[np.where(train_y == 0)[0], 1], s = 1.5)
# plt.scatter(z[np.where(train_y == 1)[0], 0], z[np.where(train_y == 1)[0], 1], s = 1.5)






