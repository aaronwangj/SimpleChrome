# Each file contains a part of the model
# Deprecated 

from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Conv1D, Flatten, Dense, Reshape, Conv1DTranspose
from tensorflow.keras.initializers import RandomNormal
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 
    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


class VAE(tf.keras.Model):
    """
        Variational AutoEncoder Module
    """
    def __init__(self, _latent_dim=2, _input_shape=(100, 5, 1), 
                 _encoder_filters=[32, 64], 
                 _encoder_kernel_sizes=[5, 5],
                 _encoder_strides=[1, 1],
                 _decoder_filters=[64, 32],
                 _decoder_kernel_sizes=[5, 5],
                 _decoder_strides=[1, 1],
                 _encoder_use_bias = True,
                 _decoder_use_bias = True
        ):
        
        super(VAE, self).__init__()
        # Shared Hyper Parameters
        self.latent_dim = _latent_dim
        self.weight_initializer = RandomNormal(mean = 0.0, stddev = 0.02)
        self.lamb = 0.00005
        self.leaky = LeakyReLU(alpha=0.7)
        # Encoder Hyper Parameters
        self.encoder_filters = _encoder_filters
        self.encoder_kernel_sizes = _encoder_kernel_sizes
        self.encoder_strides = _encoder_strides
        self.encoder_use_bias = _encoder_use_bias
        # Decoder Hyper Parameters
        self.decoder_filters = _decoder_filters
        self.decoder_kernel_sizes = _decoder_kernel_sizes
        self.decoder_strides = _decoder_strides
        self.decoder_use_bias = _decoder_use_bias
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        
        # Encoder
        self.encoder = tf.keras.Sequential(
            [
                Conv1D( filters=self.encoder_filters[0], 
                    kernel_size=self.encoder_kernel_sizes[0], 
                    strides=self.encoder_strides[0], 
                    padding='SAME',
                    use_bias = self.encoder_use_bias, 
                    kernel_initializer = self.weight_initializer,
                    input_shape=_input_shape
                ),
                self.leaky,
                Conv1D( filters=self.encoder_filters[1], 
                    kernel_size=self.encoder_kernel_sizes[1], 
                    strides=self.encoder_strides[1], 
                    padding='SAME',
                    use_bias = self.encoder_use_bias, 
                    kernel_initializer = self.weight_initializer
                ),
                self.leaky,
                Flatten(),
                Dense(self.latent_dim + self.latent_dim),
            ]
        )
        # Decoder
        self.decoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(self.latent_dim,)),
                Dense(units=100*64),
                self.leaky,
                Reshape(target_shape=(100, 64)),
                Conv1DTranspose(filters=self.decoder_filters[0],
                    kernel_size=self.decoder_kernel_sizes[0], 
                    strides=self.decoder_strides[0], 
                    padding='SAME',
                    use_bias = self.decoder_use_bias, 
                    kernel_initializer = self.weight_initializer
                ),
                self.leaky,
                Conv1DTranspose(filters=self.decoder_filters[1],
                    kernel_size=self.decoder_kernel_sizes[1], 
                    strides=self.decoder_strides[1], 
                    padding='SAME',
                    use_bias = self.decoder_use_bias, 
                    kernel_initializer = self.weight_initializer
                ),
                self.leaky,
                Conv1DTranspose(filters=5, 
                    kernel_size=1, 
                    strides=1, 
                    padding='SAME',
                    activation = tf.nn.tanh
                ) 
            ]
        )
        
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
            
        return self.decode(eps, apply_sigmoid=True)
    
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape, mean=0.0, stddev = 1.0)
        return eps * tf.exp(logvar * .5) + mean 
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        
        return logits

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. *
                            tf.exp(-logvar) + logvar + log2pi),
                            axis=raxis)
    def loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        output = self.decode(z)
        #reconstruction loss
        rec = tf.reduce_mean(tf.square(output - x))
        #kl loss
        kl = -0.5*tf.reduce_sum(logvar+1-tf.math.square(mean)-tf.math.exp(logvar))/mean.shape[0]
        negelbo = rec + self.lamb*kl
        return rec, kl, negelbo
    
    def _hidden_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
        print(logpx_z.shape)
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def train_step(self, x):
        with tf.GradientTape() as tape:
            _, _, loss = self.loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def fit(self, X_data, batch_size=64, epochs=1):
        X_train, X_valid = train_test_split(X_data, test_size=0.1)
        
        loss_list = []
        for epoch in range(epochs):

            nbatch = round(X_train.shape[0]/batch_size)
            loss_list = []
            for i in range(nbatch):
                temp_id = batch_size*i + np.array(range(batch_size))
                batch = X_train[np.min(temp_id):(np.max(temp_id)+1), :]
                batch_loss = self.train_step(batch)
                loss_list.append(batch_loss)
                print("Batch: {}\nBatch Loss: {}".format(i, batch_loss))
        print('Final Loss: {}\n Average Loss: {}'.format(loss_list[-1], np.mean(loss_list)))
        visualize_loss(loss_list)

            # validation_loss = 



# class Encoder(tf.keras.Model):
#     """
#     Encode model with 2 conv blocks followed by a single diverging Dense layer
#     """
#     def __init__(self, _z_dim, _first_conv_size=64, _second_conv_size=128):
#         super(Encoder, self).__init__()
#         # Setting up hyper parameters
#         self.z_dim = _z_dim
#         self.first_conv_size = _first_conv_size
#         self.second_conv_size = _second_conv_size 
#         self.first_conv_strides = (2,2)
#         self.second_conv_strides = (2,2)
#         self.first_conv_kernel = (5,5)
#         self.second_conv_kernel = (5,5)
        
#         # Variable Initializer
#         self.weight_initializer = tf.keras.RandomNormal(mean=0.0, stddev=0.02)
        

#         # Setup Model Layers
        
#         # First Convolutional Block
#         self.conv_1 = tf.keras.layers.Conv2D(self.first_conv_size, 
#                                         self.first_conv_kernel, 
#                                         strides = self.first_conv_strides, 
#                                         padding = "SAME", 
#                                         use_bias = False, 
#                                         kernel_initializer = self.weight_initializer
#                                     )
        
#         self.batch_norm_1 = tf.keras.layers.BatchNormalization()
#         self.relu_1 = tf.keras.layers.ReLU()

#         # Second Convolutional Block
#         self.conv_2 = tf.keras.layers.Conv2D(self.second_conv_size,
#                                              self.second_conv_kernel, 
#                                              strides = self.second_conv_strides, 
#                                              padding = "SAME", 
#                                              use_bias = False, 
#                                              kernel_initializer = self.weight_initializer
#                                             )
#         self.batch_norm_2 = tf.keras.layers.BatchNormalization()
#         self.relu_2 = tf.keras.layers.ReLU()
        
        
#         # Flatten for subsequent Dense layers
#         self.flatten = tf.keras.layers.Flatten()
        
#         # Dense for Means
#         self.dense_means = tf.keras.layers.Dense(self.z_dim)
        
#         # Dense for Std
#         self.dense_std = tf.keras.layers.Dense(self.z_dim)

#     def call(self, inputs):
#         """
#         Call function for Encoder Model
#         @params self <Object>, passed by default the model object itself
#         @params inputs <np.array>, the batch or single input that needs to passed through this model
#         @returns <np.array>, <np.array> both with sizes (batch, 1)
#         """
#         conv_block_1_output = self.relu_1(self.batch_norm_1(self.conv_1(inputs)))
#         conv_block_2_output = self.relu_2(self.batch_norm_2(self.conv_2(conv_block_1_output)))
#         flattened = self.flatten(conv_block_2_output)
#         means = self.dense_means(flattened)
#         std = self.dense_std(flattened)

#         return means, std
        

# class Decoder(tf.keras.Model):
#     """
#     decoder model
#     """
#     def __init__(self):
#         super(Decoder, self).__init__()
#         #  use sequential model
#         self.model = tf.keras.models.Sequential()
#         self.model.add(Dense(25*400*2, use_bias=False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)))        
#         self.model.add(BatchNormalization())
#         self.model.add(ReLU()) 
#         self.model.add(Reshape((5, 100, 40)))
#         # self.model.add(Conv2DTranspose(512,(5, 5), strides = (2,2), padding = "SAME", use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)))
#         # self.model.add(BatchNormalization())
#         # self.model.add(ReLU())
#         # self.model.add(Conv2DTranspose(256,(5, 5), strides = (2,2), padding = "SAME", use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)))
#         # self.model.add(BatchNormalization())
#         # self.model.add(ReLU())
#         self.model.add(Conv2DTranspose(128,(5, 5), strides = (1,1), padding = "SAME", use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)))
#         self.model.add(BatchNormalization())
#         self.model.add(ReLU())
#         self.model.add(Conv2DTranspose(1,(5, 5), strides = (1,1), padding = "SAME", use_bias = False, activation = tf.nn.tanh, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)))
#         self.model.compile()
#         """
#         """
#     def call(self, inputs):
#         """
#         pass inputs
#         """
#         outputs = self.model(inputs)
#         return outputs
#         """
#         """



# class VAE(tf.keras.Model):
#     """
#     whole model
#     """
#     def __init__(self,z_dim, u, v, lamb):
#         super(VAE, self).__init__()
#         self.z_dim = z_dim
#         #prior mean, normally it's just 0
#         self.u = u
#         #prior covariance, normally it's I
#         self.v = v
#         #lambda weight for KL divergvence
#         self.lamb = lamb
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1 = 0.9, beta_2 = 0.999)
#         self.encoder = Encoder(z_dim)
#         self.decoder = Decoder()
#         """
#         """
#     def call(self, inputs):
#         """
#         pass inputs with reparameterization trick
#         """
#         means, logvar = self.encoder(inputs)
#         eps = tf.random.normal(means.shape, mean=0.0, stddev = 1.0)
#         z = means+eps*tf.math.sqrt(tf.math.exp(logvar))
#         outputs = self.decoder(z)
#         return means, logvar, outputs
#         """
#         """
#     def loss(self, means, logvar, inputs, outputs):
#         #reconstruction loss
#         rec = tf.reduce_mean(tf.square(outputs - inputs))
#         #kl loss
#         kl = -0.5*tf.reduce_sum(logvar+1-tf.math.square(means)-tf.math.exp(logvar))/means.shape[0]
#         #total loss
#         negelbo = rec + self.lamb*kl
#         return rec, kl, negelbo








