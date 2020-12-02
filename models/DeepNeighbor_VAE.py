# Each file contains a part of the model

from __future__ import absolute_import
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, ReLU



class Encoder(tf.keras.Model):
    """
    Encode model with 2 conv blocks followed by a single diverging Dense layer
    """
    def __init__(self, _z_dim, _first_conv_size=64, _second_conv_size=128):
        super(Encoder, self).__init__()
        # Setting up hyper parameters
        self.z_dim = _z_dim
        self.first_conv_size = _first_conv_size
        self.second_conv_size = _second_conv_size 
        self.first_conv_strides = (2,2)
        self.second_conv_strides = (2,2)
        self.first_conv_kernel = (5,5)
        self.second_conv_kernel = (5,5)
        
        # Variable Initializer
        self.weight_initializer = tf.keras.RandomNormal(mean=0.0, stddev=0.02)
        

        # Setup Model Layers
        
        # First Convolutional Block
        self.conv_1 = tf.keras.layers.Conv2D(self.first_conv_size, 
                                        self.first_conv_kernel, 
                                        strides = self.first_conv_strides, 
                                        padding = "SAME", 
                                        use_bias = False, 
                                        kernel_initializer = self.weight_initializer
                                    )
        
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.relu_1 = tf.keras.layers.ReLU()

        # Second Convolutional Block
        self.conv_2 = tf.keras.layers.Conv2D(self.second_conv_size,
                                             self.second_conv_kernel, 
                                             strides = self.second_conv_strides, 
                                             padding = "SAME", 
                                             use_bias = False, 
                                             kernel_initializer = self.weight_initializer
                                            )
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.relu_2 = tf.keras.layers.ReLU()
        
        
        # Flatten for subsequent Dense layers
        self.flatten = tf.keras.layers.Flatten()
        
        # Dense for Means
        self.dense_means = tf.keras.layers.Dense(self.z_dim)
        
        # Dense for Std
        self.dense_std = tf.keras.layers.Dense(self.z_dim)

    def call(self, inputs):
        """
        Call function for Encoder Model
        @params self <Object>, passed by default the model object itself
        @params inputs <np.array>, the batch or single input that needs to passed through this model
        @returns <np.array>, <np.array> both with sizes (batch, 1)
        """
        conv_block_1_output = self.relu_1(self.batch_norm_1(self.conv_1(inputs)))
        conv_block_2_output = self.relu_2(self.batch_norm_2(self.conv_2(conv_block_1_output)))
        flattened = self.flatten(conv_block_2_output)
        means = self.dense_means(flattened)
        std = self.dense_std(flattened)

        return means, std
        








class Decoder(tf.keras.Model):
    """
    decoder model
    """
    def __init__(self):
        super(Decoder, self).__init__()
        #  use sequential model
        self.model = tf.keras.models.Sequential()
        self.model.add(Dense(25*400*2, use_bias=False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)))        
        self.model.add(BatchNormalization())
        self.model.add(ReLU()) 
        self.model.add(Reshape((5, 100, 40)))
        # self.model.add(Conv2DTranspose(512,(5, 5), strides = (2,2), padding = "SAME", use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)))
        # self.model.add(BatchNormalization())
        # self.model.add(ReLU())
        # self.model.add(Conv2DTranspose(256,(5, 5), strides = (2,2), padding = "SAME", use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)))
        # self.model.add(BatchNormalization())
        # self.model.add(ReLU())
        self.model.add(Conv2DTranspose(128,(5, 5), strides = (1,1), padding = "SAME", use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)))
        self.model.add(BatchNormalization())
        self.model.add(ReLU())
        self.model.add(Conv2DTranspose(1,(5, 5), strides = (1,1), padding = "SAME", use_bias = False, activation = tf.nn.tanh, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)))
        self.model.compile()
        """
        """
    def call(self, inputs):
        """
        pass inputs
        """
        outputs = self.model(inputs)
        return outputs
        """
        """






class VAE(tf.keras.Model):
    """
    whole model
    """
    def __init__(self,z_dim, u, v, lamb):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        #prior mean, normally it's just 0
        self.u = u
        #prior covariance, normally it's I
        self.v = v
        #lambda weight for KL divergvence
        self.lamb = lamb
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1 = 0.9, beta_2 = 0.999)
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder()
        """
        """
    def call(self, inputs):
        """
        pass inputs with reparameterization trick
        """
        means, logvar = self.encoder(inputs)
        eps = tf.random.normal(means.shape, mean=0.0, stddev = 1.0)
        z = means+eps*tf.math.sqrt(tf.math.exp(logvar))
        outputs = self.decoder(z)
        return means, logvar, outputs
        """
        """
    def loss(self, means, logvar, inputs, outputs):
        #reconstruction loss
        rec = tf.reduce_mean(tf.square(outputs - inputs))
        #kl loss
        kl = -0.5*tf.reduce_sum(logvar+1-tf.math.square(means)-tf.math.exp(logvar))/means.shape[0]
        #total loss
        negelbo = rec + self.lamb*kl
        return rec, kl, negelbo








