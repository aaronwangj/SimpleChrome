from __future__ import absolute_import
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, ReLU
from tensorflow.keras import Sequential



class MLP_Simple(tf.keras.Model):
    """
        A simple 2 layer multi layer perceptron model for predicting gene-expressing 
        using the HM
    """
    def __init__(self, _first_layer_size=256, _second_layer_size=128, _input_shape=100*5,
                _optimizer='adam', _loss='binary_crossentropy'):
        super(MLP_Simple, self).__init__()
        # Setup Model'
        print("Initializing Model")
        self.optimizer = _optimizer
        self.loss = _loss
        
        self.model = Sequential()
        
        self.model.add(Dense(_first_layer_size, activation='relu', 
                            kernel_initializer='he_normal', 
                            input_shape=(_input_shape,)))
        self.model.add(Dense(_second_layer_size, 
                            activation='relu', 
                            kernel_initializer='he_normal'))
        
        self.model.add(Dense(1, activation='sigmoid'))

        
    def compile(self):
        self.model.compile(optimizer=self.optimizer, 
            loss=self.loss, 
            metrics=['accuracy']
        )
    
    def fit(self, X_train, Y_train,
            epochs=20, batch_size=32, validation_split=0.1):
        
        self.model.fit(X_train, Y_train,
                epochs=epochs, batch_size=batch_size, 
                validation_split=validation_split, 
                verbose=1)

    def evaluate(self, X_test, Y_test):
        loss, accuracy = self.model.evaluate(X_test, Y_test,
                            verbose=1)
        return loss, accuracy
    
    def call(self, inputs):
        """
            Call the model over the inputs
        """ 
        return self.model(inputs)


        

    # def evaluate(self, x_test, y_test):
    #     return self.model.evaluate(x_test, y_test, verbose=0)



    

