from __future__ import absolute_import
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras import Sequential



class DeepChrome(tf.keras.Model):
    """
        A tensorflow implementation of original DeepChrome model
    """
    def __init__(self, _num_filters=625, kernel_size=10,
                 _pool_size=5, 
                 _first_layer_size=256, 
                 _second_layer_size=128,
                 _input_shape=(100, 5),
                 _optimizer='adam',
                 _loss='binary_crossentropy'):
        super(DeepChrome, self).__init__()
        # Setup Model'
        print("Initializing Model")
        self.optimizer = _optimizer
        self.loss = _loss
        
        self.model = Sequential()
        
        # Convolutional Block
        self.model.add(Conv1D(625,10,
                input_shape=_input_shape,
                activation='relu')
        )

        self.model.add(MaxPooling1D(5))
        
        # Flatten for dense layers
        self.model.add(Flatten())

        # standard 2 layer dense model
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
