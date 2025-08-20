import os
from tensorflow import keras
from tensorflow.keras import layers, models

# Models architectures

class Basic_Custom_CNN:
    '''
    Basic custom model architecture. Setups, runs and saves model
    '''
    def __init__(self, input_shape=(256, 256, 1), num_classes=2, epochs=10):
        self.input_shape = input_shape
        self.classes = num_classes
        self.epochs = epochs
        self.model = None
    
    def architecture(self):
        '''Sets up model architecture for custom CNN'''
        inputs = keras.Input(shape=self.input_shape)
        model = models.Sequential([
                                    inputs,
                                    layers.Rescaling(1./255),                                           
                                    layers.Conv2D(filters=32, kernel_size=3, activation='relu'),        # kernel size 3x3
                                    layers.MaxPool2D(pool_size=2),                                      # pool size 2x2
                            
                                    layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
                                    layers.MaxPool2D(pool_size=2),
                            
                                    layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
                                    layers.MaxPool2D(pool_size=2),
                            
                                    layers.Flatten(),
                                    layers.Dense(1, activation='sigmoid')   
                                ])
        
        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                      metrics=['accuracy',
                                keras.metrics.Precision(name='precision'),
                                keras.metrics.Recall(name='recall'),
                                keras.metrics.AUC(name='auc')
                               ]
                       )
    
        self.model = model
        return self.model
    
    def train_model(self, train_gen, val_gen=None):
        '''Trains model'''
        # verify that model is initialized
        if self.model is None:
            raise ValueError("You need to initialize model using architecture() before training")
            
        # fit data to model
        history = self.model.fit(train_gen, validation_data=val_gen, epochs=self.epochs)
        return history.history
        
    def get_model(self):
        '''Returns trained model'''
        return self.model

    def save_model(self, models_directory="Models", model_file="new_file"):
        '''Save model data'''

        # makes directory for outputs
        os.makedirs(models_directory, exist_ok=True)

        # creates path and saves model
        path = os.path.join(models_directory, model_file)
        self.model.save(path)

        return path

class Custom_CNN:
    def __init__(self, phase, input_shape, train_data, val_data, test_data, epochs):
        self.phase = phase
        self.input_shape = input_shape
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.epochs = epochs
    
    def architecture(self):
        pass
    
    def train(self):
        pass
    
    def fit(self):
        pass

class VGG_CNN:
    def __init__(self, phase, input_shape, train_data, val_data, test_data, epochs):
        self.phase = phase
        self.input_shape = input_shape
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.epochs = epochs
    
    def architecture(self):
        pass
    
    def train(self):
        pass
    
    def fit(self):
        pass

class ResNet_CNN:
    def __init__(self, phase, input_shape, train_data, val_data, test_data, epochs):
        self.phase = phase
        self.input_shape = input_shape
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.epochs = epochs
    
    def architecture(self):
        pass
    
    def train(self):
        pass
    
    def fit(self):
        pass
    





