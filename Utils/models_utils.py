import os
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers

# Models architectures

class Basic_Custom_CNN:
    '''
    Basic custom model architecture. Setups, runs and saves model
    '''
    def __init__(self, input_shape=(256, 256, 1), num_classes=1, epochs=10):
        self.input_shape = input_shape
        self.classes = num_classes
        self.epochs = epochs
        self.model = None
    
    def architecture(self):
        '''Sets up model architecture for custom CNN'''
        model = models.Sequential([
                                    layers.InputLayer(input_shape=self.input_shape),
                                    layers.Rescaling(1./255), 
            
                                    layers.Conv2D(filters=32, kernel_size=3, activation='relu'),        
                                    layers.MaxPool2D(pool_size=2),                                      
                            
                                    layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
                                    layers.MaxPool2D(pool_size=2),
                            
                                    layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
                                    layers.MaxPool2D(pool_size=2),
                            
                                    layers.Flatten(),
                                    layers.Dense(1, activation='sigmoid')   
                                ])
        
        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                      metrics=['accuracy']
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

class Dynamic_Custom_CNN:
    '''
    Dynamic custom model architecture. Setups, runs and saves model
    '''
    def __init__(self, input_shape=(256, 256, 1), num_classes=1, epochs=10, layer_sizes=[32, 64, 128], activation='relu', dense_units=None, dropout=None):
        self.input_shape = input_shape
        self.classes = num_classes
        self.epochs = epochs
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.dense_units = dense_units
        self.dropout = dropout
        self.model = None
    
    def architecture(self):
        '''Sets up model architecture for custom CNN'''
        model = models.Sequential()
        model.add(layers.Input(shape=self.input_shape))

        # add convolutional layers
        for size in self.layer_sizes:
            model.add(layers.Conv2D(size, (3,3), padding="same"))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation(self.activation))
            model.add(layers.MaxPooling2D((2,2)))

        model.add(layers.Flatten())

        # add densly connected layers
        if self.dense_units is not None:
            for d_units in self.dense_units:
                model.add(layers.Dense(d_units, activation=self.activation))
                if self.dropout is not None:
                    model.add(layers.Dropout(self.dropout))
            
        if self.classes == 1:
            # binary classification
            model.add(layers.Dense(1, activation="sigmoid")) 
            loss = "binary_crossentropy"
        else:
            # multy label class
            model.add(layers.Dense(self.classes, activation="softmax"))     
            loss = "categorical_crossentropy"
        
        model.compile(loss=loss,
                      optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                      metrics=['accuracy']
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



class VGG16_Transfer:
    def __init__(self, input_shape=(224, 224, 3), num_classes=1, epochs=10, dropout=0.4, lr=1e-4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.dropout = dropout
        self.lr = lr
        self.vgg_model = None

    def architecture(self):
        '''Builds VGG16 model architecture for transfer learning '''
        # load pretrained model, fully connected layers(top) are not downloaded
        vgg_model_base = VGG16(weights="imagenet", include_top=False, input_shape=self.input_shape)
        
        # freeze base layers to add custom classifier
        for layer in vgg_model_base.layers:
            layer.trainable = False
        
        # create architecture: add base ans custom classifier 
        vgg_model = models.Sequential([
            vgg_model_base,                            # convolutional layers
            layers.Flatten(),
            layers.Dense(256, activation="relu"),      # fully connected layer
            layers.Dropout(self.dropout),                       # add settings selected in P2 to not overfit
            layers.Dense(self.num_classes, activation="sigmoid")      # binary classification(scalar value)
        ])
        
        vgg_model.compile(loss="binary_crossentropy",
                          optimizer=optimizers.Adam(self.lr),
                          metrics=['accuracy']
                         )
        self.vgg_model = vgg_model
        return vgg_model

    def train_model(self, train_gen, val_gen=None):
        '''Trains model'''
        # verify that model is initialized
        if self.vgg_model is None:
            raise ValueError("You need to initialize model using architecture() before training")
            
        # fit data to model
        history = self.vgg_model.fit(train_gen, validation_data=val_gen, epochs=self.epochs)
        return history.history
        
    def get_model(self):
        '''Returns trained model'''
        return self.vgg_model


    def save_model(self, models_directory="Models", model_file="new_file"):
        '''Save model data'''

        # makes directory for outputs
        os.makedirs(models_directory, exist_ok=True)

        # creates path and saves model
        path = os.path.join(models_directory, model_file)
        self.vgg_model.save(path)

        return path

class ResNet50_Transfer:
    def __init__(self, input_shape=(224, 224, 3), num_classes=1, epochs=10, dropout=0.4, lr=1e-4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.dropout = dropout
        self.lr = lr
        self.resnet_model = None

    def architecture(self):
        '''Builds ResNet50 model architecture for transfer learning '''
        # load pretrained model, fully connected layers(top) are not downloaded
        resnet_model_base = ResNet50(weights="imagenet", include_top=False, input_shape=self.input_shape)
        
        # freeze base layers to add custom classifier
        for layer in resnet_model_base.layers:
            layer.trainable = False
        
        # create architecture: add base ans custom classifier 
        resnet_model = models.Sequential([
            resnet_model_base,                            # convolutional layers
            layers.GlobalAveragePooling2D(),             # used inted of flatten
            layers.Dense(256, activation="relu"),      # fully connected layer
            layers.Dropout(self.dropout),                       # add settings selected in P2 to not overfit
            layers.Dense(self.num_classes, activation="sigmoid")      # binary classification(scalar value)
        ])
        
        resnet_model.compile(loss="binary_crossentropy",
                          optimizer=optimizers.Adam(self.lr),
                          metrics=['accuracy']
                            )
        self.resnet_model = resnet_model
        return resnet_model 
       
    def train_model(self, train_gen, val_gen=None):
        '''Trains model'''
        # verify that model is initialized
        if self.resnet_model is None:
            raise ValueError("You need to initialize model using architecture() before training")
            
        # fit data to model
        history = self.resnet_model.fit(train_gen, validation_data=val_gen, epochs=self.epochs)
        return history.history
        
    def get_model(self):
        '''Returns trained model'''
        return self.resnet_model


    def save_model(self, models_directory="Models", model_file="new_file"):
        '''Save model data'''

        # makes directory for outputs
        os.makedirs(models_directory, exist_ok=True)

        # creates path and saves model
        path = os.path.join(models_directory, model_file)
        self.resnet_model.save(path)

        return path
    





