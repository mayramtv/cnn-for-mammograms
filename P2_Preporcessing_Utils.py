import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Test
class Test_py:
    def __init__(self, t_name="test0"):
        self.t_name = t_name

    def print_(self):
        return "Class: Try if python utils connects to notebook: " + self.t_name

def test_py(t_name):
    return "Function: Try if python utils connects to notebook: " + t_name

# Preprocessing techniques to try

class Image_Enhancement:
    def __init__(self, data, preprocessing_options):
        self.data = data
        self.preprocessing_options = preprocessing_options

    def crop(self):
        pass
    
    def noise_reduction(self):
        pass
    
    def edge_enhancement(self):
        pass
    
    def background_removal(self):
        pass
    
    def textural_preporcessing(self):
        pass



# Data loading and encoding
def data_loading(train_file, test_file):
    '''Loads data using the files names'''
    
    base_path = "CBIS-DDSM_Clean_Data/"
    train = pd.read_csv(base_path + train_file)
    test = pd.read_csv(base_path + test_file)
    
    return train, test
    
    
def labels_encoding(train, test):
    '''Find the number of classes and encodes the classes to integers'''
    
    train_data = train.copy()
    test_data = test.copy()
    train_data["label"] = LabelEncoder().fit_transform(train_data["pathology"]).astype(np.int32)
    test_data["label"] = LabelEncoder().fit_transform(test_data["pathology"]).astype(np.int32)
    
    return train_data, test_data
    
    
def split_train(train, test, val_size, stratify_col="label"):
    '''Divide training data into training and validation makes a copy of test data'''
    train_data, val_data = train_test_split(train, 
                                        test_size=val_size, 
                                        stratify=train[stratify_col], 
                                        random_state=42
                                       )
    test_data = test.copy()
    total = len(train_data) + len(val_data) + len(test_data)
    train_percent =  round(((len(train_data) * 100)/ total), 2)
    val_percent =  round(((len(val_data) * 100)/ total), 2)
    test_percent =  round(((len(test_data) * 100)/ total), 2)
    
    print("Train set:", len(train_data), "cases,", train_percent, "%")
    print("Validation set:", len(val_data), "cases,", val_percent, "%")
    print("Test set:", len(test_data), "cases,", test_percent, "%")
    return train_data, val_data, test_data
    

def image_iterator(data_sets, suffle=False):
    '''Generate a data generator for processing each image''' 
    
    # function for setup generators
    def data_generator(dataset, target_size, shuffle):
        # initiate generators
        gen = ImageDataGenerator()
        data_gen = t_generator.flow_from_dataframe(
                                            dataframe=dataset,
                                            x_col="image_path",
                                            y_col="label",
                                            target_size=target_size,
                                            color_mode="grayscale",
                                            class_mode="raw",
                                            batch_size=32,
                                            shuffle=shuffle,
                                            seed=42
                                            )
        return data_gen

    # setup generators
    train_gen = data_generator(train_data, (256, 256), True)
    val_gen = data_generator(val_data, (256, 256), False)
    test_gen = data_generator(test_data, (256, 256), False)
    
    return train_gen, val_gen, test_gen


# Basic custom model

class Basic_Custom_CNN:
    '''Setup and runs model'''
    def __init__(self, phase=2, input_shape=(256, 256, 1), num_classes=2, epochs):
        self.phase = phase
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
                                layers.Dense(self.classes, activation='softmax')   
                                ])
        model. compile(loss='sparse_categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy']
                       )
    
        self.model = model
    
    def train_model(self):
        # fit data to model
        history = model.fit(train_gen, validation_data=val_gen, epochs=10)
        history
    
    def custom_CNN_fit(self):
        pass



# Models evaluation
class Evaluation:
    def __init__(self, model, test_data):
        self.model = model
        self.X_test_data = test_data[0]
        self.y_actual = test_data[1]

    def custom_CNN_predict(self):
        pass 

    def confusion_matrix(self):
        pass

    def calculate_metrics(self):
        pass

# Models visualization 
class Visualization:
    def __init__(self, results):
        pass

    def confusion_matrix(self):
        pass

    def line_plot(self):
        pass

    def learning_curves(self):
        pass

    def spider_radar(self):
        pass

    def ROC_curve(self):
        pass


# Save output data
class Save_Data:
    def __init__(self, model=None, model_name="model", save_directory="Outputs"):
        self.model = model
        self.model_name = model_name
        self.save_directory = save_directory
        self.data = {} 

    def add_metric(self, metric, value):
        pass
        
    def add_description(self, description):
        pass

    def save_data(self, filename=None):
        pass
    


