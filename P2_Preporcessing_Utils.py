import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,  f1_score, precision_score, recall_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns

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
    def __init__(self, input_shape=(256, 256, 1), num_classes=2, epochs):
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
    
    def train_model(self, train_gen, val_gen=None epochs=10):
        '''Trains model'''
        # verify that model is initialized
        if self.model is None:
            raise ValueError("You need to initialize model using architecture() before training")
            
        # fit data to model
        model.fit(train_gen, validation_data=val_gen, epochs=epochs)
        
    def get_model(self):
        '''Returns trained model'''
        return self.model

    def save_model(self, models_directory="Models", model_file):
        '''Save model '''

        # makes directory for outputs
        os.makedir(models_directory, exist_ok=True)

        # creates path and saves model
        path = os.path.join(models_directory, model_file)
        self.model.save(path)


# Models evaluation
class Evaluation:
    '''Make predictions and evaluation of the model and calculates evaluation metrics'''
    def __init__(self, model):
        self.model = model
        self.metrics = {}

    def evaluate(self, test_gen):
        '''Evaluate model and return metrics defined during model compile'''
        return self.model.evaluate(test_gen)

    def predict(self, test_gen):
        '''Makes predictions on unseen data'''
        return self.model.predict(test_gen) 

    def confusion_matrix(self):
        '''Claculate confusion metrics and return:
        True Negative, False Positive, False Negative and True Positive'''

        

    def calculate_metrics(self, y_true, y_predics, conf_matrix):
        '''Claculate confusion matrix (True Negative, False Positive, False Negative and True Positive) 
        and evaluation metrics (Accuracy, Precision, Recall, F1 score, AUC
        Specificity, False Positive Rate, and False Negative Rate)'''
        # find predicted class
        y_pred_class = (y_predics > 0.5).astype(int)

        # get confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
        
        # calculate and save metrics
        self.metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred_class)
        self.metrics["accuracy"] = accuracy_score(y_true, y_pred_class)
        self.metrics["precision"] = precision_score(y_true, y_pred_class) # Positive predicive value: identify correct positives
        self.metrics["recall"] = recall_score(y_true, y_pred_class)       # sensitivity: identify positives 
        self.metrics["f1_score"] = f1_score(y_true, y_pred_class)
        self.metrics["auc"] = roc_auc_score(y_true, y_predics)
        self.metrics["specificity"] = tn / (tn + fp)     # Identify negatives 
        self.metrics["fpr"] = fp / (fp + tn)             # identify false positive rate
        self.metrics["fnr"] = fn / (fn + tp)             # identify false negative rate


# Save output data
class Save_Data:
    '''Save models data: model information, evaluation metrics, data and comments'''
    
    def __init__(self, file_name="models_data.json", out_directory="Outputs"):
        self.out_directory = out_directory
        self.file_name = file_name
        self.path = os.path.join(self.out_directory, self.file_name)
        
        # makes directory for outputs
        os.makedir(out_directory, exist_ok=True)

        # open json file, if exist, to load existing output data
        if os.path.exist(self.path):
            with open(self.path, 'r') as file:
                self.output_data = json.load(file)
        else:
            self.output_data = {}
        
    def add_model_data(self, model_name, model_path, metrics, project_phase, comments=""):
        '''Saves model data into a dictionary'''
        
        self.output_data[model_name] = {
            'time_stamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': model_path,
            'evaluation_metrics': metrics,
            'project_phase': project_phase,
            'comments': comments
        }
        
        
    def save_model_data(self):
        '''Saves data dictionary into json file'''
        with open(self.path, 'w') as file:
                json.dump(self.output_data, file, indent=4)
        print(f"[INFO]Models data is saved to {self.path}")
        

# Models visualization 
class Visualization:
    def __init__(self, file_name="models_data.json", out_directory="Outputs"):
        self.out_directory = out_directory
        self.file_name = file_name
        self.path = os.path.join(self.out_directory, self.file_name)
        self.models_data = None

        # open json file to load existing output data
        if os.path.exist(self.path):
            with open(self.path, 'r') as file:
                self.models_data = json.load(file)
        else:
            print("No models are saved")

    def confusion_matrix(self, model_name):
        

    def line_plot(self):
        pass

    def learning_curves(self):
        pass

    def spider_radar(self):
        pass

    def ROC_curve(self):
        pass


