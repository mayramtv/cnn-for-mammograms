import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,  f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
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

    def background_removal(image):
        # smooth image
        blur_img = cv2.GaussianBlur(image, (5,5), 0)
    
        # gets Otsu threshold 
        _, thresh = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    
        # apply morphological closing to make sure parts of the breast are not removed 
        kernel = np.ones((15, 15), np.uint8) 
        closed_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
        # Identify connecting regions for each pixel edge and corner (8) of binary image 
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_img, connectivity=8)
    
        # find the largest component that is connected
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # generate mask of black background
        mask = (labels == largest_label).astype(np.uint8) * 255
        breast_img = cv2.bitwise_and(image, image, mask=mask)

    return breast_img, mask

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
    
    
def split_data(train, test, val_size, stratify_col="label"):
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
        slef.labels = {}

    def evaluate(self, test_gen):
        '''Evaluate model and return metrics defined during model compile'''
        return self.model.evaluate(test_gen)

    def predict(self, test_gen):
        '''Makes predictions on unseen data'''
        return self.model.predict(test_gen) 
        

    def calculate_metrics(self, y_true, y_probs, conf_matrix):
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
        self.metrics["roc_auc"] = roc_auc_score(y_true, y_probs)
        self.metrics["specificity"] = tn / (tn + fp)     # Identify negatives 
        self.metrics["fpr"] = fp / (fp + tn)             # identify false positive rate
        self.metrics["fnr"] = fn / (fn + tp)             # identify false negative rate

        # saves y_true labels and predictions 
        self.labels["y_true"] = y_true
        self.labels["y_probs"] = y_probs


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
        
    def add_model_data(self, model_name, model_path, metrics, y_labels, project_phase, comments=""):
        '''Saves model data into a dictionary'''
        
        self.output_data[model_name] = {
            'time_stamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': model_path,
            'metrics': metrics,
            'labels': y_labels,
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

    def confusion_matrices(self, models_data, models_to_show, classes=None):
        '''
        Plot one or more confusion matrices 
        
        Parameters:
        - models data: a dictionary with each model and its respective data
        - models_to_show: a list of the models confusion matrices needed to be displayed
        - classes: a list of the classes ('Benigant', 'Malignant')
        '''
        if len(models_to_show) == 1:
            # gets model's data
            model_name = models_to_show[0]
            cm_data = models_data[model_name]["metrics"]["confusion_matrix"]
            # creates display for confusion matrix 
            # code inspiration from 
            # https://medium.com/@eceisikpolat/plot-and-customize-multiple-confusion-matrices-with-matplotlib-a19ed00ca16c
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_data, display_labels=classes)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix: {model_name}")
            plt.show()
        else:
            num_models = len(models_to_show)
            if num_models % 2 == 0:
                cols = 2
                rows = int(num_models/cols)
            else:
                cols = 3
                rows = int(np.ceil(num_models/cols))
            fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            axs = axs.flatten()
    
            # iterate models to display confusion matrices
            for i, model in enumerate(models_to_show):
                cm_data = models_data[model]["metrics"]["confusion_matrix"]
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_data, display_labels=classes)
                disp.plot(ax=axs[i], cmap=plt.cm.Blues)
                axs[i].set_title(f"{model}")
    
            # remove not used axes
            for ax in range(num_models, len(axs)):
                fig.delaxes(axs[ax])
    
            # show plot
            plt.tight_layout()
            plt.show()
        

    def line_plot(self, models_data, models_to_show, metrics):
        '''
        Plot one or more metrics in a line plot 
        
        Parameters:
        - models data: a dictionary with each model and its respective data
        - models_to_show: a list of the models confusion matrices needed to be displayed
        - metrics: a list of the metrics to add at the line plot.
        '''
    
        # plots each metric for each model
        for metric in metrics:
            y_vals = []
            for model in models_to_show:
                y_vals.append(models_data[model]["metrics"][metric]) 
            plt.plot(models_to_show, y_vals, label=metric, marker='o')
    
        # Add labels to the plot
        plt.xlabel('Models')
        plt.ylabel('Performance Score')
        plt.title('Performance Metrics Comparison Across Models')
        plt.legend()
        
        # Show plot
        plt.show()
    

    def radar_chart(self, models_data, models_to_show, metrics):
        # number of metrics
        num_metrics = len(metrics)
        
        # calculate the angles of each axis
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]
    
        # create plot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
        # plots each model for each metric
        for model in models_to_show:
            y_vals = []
            for metric in metrics:
                y_vals.append(models_data[model]["metrics"][metric]) 
            y_vals.append(y_vals[0])
            print(f"Model: {model} | angles: {len(angles)} | y_vals: {len(y_vals)}")
            ax.plot(angles, y_vals, label=model, marker='o')
            ax.fill(angles, y_vals, alpha=0.1)
    
        # Set labels for each metric
        ax.set_xticks(angles[:-1]) #removes closing tick
        ax.set_xticklabels(metrics)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'])
    
        # center axis
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        
        plt.title('Model Performance Comparison Across Metrics')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        plt.show()

    def ROC_curve(self, models_data, models_to_show):
        for model in models_to_show:
            # calculate false positive rate and true positive rate using the ROC curve 
            fpr, tpr, _ = roc_curve(models_data[model]["labels"]["y_true"], models_data[model]["labels"]["y_probs"])
            # get roc_auc value 
            roc_auc = models_data[model]["metrics"]["roc_auc"]
    
            # initiate plot
            plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')
    
        # format plot 
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curves Comparison')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc='lower right')
        plt.show()

    def learning_curves(self):
        pass
