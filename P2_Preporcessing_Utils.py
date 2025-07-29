import pandas as pd

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
    
def labels_encoding():
    pass

def split_train():
    pass

def image_iterator(d_set, suffle=False):
    pass


# Basic custom model

class Basic_Custom_CNN:
    def __init__(self, phase, input_shape, train_data, val_data, test_data, epochs):
        self.phase = phase
        self.input_shape = input_shape
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.epochs = epochs
    
    def custom_CNN_architecture(self):
        pass
    
    def custom_CNN_train(self):
        pass
    
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
    


