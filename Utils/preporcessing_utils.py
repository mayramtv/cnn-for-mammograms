import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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









        


