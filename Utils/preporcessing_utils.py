import pandas as pd
import numpy as np
from functools import partial
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Preprocessing techniques to try

def image_preprocessing(image, 
                       preprocessing_techniques,
                       is_resnet_vgg=False,
                       custom_cnn_size=256, 
                       resnet_vgg_size=224):
    '''
    Preprocessed an image based in the techniques passed as argument.
    Parameters:
        - image for preprocessing
        - preprocessing_techniques to be applied
        - is_resnet_vgg is True or False depending on type of model
        - custom_cnn_size is the size for the Custom CNN model input image to be resized
        - resnet_vgg_size is the size for the ResNet/VGG models input image to be resized"
    '''

    def background_removal(image):
        '''
            Removes edge of whole image, blur to find Otsu threshold, finds closed mask, 
            find the largest connected region and generate Otsu mask to remove background and leave only breast
        '''
        # resize to remove a contour of the whole image to remove some of the marks of x-rays that are not the breast
        height, width = image.shape[:2]
        image = image[45:height-45, 45:width-45]
    
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
    
        return closed_img, image, mask, breast_img

    

    def crop(image, breast_mask=None):    
        ''' Find contours of breast image using the mask.''' 
        # - RETR_EXTERNAL: defines only external countour of the biggest section, 
        # - CHAIN_APPROX_SIMPLE: saves only non redundant and the simplest points of the countour 
        # source: https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
        if breast_mask == None:
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if breast_mask:
            contours, _ = cv2.findContours(breast_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # checks for non-countour
        if len(contours) == 0:
            return image
    
        # find the countour area
        # source: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        area = max(contours, key=cv2.contourArea)
        
        # find the bounding box
        x, y, w, h = cv2.boundingRect(area)
    
        # crops image using bounding box
        cropped = image[y:y+h, x:x+w]
        
        return cropped       
    
     
    def noise_reduction(image):
        '''
        Noise removal using Wavelet with Soft Otsu Threshold: 
           - calculating coefficients of approximation and detail
           - calculating sigma value
           - calculating and applying soft thresholding value to coefficients 
           - and reconstructing image using coefficinets
        '''
        
        # Calculate coefficients for the image using wavedec2 for 2D (image) decomposition 
        # using the  Daubechies wavelet db1 (Haar wavelet of interval of 0-1)  
        init_coeffs = pywt.wavedec2(image, wavelet="db1", level=2)
    
        # calculate sigma of the detail coefficients
        sigma = estimate_sigma(image, channel_axis=None)
    
        # calculate initial threshold
        init_threshold = sigma * np.sqrt(2 * np.log2(image.size))
    
        # iterate through detail coefficients to apply threshhold function using the initial threshold
        new_coeffs = [init_coeffs[0]]
    
        for level in init_coeffs[1:]:
            tuple_vals = tuple(pywt.threshold(detail, init_threshold, mode='soft') for detail in level)
            new_coeffs.append(tuple_vals)
    
        # Reconstruct image using waverec2
        denoised_img = pywt.waverec2(new_coeffs, wavelet="db1")
        reconstructed = denoised_img[:image.shape[0], :image.shape[1]]
        
        return reconstructed

    

    def contrast_enhancement(image):
        '''
        Distribute gray peaks of image
        '''
        # code provenance 
        # https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
    
        # verify the image is in gray scale
        if len(image.shape) == 3:  # If color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # verify image is intiger type 
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = image.astype(np.uint8)
    
        # initialize CLAHE and apply to image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clh_image = clahe.apply(image)
        
        return clh_imag

    
    
    def edge_enhancement(guide_image, input_image, radius=8, epsilon = 0.0002):    
        '''
            Uses guided filter to smooth image and keep edges sharp. 
            Uses input image to be filter and as guide for pixel smoothness. Parameters are:
            - guided image
            - input image
            - kernel radius
            - regularization value
        '''
    
        # Normalize value
        guide_image = guide_image.astype(np.float32) / 255.0
        input_image = input_image.astype(np.float32) / 255.0
        
        # compute local mean: guidance and input image
        mean_guide = cv2.boxFilter(guide_image, -1, (radius, radius)) # depth is -1 for source depth
        mean_input = cv2.boxFilter(input_image, -1, (radius, radius)) # depth is -1 for source depth
        mean_g_i = cv2.boxFilter(guide_image * input_image, -1, (radius, radius)) # mean of product of both images
        mean_g_g = cv2.boxFilter(guide_image * guide_image, -1, (radius, radius))# loacl mean of product of guide image
        
        # calculate variance 
        covar_g_i = mean_g_i - mean_guide * mean_input # of the product of loacl means of both images
        var_g = mean_g_g - mean_guide * mean_guide # of guidance image 
    
        # calculate coefficinets using means , covariance and variance
        alpha = covar_g_i / (var_g + epsilon)
        beta = mean_input - alpha * mean_guide
    
        # calculate mean of coefficinets
        mean_alpha = cv2.boxFilter(alpha, -1, (radius, radius))
        mean_beta = cv2.boxFilter(beta, -1, (radius, radius))
    
        # apply mean of alpha to the guide image and add mean of beta
        filtered = mean_alpha * guide_image + mean_beta
    
        # convert back to intigers
        filtered = (filtered * 255).astype(np.uint8)
    
        return filtered

    
    
    def lbp_texturizer(image, n_points=8, radius=1):
        '''Calculate the local binary pattern of an image: checks for sourronding points of the kernel
            and gives a binary value depending if bigger or smaller than the center point. Parameters are:
            - image
            - number of points around center point
            - radius of kernel
        '''
        # convert to gray scale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # calculate local binary pattern
        lbp = local_binary_pattern(image, n_points, radius)
    
        # normalize image for visualization
        lbp_img = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min()))
    
        return lbp_img


        def resize(image, is_resnet_vgg=False, custom_cnn_size=256, resnet_vgg_size=224, is_lbp=False):
    
            '''
            Resize an image to fit a custom CNN, ResNet and VGG models
            Parameters:
                image: image to be resized
                custom_cnn: Boolean stating if the size is for custom CNN
                resnet_vgg: Boolean stating if the size is for ResNet/VGG models
            '''
        
            # convert image to tensorflow image
            img = tf.image.convert_image_dtype(image, tf.float32)
        
            # Adds  3rd channel
            if img.ndim == 2:
                img = tf.expand_dims(img, axis=-1)
        
            if is_lbp:
                    method = "nearest"
            else:
                method = "bilinear"
                
            if is_resnet_vgg == False:
                    
                # resize image and add pad to keep image proportions
                # source https://www.tensorflow.org/api_docs/python/tf/image/resize_with_pad
                resized_img = tf.image.resize_with_pad(img, custom_cnn_size, custom_cnn_size, method=method)
        
                # normalize image
                # source https://www.tensorflow.org/api_docs/python/tf/clip_by_value
                input_resized_img = tf.clip_by_value(resized_img, 0.0, 1.0)
                
            else:
                # converts image to RGB for Resnet/VGG input
                if img.shape[-1] == 1:
                    img = tf.image.grayscale_to_rgb(img)
                
                # resize image and add pad to keep image proportions
                # https://www.tensorflow.org/api_docs/python/tf/image/resize
                resized_img = tf.image.resize_with_pad(img, resnet_vgg_size, resnet_vgg_size, method=method)
        
                # normalize image
                # source https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/preprocess_input
                input_resized_img = tf.keras.applications.resnet50.preprocess_input(resized_img * 255.0)
        
            return input_resized_img

    img = image.copy()
    breast_mask = None
    is_lbp = False
    
    if preprocessing_techniques["apply_background_removal"] == True:
        _, _, breast_mask, img = background_removal(img)

    if preprocessing_techniques["apply_crop"] == True:
        img = crop(img, breast_mask)

    if preprocessing_techniques["apply_noise_reduction"] == True:
        img = noise_reduction_WT(img)

    if preprocessing_techniques["apply_contrast_enhancement"] == True:
        img = contrast_enhancement(img)

    if preprocessing_techniques["apply_edge_enhancement"] == True:
        img = edge_enhancement(img, img)

    if preprocessing_techniques["apply_lbp_texturizer"] == True:
        img = lbp_texturizer(img)
        is_lbp=True

    img = resize(img, 
                 is_resnet_vgg=is_resnet_vgg, 
                 custom_cnn_size=custom_cnn_size, 
                 resnet_vgg_size=resnet_vgg_size, 
                 is_lbp=is_lbp)

    return img
    

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
    

def image_iterators(data_sets, is_resnet_vgg=False, preprocessing_techniques=None):
    '''
        Generate a data generator for each dataset 
    '''
    if is_resnet_vgg:
        size = 224
    else:
        size = 256

    # function contain only the image and other arguments are frozen 
    preprocessing_function = partial(image_preprocessing, 
                       preprocessing_techniques,
                       is_resnet_vgg=is_resnet_vgg,
                       custom_cnn_size=size, 
                       resnet_vgg_size=size)
    
    # function for setup generators
    def data_generator(dataset, target_size=(256,256), shuffle=False, preprocessing_func=preprocessing_function):
        '''
        Generate a data generator for processing each image
        '''
        # initiate generators
        gen = ImageDataGenerator(preprocessing_function=preprocessing_func)
        data_gen = gen.flow_from_dataframe(
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
    train_data, val_data, test_data = data_sets
    
    train_gen = data_generator(data_sets, (size, size), True, preprocessing_func=preprocessing_function)
    val_gen = data_generator(val_data, (size, size), False, preprocessing_func=preprocessing_function)
    test_gen = data_generator(test_data, (size, size), False, preprocessing_func=preprocessing_function)
    
    return train_gen, val_gen, test_gen

def ablation(options):    
    '''
    Creates a dictionary with the group of techniques selected by using ablation
    '''
    # by using ablation, create the combinations of techniques 
    techniques_groups = {}
    techniques_groups["Baseline Basic Preporcessing"] = {option:False for option in options} # no techniques
    techniques_groups["All Preporcessing Techniques"] = {option:True for option in options} # all tecniuqes

    # removes one techniques at a time 
    for option in options:
        # creates the name of each technique
        group_name = option.split("_")[1:]
        group_name =  "No " + " ".join(group_name).capitalize()
        tech_group = {}
        # then uses techniques for applying a boolean
        for technique in options: 
            if technique != option:
                tech_group[technique] = True
            else:
                tech_group[technique] = False
                
        techniques_groups[group_name] = tech_group

    return techniques_groups







        


