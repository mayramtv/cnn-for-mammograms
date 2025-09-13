import pandas as pd
import numpy as np
import os
import cv2
import pywt
from functools import partial
from skimage.restoration import estimate_sigma
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
        
    def convert_uint8(img):
        '''Convert image to intiger to work with cv2'''
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)
        return img

    def expand_dim(img):
        '''Expand dimentions if channel is not added'''
        if not is_resnet_vgg and img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        elif is_resnet_vgg and img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif is_resnet_vgg and img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        return img
            

    def background_removal(image):
        '''
            Removes edge of whole image, blur to find Otsu threshold, finds closed mask, 
            find the largest connected region and generate Otsu mask to remove background and leave only breast
        '''
        
        # normalize image and convert to UINT8 if needed
        image = convert_uint8(image)
        image = expand_dim(image)

        
        # resize to remove a contour of the whole image to remove some of the marks of x-rays that are not the breast
        height, width = image.shape[:2]
        image = image[45:height-45, 45:width-45, :]

    
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

        # expand dimentions
        breast_img = expand_dim(breast_img)
    
        return closed_img, image, mask, breast_img

    

    def crop(image, breast_mask=None):    
        ''' Find contours of breast image using the mask.''' 
        
        # normalize image and convert to UINT8 if needed
        image = convert_uint8(image)
        if breast_mask is not None:
            breast_mask = convert_uint8(breast_mask)
        
        # - RETR_EXTERNAL: defines only external countour of the biggest section, 
        # - CHAIN_APPROX_SIMPLE: saves only non redundant and the simplest points of the countour 
        # source: https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
        if breast_mask is None:
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
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
        
        # convert to float
        image = image.astype(np.float32)
        
        # calculate sigma of the detail coefficients    
        if not is_resnet_vgg:
            # adjust channels
            if image.ndim == 3 and image.shape[-1] == 1:
                image = np.squeeze(image)  
            sigma = estimate_sigma(image, channel_axis=None)
        else:
            sigma = estimate_sigma(image, channel_axis=-1)

        # Calculate coefficients for the image using wavedec2 for 2D (image) decomposition 
        # using the  Daubechies wavelet db1 (Haar wavelet of interval of 0-1)  
        init_coeffs = pywt.wavedec2(image, wavelet="db1", level=2)
            
    
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
        
        # expand dimentions
        reconstructed = expand_dim(reconstructed)
        reconstructed = reconstructed.astype(np.float32)
        
        # clip values between 0-255
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.float32)
        
        return reconstructed

    

    def contrast_enhancement(image):
        '''
        Distribute gray peaks of image
        '''
        # code provenance 
        # https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
        
        # normalize image and convert to UINT8 if needed
        image = convert_uint8(image)
    
        # verify the image is in gray scale
        if len(image.shape) == 3 and image.shape[-1] == 3:  # If color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # initialize CLAHE and apply to image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clh_image = clahe.apply(image)

        # expand dimentions
        clh_image = expand_dim(clh_image)
        
        return clh_image

    
    
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
        mean_alpha = np.expand_dims(mean_alpha, axis=-1)
        mean_beta = np.expand_dims(mean_beta, axis=-1)
    
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
        # normalize image and convert to UINT8 if needed
        image = convert_uint8(image)
        
        # convert to gray scale
        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # removes 3rd channel
        if img.ndim == 3 and image.shape[-1] == 1:  # If color image
            image = np.squeeze(image)
    
        # calculate local binary pattern
        lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    
        # normalize image for visualization
        lbp_img = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min()))

        # expand dimentions
        lbp_img = expand_dim(lbp_img)
    
        return lbp_img

    def resize(img, is_resnet_vgg=False, custom_cnn_size=256, resnet_vgg_size=224, is_lbp=False):
        """
        Resize an image to fit a custom CNN, ResNet, or VGG model using TensorFlow ops.
        Maintains aspect ratio, pads with zeros, and expands channels if needed.
        """
    
        # target size
        target_size = resnet_vgg_size if is_resnet_vgg else custom_cnn_size
    
        # get current shape
        shape = tf.shape(img)
        height, width = shape[0], shape[1]

        # make sure there is not division by zero
        max_dim = tf.maximum(tf.maximum(height, width), 1) 
    
        # compute resize ratio
        ratio = tf.cast(target_size, tf.float32) / tf.cast(tf.maximum(height, width), tf.float32)
        new_height = tf.cast(tf.round(tf.cast(height, tf.float32) * ratio), tf.int32)
        new_width = tf.cast(tf.round(tf.cast(width, tf.float32) * ratio), tf.int32)
    
        # resize with bilinear interpolation (similar to cv2.INTER_AREA)
        resized = tf.image.resize(img, [new_height, new_width], method="bilinear")
    
        # pad to target_size × target_size
        pad_height = tf.maximum(target_size - new_height, 0)
        pad_width = tf.maximum(target_size - new_width, 0)
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
    
        padded = tf.pad(resized, [[top, bottom], [left, right], [0, 0]], constant_values=0.0)
    
        # make sure shape is correct
        padded = tf.image.resize_with_crop_or_pad(padded, target_size, target_size)
    
        # if custom CNN expects grayscale channel
        if not is_resnet_vgg and not is_lbp and len(padded.shape) == 2:
            padded = tf.expand_dims(padded, axis=-1)
    
        return padded


    img = image
    breast_mask = None
    is_lbp = False
    
    if not is_resnet_vgg and img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    
    if preprocessing_techniques["apply_background_removal"] == True:
        _, _, breast_mask, img = background_removal(img)

    if preprocessing_techniques["apply_crop"] == True:
        img = crop(img, breast_mask)

    if preprocessing_techniques["apply_noise_reduction"] == True:
        img = noise_reduction(img)

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
    
    # check for correct size
    if is_resnet_vgg:
        corr_size = resnet_vgg_size
        chan = 3
    else:
        corr_size = custom_cnn_size
        chan = 1

    assert img.shape[:2] == (corr_size, corr_size), f"Incorrect shape {img.shape}, expected ({corr_size}, {corr_size}, {chan})"
        
    
    # convert image to tensorflow image
    img = tf.convert_to_tensor(img, dtype=tf.float32)

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

# preprocess image in place and saves locally in case iterators do not work
def preprocess_locally(dset, techniques_groups):

    # parent directory
    parent_dir_name = "Preprocessed_Images/"
    os.makedirs(parent_dir_name, exist_ok=True)
    
    for g_name, group in techniques_groups.items():

        # group directory for outputs
        name = "custom_" + g_name.lower().replace(" ", "_")
        im_dir = os.path.join(parent_dir_name, name)
        os.makedirs(im_dir, exist_ok=True)
        print("Saving images for", g_name, ".....")
        
        for path in dset["image_path"]:
            
            # loads image
            original_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            # perprocess image
            img = image_preprocessing(original_img.copy(), techniques_groups[g_name])        

            # convet image to numpy array and to to uint8 gray scale
            img = img.numpy()
            
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)    
            
            # creates path and saves image
            file_name = path.split("/")[-1]
            
            im_path = os.path.join(im_dir, file_name)
            cv2.imwrite(im_path, img)
        print("Finish saving images for", g_name)
    
# Create Image Iterators    
def dataset_builder(dataset, 
                    is_resnet_vgg=False, 
                    preprocessing_techniques_name=None, 
                    preprocessing_techniques=None, shuffle=False):
        
    # gets paths and labels from dataset
    paths = dataset["image_path"].values
    labels = dataset["label"].values

    # create images directory path
    parent_dir = "Preprocessed_Images/"
    im_dir_name = "custom_" + preprocessing_techniques_name.lower().replace(" ", "_")
    im_dir = os.path.join(parent_dir, im_dir_name)

    # update paths to preprocessed paths
    new_paths = []
    for path in paths:
        file_name = os.path.basename(path) 
        im_path = os.path.join(im_dir, file_name)
        new_paths.append(im_path)

        
    # create new dataset
    new_dataset = tf.data.Dataset.from_tensor_slices((new_paths, labels))

    # shuffles data
    if shuffle:
        new_dataset = new_dataset.shuffle(buffer_size=len(dataset), seed=42)

    # loads each 
    def image_handling(path, label):
        # loads original image
        image = tf.io.read_file(path)

        if is_resnet_vgg:
            # make sure image is in gray scale
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.resize(image, [224, 224])
        else:
            image = tf.image.decode_png(image, channels=1)
            image = tf.image.resize(image, [256, 256])

        # normalize image for model
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image, label

    preprocessed_dset = new_dataset.map(image_handling, num_parallel_calls=tf.data.AUTOTUNE)

    preprocessed_dset = preprocessed_dset.batch(32).prefetch(tf.data.AUTOTUNE)
    return preprocessed_dset

    


        
# Create Image Iterators    
def dataset_builder_preprocess(dataset, 
                               is_resnet_vgg=False, 
                               preprocessing_techniques_name=None, 
                               preprocessing_techniques=None, 
                               shuffle=False):
    # sets size
    if is_resnet_vgg:
        size = 224
    else:
        size = 256
        
    # function contain only the image and other arguments are frozen 
    preprocessing_function = partial(image_preprocessing, 
                                   preprocessing_techniques=preprocessing_techniques,
                                   is_resnet_vgg=is_resnet_vgg,
                                   custom_cnn_size=size, 
                                   resnet_vgg_size=size
                                    )
        
    # gets paths and labels from dataset
    paths = dataset["image_path"].values
    labels = dataset["label"].values
    new_dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    # shuffles data
    if shuffle:
        new_dataset = new_dataset.shuffle(buffer_size=len(dataset), seed=42)

    # loads each 
    # def image_handling(path, label):
    def preprocess(path, label):
        # loads original image
        image = tf.io.read_file(path)
        # make sure image is in gray scale
        image = tf.image.decode_png(image, channels=1)
        # normalize image for model
        image = tf.image.convert_image_dtype(image, tf.float32)

        # preprocess images
        image = tf.numpy_function(preprocessing_function, [image], tf.float32)
        
        # sets size
        if is_resnet_vgg:
            image.set_shape((224, 224, 3))
        else:
            image.set_shape((256, 256, 1))
            
        return image, label

    preprocessed_dset = new_dataset.map(image_handling, num_parallel_calls=tf.data.AUTOTUNE)

    preprocessed_dset = preprocessed_dset.batch(32).prefetch(tf.data.AUTOTUNE)
    return preprocessed_dset

def image_iterators(datasets, 
                    with_preprocess=False, 
                    is_resnet_vgg=False, 
                    preprocessing_techniques_name=None, 
                    preprocessing_techniques=None):
    # setup generators
    train_data, val_data, test_data = datasets
    
    # chooses type of iterator dataset builder 
    if not with_preprocess:
        train_dset = dataset_builder(train_data, 
                                 is_resnet_vgg=is_resnet_vgg, 
                                 preprocessing_techniques_name=preprocessing_techniques_name, 
                                 preprocessing_techniques=preprocessing_techniques, 
                                 shuffle=False)
        val_dset = dataset_builder(val_data, 
                                   is_resnet_vgg=is_resnet_vgg, 
                                   preprocessing_techniques_name=preprocessing_techniques_name, 
                                   preprocessing_techniques=preprocessing_techniques, 
                                   shuffle=False)
        test_dset = dataset_builder(test_data, 
                                    is_resnet_vgg=is_resnet_vgg, 
                                    preprocessing_techniques_name=preprocessing_techniques_name, 
                                    preprocessing_techniques=preprocessing_techniques, 
                                    shuffle=False)
    else:
        train_dset = dataset_builder_preprocess(train_data, 
                             is_resnet_vgg=is_resnet_vgg, 
                             preprocessing_techniques_name=preprocessing_techniques_name, 
                             preprocessing_techniques=preprocessing_techniques, 
                             shuffle=False)
        val_dset = dataset_builder_preprocess(val_data, 
                                   is_resnet_vgg=is_resnet_vgg, 
                                   preprocessing_techniques_name=preprocessing_techniques_name, 
                                   preprocessing_techniques=preprocessing_techniques, 
                                   shuffle=False)
        test_dset = dataset_builder_preprocess(test_data, 
                                    is_resnet_vgg=is_resnet_vgg, 
                                    preprocessing_techniques_name=preprocessing_techniques_name, 
                                    preprocessing_techniques=preprocessing_techniques, 
                                    shuffle=False)
    
    return train_dset, val_dset, test_dset

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







        


