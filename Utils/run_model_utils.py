# from Utils.preporcessing_gen_utils import image_iterators  # comment this line to try model with tf.Dataset generator(next line)
# from Utils.preporcessing_utils import image_iterators    # comment this line to try model with tf.ImageDataGenerator generator(previus line)
from Utils.preporcessing_utils_Copy1 import image_iterators
from Utils.models_utils import Basic_Custom_CNN
from Utils.evaluation_utils import Evaluation
from Utils.save_data_utils import Save_Data

from tensorflow.keras import backend as K

def run_model(data_sets, techniques_groups, with_preprocess=False, epochs=10, project_phase="P1", change=""):
    # datasets
    train_data, val_data, test_data = data_sets

    # y_labels
    y_true = test_data["label"]
    
    # iterate trough techniques groups for training a model with each group
    for technique_name, techniques in techniques_groups.items():
        
        # create model name
        model_name = "Custom" + str(epochs) + " - " + technique_name + " - " + change
        print("Training " + model_name)

        # create files name
        name = model_name.lower().replace(" ", "_") 
        
        # reset and clears variables before creating a new model 
        K.clear_session()
        
        # Create image iterators with preprocessing function for each set of preprocessing techniques 
        train_generator, val_generator, test_generator = image_iterators((train_data, val_data, test_data),
                                                                            with_preprocess=with_preprocess,
                                                                            is_resnet_vgg=False,
                                                                            preprocessing_techniques_name=technique_name,
                                                                            preprocessing_techniques=techniques
                                                                          )
        
        # initiate model class
        model_instance = Basic_Custom_CNN(input_shape=(256, 256, 1), 
                                          num_classes=1, 
                                          epochs=epochs)
        
        # create model architecture
        model_instance.architecture()
        model_instance.get_model().summary()
        
        # train model
        history = model_instance.train_model(train_generator, 
                                             val_gen=val_generator)
        
        # save model and get path
        keras_name = name + ".keras"
        model_path = model_instance.save_model(models_directory="Models", 
                                               model_file=keras_name)
    
        # evaluate model by making predictions
        evaluation = Evaluation(model_instance.get_model())
        y_probs = evaluation.predict(test_generator)
    
        # calculate metrics
        metrics = evaluation.calculate_metrics(y_true, y_probs)
    
        # get labels dictionary
        y_labels = evaluation.get_labels()
    
        # save data
        json_name = name + ".json"
        save_data = Save_Data(file_name=json_name, out_directory="Outputs")
        save_data.add_model_data(model_name, 
                                 model_path, 
                                 epochs, 
                                 history, 
                                 metrics, 
                                 y_labels, 
                                 project_phase, 
                                 comments="")
        save_data.save_model_data()


