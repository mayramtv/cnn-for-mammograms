# from Utils.preporcessing_gen_utils import image_iterators  # comment this line to try model with tf.Dataset generator(next line)
# from Utils.preporcessing_utils import image_iterators    # comment this line to try model with tf.ImageDataGenerator generator(previus line)
from Utils.preporcessing_utils_Copy1 import image_iterators
from Utils.models_utils import Basic_Custom_CNN
from Utils.models_utils import Dynamic_Custom_CNN
from Utils.evaluation_utils import Evaluation
from Utils.save_data_utils import Save_Data

from tensorflow.keras import backend as K
from pathlib import Path

def run_model(data_sets, 
              techniques_groups, 
              project_phase,
              iteration,
              with_preprocess=False, 
              model_type="custom",
              is_resnet_vgg=False,
              epochs=10, 
              models_settings=None,
              change=""):
    # datasets
    train_data, val_data, test_data = data_sets

    # y_labels
    y_true = test_data["label"]
    
    # iterate trough techniques groups for training a model with each group
    for technique_name, techniques in techniques_groups.items():
        
        # reset and clears variables before creating a new model 
        K.clear_session()
        
        # Create image iterators with preprocessing function for each set of preprocessing techniques 
        train_generator, val_generator, test_generator = image_iterators((train_data, val_data, test_data),
                                                                            with_preprocess=with_preprocess,
                                                                            is_resnet_vgg=is_resnet_vgg,
                                                                            preprocessing_techniques_name=technique_name,
                                                                            preprocessing_techniques=techniques
                                                                          )
        if model_type == "custom":
            # create model name
            model_name = model_type.title() + str(epochs) + " - " + technique_name + " - " + change
            print("Training " + model_name)
    
            # create files name
            f_name = model_name.lower().replace(" ", "_") 
        
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
            keras_name = f_name + ".keras"
            m_path_dir = "Models/Iteration" + str(iteration)
            model_path = model_instance.save_model(models_directory=m_path_dir, 
                                                   model_file=keras_name)
        
            # evaluate model by making predictions
            evaluation = Evaluation(model_instance.get_model())
            y_probs = evaluation.predict(test_generator)
        
            # calculate metrics
            metrics = evaluation.calculate_metrics(y_true, y_probs)
        
            # get labels dictionary
            y_labels = evaluation.get_labels()
        
            # save data
            json_name = f_name + ".json"
            out_path_dir = "Outputs/Iteration" + str(iteration)
            save_data = Save_Data(file_name=json_name, out_directory=out_path_dir)
            save_data.add_model_data(model_name, 
                                     model_path, 
                                     epochs, 
                                     history, 
                                     metrics, 
                                     y_labels, 
                                     project_phase, 
                                     comments="")
            save_data.save_model_data()
        
        elif model_type == "dynamic":

            # passes settings for model
            if models_settings is None:
                models_settings = {"Baseline_2(No Dropout)": {"epochs":10, "layers": [32, 64, 128], "activation": 'relu', "dense_units":None, "dropout": None}}
                print("No model settings were provided. The model will use default settings: \n", models_settings)
            
            for m_name, model_settings in models_settings.items():
                # reset and clears variables before creating a new model 
                K.clear_session()

                # initiate model
                model_instance = Dynamic_Custom_CNN(input_shape=(256, 256, 1), 
                                                    num_classes=1, 
                                                    epochs=model_settings["epochs"], 
                                                    layer_sizes=model_settings["layers"], 
                                                    activation=model_settings["activation"], 
                                                    dense_units=model_settings["dense_units"], 
                                                    dropout=model_settings["dropout"])

                # create model name
                model_name = model_type.title() + str(model_settings["epochs"]) + " - " + technique_name + " - " + m_name
                print("Training " + model_name)
        
                # create files name
                f_name = model_name.lower().replace(" ", "_")
    
                # create model architecture
                model_instance.architecture()
                model_instance.get_model().summary()
                
                # train model
                history = model_instance.train_model(train_generator, 
                                                     val_gen=val_generator)
                
                # save model and get path
                keras_name = f_name + ".keras"
                m_path_dir = "Models/Iteration" + str(iteration)
                model_path = model_instance.save_model(models_directory=m_path_dir, 
                                                       model_file=keras_name)
            
                # evaluate model by making predictions
                evaluation = Evaluation(model_instance.get_model())
                y_probs = evaluation.predict(test_generator)
            
                # calculate metrics
                metrics = evaluation.calculate_metrics(y_true, y_probs)
            
                # get labels dictionary
                y_labels = evaluation.get_labels()
            
                # save data
                json_name = f_name + ".json"
                out_path_dir = "Outputs/Iteration" + str(iteration)
                save_data = Save_Data(file_name=json_name, out_directory=out_path_dir)
                save_data.add_model_data(model_name, 
                                         model_path, 
                                         epochs, 
                                         history, 
                                         metrics, 
                                         y_labels, 
                                         project_phase, 
                                         comments="")
                save_data.save_model_data()

        elif model_type == "VGG16":
            # passes settings for model
            if models_settings is None:
                models_settings = {"Baseline_2(No Dropout)": {"epochs":10, "activation": 'relu', "dense_units":None, "dropout": None}}
                print("No model settings were provided. The model will use default settings: \n", models_settings)
            
            for m_name, model_settings in models_settings.items():
                # reset and clears variables before creating a new model 
                K.clear_session()

                # initiate model
                model_instance = VGG16_Transfer(input_shape=(224, 224, 3), 
                                                    num_classes=1, 
                                                    epochs=model_settings["epochs"], 
                                                    activation=model_settings["activation"], 
                                                    dense_units=model_settings["dense_units"], 
                                                    dropout=model_settings["dropout"])

                # create model name
                model_name = model_type.title() + ":" + str(model_settings["epochs"]) + " - " + technique_name + " - " + m_name
                print("Training " + model_name)
        
                # create files name
                f_name = model_name.lower().replace(" ", "_")
    
                # create model architecture
                model_instance.architecture()
                model_instance.get_model().summary()
                
                # train model
                history = model_instance.train_model(train_generator, 
                                                     val_gen=val_generator)
                
                # save model and get path
                keras_name = f_name + ".keras"
                m_path_dir = "Models/Iteration" + str(iteration)
                model_path = model_instance.save_model(models_directory=m_path_dir, 
                                                       model_file=keras_name)
            
                # evaluate model by making predictions
                evaluation = Evaluation(model_instance.get_model())
                y_probs = evaluation.predict(test_generator)
            
                # calculate metrics
                metrics = evaluation.calculate_metrics(y_true, y_probs)
            
                # get labels dictionary
                y_labels = evaluation.get_labels()
            
                # save data
                json_name = f_name + ".json"
                out_path_dir = "Outputs/Iteration" + str(iteration)
                save_data = Save_Data(file_name=json_name, out_directory=out_path_dir)
                save_data.add_model_data(model_name, 
                                         model_path, 
                                         epochs, 
                                         history, 
                                         metrics, 
                                         y_labels, 
                                         project_phase, 
                                         comments="")
                save_data.save_model_data()

