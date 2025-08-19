
import numpy as np
import os
from sklearn.metrics import confusion_matrix, auc
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.exposure import rescale_intensity

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
            
    @staticmethod
    def vis_plots(data):
        '''Plots one or more images or x-rays. Parameters:
            -data: a dictionary with the names as keys and image as values.
        '''
    
        # calculate num of columns and rows
        num_vis = len(data)
        if num_vis % 4 == 0:
            cols = 4
            rows = int(num_vis / cols)
        else:
            cols = 3
            rows = int(np.ceil(num_vis / cols))
    
        # initiate subplots
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 4))
        axs = axs.flatten()
    
        # iterate models to display x-rays images
        for i, (name,image) in enumerate(data.items()):
            axs[i].imshow(image, cmap='gray')
            axs[i].set_title(f"{name}")
            axs[i].axis("off")
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        
        # remove not used axes
        for ax in range(num_vis, len(axs)):
            fig.delaxes(axs[ax])
    
        # show plot
        plt.tight_layout()
        plt.show()

    @staticmethod
    def img_difference(orig_img, new_img):
        '''Calculate the difference between two images'''
    
        # Ensure original stays intact
        orig_img = orig_img.copy()
        new_img = new_img.copy()
        
        # calculate Peak to Signal Noise Ratio
        psnr_val = psnr(orig_img, new_img)
        print("PSNR Value:", psnr_val)
    
        # calcualte absolute value and rescale
        diff_abs = np.abs(orig_img - new_img)
        diff_rescaled = rescale_intensity(diff_abs, in_range='image', out_range=(0,1))
    
        return diff_rescaled

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

    def learning_curves(self, history):
        pass