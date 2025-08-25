import os
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Save output data
class Save_Data:
    '''Save models data: model information, evaluation metrics, data and comments'''
    
    def __init__(self, file_name="models_data.json", out_directory="Outputs"):
        self.out_directory = out_directory
        self.file_name = file_name
        self.path = os.path.join(self.out_directory, self.file_name)
        
        # makes directory for outputs
        os.makedirs(out_directory, exist_ok=True)

        # open json file, if exist, to load existing output data
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            try:
                with open(self.path, 'r') as file:
                    self.output_data = json.load(file)
            except json.JSONDecodeError:
                # if file is corrupted create empty dictionary 
                self.output_data = {}    
        else:
            self.output_data = {}
        
    def add_model_data(self, model_name, model_path, epochs, history, metrics, y_labels, project_phase, comments=""):
        '''Saves model data into a dictionary'''
        
        self.output_data[model_name] = {
            'time_stamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': model_path,
            'epochs': epochs,
            'history': history,
            'metrics': metrics,
            'labels': y_labels,
            'project_phase': project_phase,
            'comments': comments
        }
        
        
    def save_model_data(self):
        '''Saves data dictionary into json file'''

        def _make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()   # convert numpy array to list
            if isinstance(obj, (np.int32, np.int64)):  
                return int(obj)       # convert numpy ints to python ints
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)     # convert numpy floats to python floats
            if isinstance(obj, pd.core.series.Series):
                return obj.tolist()   # convert panda series to list
            if isinstance(obj, datetime):
                return obj.isoformat()
            
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(self.path, 'w') as file:
            json.dump(self.output_data, file, indent=4, default=_make_serializable)
        print(f"[INFO]Models data is saved to {self.path} \n\n")