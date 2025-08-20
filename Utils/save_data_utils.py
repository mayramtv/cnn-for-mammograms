import os
from datetime import datetime

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
        if os.path.exists(self.path):
            with open(self.path, 'r') as file:
                self.output_data = json.load(file)
        else:
            self.output_data = {}
        
    def add_model_data(self, model_name, model_path, history, metrics, y_labels, project_phase, comments=""):
        '''Saves model data into a dictionary'''
        
        self.output_data[model_name] = {
            'time_stamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': model_path,
            "history": history,
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