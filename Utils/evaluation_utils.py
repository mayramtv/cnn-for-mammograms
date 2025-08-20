from sklearn.metrics import confusion_matrix, accuracy_score,  f1_score, precision_score, recall_score, roc_auc_score

# Models evaluation
class Evaluation:
    '''Make predictions and evaluation of the model and calculates evaluation metrics'''
    def __init__(self, model):
        self.model = model
        self.metrics = {}
        self.labels = {}

    def evaluate(self, test_gen):
        '''Evaluate model and return metrics defined during model compile'''
        return self.model.evaluate(test_gen)

    def predict(self, test_gen):
        '''Makes predictions on unseen data'''
        return self.model.predict(test_gen) 
        

    def calculate_metrics(self, y_true, y_probs):
        '''Claculate confusion matrix (True Negative, False Positive, False Negative and True Positive) 
        and evaluation metrics (Accuracy, Precision, Recall, F1 score, AUC
        Specificity, False Positive Rate, and False Negative Rate)'''
        # find predicted class
        if y_probs.ndim > 1 and y_probs.shape[1] == 2:
            y_probs = y_probs[:, 1]
        y_pred_class = (y_probs > 0.5).astype(int)

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

        return self.metrics

    def get_metrics(self):
        '''Returns model evaluation metrics'''
        return self.metrics

    def get_labels(self):
        '''Returns model labels'''
        return self.labels