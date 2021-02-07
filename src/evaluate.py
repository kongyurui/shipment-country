import os
import pickle

import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from structlog import get_logger

from common import DATA_DIR, MODEL_DIR, METRICS_FILE
from predict import Predictor

logger = get_logger()

class Evaluator():
    "Evaluate a specific model over a validation set, saving metrics with the model."
    
    def __init__(self):
        self.predictor = Predictor()
        
    def _load_eval_set(self, eval_file):
        self.eval_df = pd.read_pickle(eval_file)
        
    def evaluate(self, version):
        """- Load validation data frame
        - Maps previously unseed countries to OTHER
        - Preprocesses data frame for additional features
        - Run prediction
        - Calculate confusion matrix and classification report with precision/recall/f-score
          for each country
        - Save metrics to separate file in model directory for future reference (could also save as
          a json for easy automatic loading and comparison)
        """
        self.predictor.load_model(version)
        self._load_eval_set(f"{DATA_DIR}/validation.pkl")
        
        # Deal with issue of countries unseen in training data
        self.eval_df['COUNTRY.OF.ORIGIN'].fillna('OTHER', inplace=True)
        seen_countries = set(self.predictor.label_encoder.classes_)
        self.eval_df['COUNTRY.OF.ORIGIN.MAPPED'] = self.eval_df['COUNTRY.OF.ORIGIN'].apply(
            lambda x: 'OTHER' if x not in seen_countries else x)
        gold_labels = self.predictor.label_encoder.transform(self.eval_df['COUNTRY.OF.ORIGIN.MAPPED'])

        predictions = self.predictor.predict(self.eval_df)
    
        conf_mat = confusion_matrix(gold_labels, predictions)
        logger.info("Confusion matrix", confusion_matrix=conf_mat)

        report = metrics.classification_report(gold_labels, predictions, 
                                               target_names=self.predictor.label_encoder.classes_)
        logger.info("Classification report", report=report)
        
        metrics_path = os.path.join(MODEL_DIR, version, METRICS_FILE)
        # Having trouble getting full matrix to print, will do later
        with open(metrics_path, "w") as metrics_fd:
            metrics_fd.write(f"* Confusion matrix:\n{conf_mat}\n")
            metrics_fd.write(f"* Classification report:\n{report}")
        

        