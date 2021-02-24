import os
import pickle

import pandas as pd

from common import MODEL_DIR, MODEL_FILE, LABEL_FILE
from preprocess import Preprocessor

class Predictor():
    """Class to run inference over new shipment data to predict country of origin"""

    def __init__(self):
        self.preprocessor = Preprocessor()

    def load_model(self, version: str):
        "Load model and label encoder"
        model_path = os.path.join(MODEL_DIR, version, MODEL_FILE)
        self.model = pickle.load(open(model_path, "rb"))
        label_path = os.path.join(MODEL_DIR, version, LABEL_FILE)
        self.label_encoder = pickle.load(open(label_path, "rb"))

    def predict(self, instances: pd.DataFrame):
        "Preprocess and predict on incoming data frame"
        preprocessed_instances = self.preprocessor.preprocess(instances)

        predictions = self.model.predict(preprocessed_instances)

        return predictions