import datetime
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from structlog import get_logger

from common import MODEL_DIR, MODEL_FILE, LABEL_FILE
from preprocess import Preprocessor

logger = get_logger()


class ShipmentTrainer():
    def __init__(self):
        self.preprocessor = Preprocessor()

    def _load_training_set(self, training_file: str = "../data/processed/training.pkl"):
        self.train_df = pd.read_pickle(training_file)

    def _create_model_path(self):
        """Store each model version in a separate directory, keep copy of config, model, label encoder and metrics"""
        today = datetime.datetime.utcnow()
        version = f'{today.year}{today.month:02d}{today.day:02d}T{today.hour:02d}{today.minute:02d}{today.second:02d}'

        model_path = os.path.join(MODEL_DIR, version)
        # If it does exist, maybe should just fail here or add arg to allow overwriting
        if os.path.exists(model_path):
            logger.error("Model path already exists", model_path=model_path)
            quit()

        os.makedirs(model_path)

        return version, model_path

    def _fit_label_encoder(self, model_path: str):
        """Replace low frequency countries with "OTHER"
        Map country names to index number
        Save encoder for use in prediction
        """

        # Replace countries occurring less than N times with "OTHER"
        self.train_df['COUNTRY.OF.ORIGIN'].fillna('OTHER', inplace=True)
        country_counts = pd.DataFrame(self.train_df['COUNTRY.OF.ORIGIN'].value_counts())
        common_countries = set(country_counts[country_counts['COUNTRY.OF.ORIGIN'] > 50].index)
        self.train_df['COUNTRY.OF.ORIGIN.MAPPED'] = self.train_df['COUNTRY.OF.ORIGIN'].apply(
            lambda x: 'OTHER' if x not in common_countries else x)

        # Write out label encoder for use in prediction
        label_encoder = LabelEncoder()
        country_labels = label_encoder.fit_transform(self.train_df['COUNTRY.OF.ORIGIN.MAPPED'])
        pickle.dump(label_encoder, open(os.path.join(model_path, LABEL_FILE), "wb"))

        return country_labels

    def train(self):
        """Method to train model.
        - Loads the training data
        - Creates a new directory for the model
        - Creates a pipeline for model training
        - Fits the model to the data
        - Saves the model to disk as a pickle file

        :returns: version - The version of the trained model
        """

        self._load_training_set()

        version, model_path = self._create_model_path()
        country_labels = self._fit_label_encoder(model_path)

        # Process input df to add arrival date derived features, clean up product details
        preprocessed_train_df = self.preprocessor.preprocess(self.train_df)

        # Includes US.PORT and the ARRIVAL.DATE derived day/month/quarter (could also try WEEKOFYEAR)
        # With more time, would be able to select the features in a separate configuration that can be
        # easily edited, or used for a grid search
        category_columns = ['US.PORT', 'ARRIVAL.DAY', 'ARRIVAL.DAYOFWEEK', 'ARRIVAL.MONTH', 'ARRIVAL.QUARTER']
        category_pipeline = Pipeline([
            ('cat_encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        numerical_columns = ['WEIGHT..KG.']
        numerical_pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree = 2)),
            ('scaler', StandardScaler())
        ])

        text_columns = 'PRODUCT.DETAILS.SPACED'
        text_pipeline = Pipeline([
            # Other options here would be to try different vocabulary sizes, bigrams
            ('text_encoder', TfidfVectorizer(min_df=10, max_features=10000))
        ])

        combined_pipeline = ColumnTransformer(transformers=[
            ('category_pipeline', category_pipeline, category_columns),
            ('numerical_pipeline', numerical_pipeline, numerical_columns),
            ('text_pipeline', text_pipeline, text_columns)
        ])

        self.model = Pipeline([
            ('combined_pipeline', combined_pipeline),
            # Upped max_iter since model wasn't converging within 100 iterations
            # Could also try other classification algorithms, different regularizations
            ('classification', LogisticRegression(class_weight='balanced', max_iter=1000)),  # , C=10
        ])

        self.model.fit(preprocessed_train_df, country_labels)
        logger.info("Score on training set", training_score=self.model.score(preprocessed_train_df, country_labels))

        # With more time, would also investigate which features were most useful for model prediction

        pickle.dump(self.model, open(os.path.join(model_path, MODEL_FILE), "wb"))

        return version