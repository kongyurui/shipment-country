import re

import pandas as pd


class Preprocessor():
    @staticmethod
    def preprocess(df: pd.DataFrame):
        """Run common preprocessing on data frames for training and inference
        """
        # Dates by themselves won't be useful for predicting future events
        # Break date down into subcomponents that might be useful
        df['ARRIVAL.DATE.PROCESSED'] = pd.to_datetime(df['ARRIVAL.DATE'])
        df['ARRIVAL.DAY'] = df['ARRIVAL.DATE.PROCESSED'].dt.day
        df['ARRIVAL.MONTH'] = df['ARRIVAL.DATE.PROCESSED'].dt.month
        df['ARRIVAL.QUARTER'] = df['ARRIVAL.DATE.PROCESSED'].dt.quarter
        df['ARRIVAL.DAYOFWEEK'] = df['ARRIVAL.DATE.PROCESSED'].dt.dayofweek

        df['US.PORT'].fillna('UNKNOWN', inplace=True)
        df['WEIGHT..KG.'].fillna(0, inplace=True)

        # Product details has lots of unspaced text.  Add spaces between numbers and
        # letters to make tf-idf processing more effective
        df['PRODUCT.DETAILS.SPACED'] = df['PRODUCT.DETAILS'].apply(lambda x: re.sub("([A-Za-z])(\d)", r"\1 \2", re.sub(r"(\d)([A-Za-z])", r"\1 \2", str(x))))

        return df[['ARRIVAL.DAY', 'ARRIVAL.MONTH', 'ARRIVAL.QUARTER', 'ARRIVAL.DAYOFWEEK', 'US.PORT', 'WEIGHT..KG.', 'PRODUCT.DETAILS.SPACED']]

