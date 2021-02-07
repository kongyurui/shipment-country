import os
import re

import numpy as np
import pandas as pd
from structlog import get_logger

logger = get_logger()

class DataAcquisition():
    """This class formalizes the data loading and clean up initially developed in the DataAnalysis 
    Jupyter notebook associated with this project.  Only really needs to be run once for any new raw input 
    data set.
    """

    @staticmethod
    def acquire(csv_file : str = "ds-project-train.csv", dataset_type: str = "training"):
        """Read in source CSV file and clean up
        - remove rows with unknown country of origin
        - replace empty US.PORT with "UNKNOWN"
        - Write out to pickle file for use in training
        """
        csv_path = os.path.join("..", "data", "raw",  csv_file)
        shipment_df = pd.read_csv(csv_path)
        shipment_df.dropna(subset=['COUNTRY.OF.ORIGIN'], inplace=True)
        # Replace empty US.PORT with "UNKNOWN"
        shipment_df['US.PORT'].fillna('UNKNOWN', inplace=True)
        # Here also map the ARRIVAL.DATE to the internal pandas DateTime format
        shipment_df['ARRIVAL.DATE.PROCESSED']= pd.to_datetime(shipment_df['ARRIVAL.DATE'])
        
        shipment_df['WEIGHT..KG.'].fillna(0, inplace=True)
        
        processed_path = os.path.join("..", "data", "processed",  dataset_type + ".pkl")
        shipment_df.to_pickle(processed_path)

        logger.info("Total data points", dataset_type=dataset_type, count=shipment_df.shape[0])
        

        
        