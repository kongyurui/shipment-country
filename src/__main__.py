import argparse
import os

import boto3

from acquire import DataAcquisition
from common import MODEL_BUCKET, MODEL_DIR
from training import ShipmentTrainer
from evaluate import Evaluator


def upload_model_files(version: str):
    """Upload all files associated with a specific model to S3"""
    s3_client = boto3.client('s3')
    model_path = os.path.join(MODEL_DIR, version)
    if not os.path.exists(model_path):
        logger.error("Specified model version does not exist", version=version, model_path=model_path)
        return
    for root, subdirs, files in os.walk(model_path):
        for file in files:
            s3_client.upload_file(os.path.join(root, file), MODEL_BUCKET, os.path.join(model_path, file))

def main(args):
    """Method to run the data processing/model building pipeline"""
    version = None

    if args.acquire or args.pipeline:
        da = DataAcquisition()
        da.acquire("ds-project-train.csv", "training")
        da.acquire("ds-project-validation.csv", "validation")
        
    if args.train or args.pipeline:
        trainer = ShipmentTrainer()
        version = trainer.train()

    if args.evaluate or args.pipeline:
        if not version and not args.version:
            logger.error("Must either train or specify model version to evaluate")
            
        if not version:
            version = args.version
        
        evaluator = Evaluator()
        evaluator.evaluate(version)
        
    if args.upload:
        if not version and not args.version:
            logger.error("Must either train or specify model version to upload model")
            
        if not version:
            version = args.version
        upload_model_files(version)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shipment Country Model')
    parser.add_argument('-a','--acquire', help='Run data acquisition', action='store_true')
    parser.add_argument('-t','--train', help='Run training', action='store_true')
    parser.add_argument('-e','--evaluate', help='Run evaluation', action='store_true')
    parser.add_argument('-p','--pipeline', help='Run full pipeline', action='store_true')
    parser.add_argument('-u','--upload', help='Upload model and metrics to S3', action='store_true')
    parser.add_argument('-v','--version', help='Model version to evaluate and/or upload')

    args = parser.parse_args()

    main(args)