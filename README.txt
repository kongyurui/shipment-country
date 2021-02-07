Erik Peterson
Altana Machine Learning Project

Please read DataAnalysis.pdf for my analysis of the training data, looking at the distribution of the source and target values, along with a description of how I cleaned the data and other conclusions.

Some principles I try to follow when building a machine learning project are:

- Consistency: Organize the common steps of the project in such as way that some one else (or future me) would be able to pick up and restart the project with minimal confusion.
- Tracking: Keep track of experiments and have explicit model versioning
- Portability: Try to make the project simple to use in different environments, either local or cloud.  For example, store data and model versions on cloud buckets for easy access and serving.  If I had more time, I'd dockerize the training and inference.


To run the system:

- Copy the ds-project-train.csv and ds-project-validation.csv files to the data/raw directory
  - Ideally these would be downloaded from a common source such as an S3 bucket.  I could create a role for Altana to have access to my AWS s3 bucket.

- Install the needed python modules.  Go to the top project directory and run:

> pip install -r requirements.txt

- Next change directories to "src".  You should be able to run the full project pipeline with one command (using the __main__.py file):

> python . -p

This processes and cleans the training and validation sets, runs the training script, and evaluates the model produced from the training scripts.

Each new model version is saved to a timestamped directory name in "models" which includes the model pickle file, the country label encoder, and the metrics on the validation set for that model.  The script also allows you to run each stage separately and to upload a model to S3 to use for a service (not yet included).