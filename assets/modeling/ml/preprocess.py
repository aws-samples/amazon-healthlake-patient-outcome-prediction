import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

label_column = 'readm'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
    df = pd.read_csv(os.path.join(args.train, 'train-static.csv'))
    # Impute median for static features
    preprocessor = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor.fit(df)
    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("preprocessor saved!")


def input_fn(input_data, content_type):
    """Parse input data payload

    Args:
        input_data: str, input data (csv) in a string format
        content_type: str, only 'text/csv' is supported
    Raise:
        ValueError: other file types are not supported
    Returns: 
        DataFrame
    """
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data))
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().
    """
    features = model.transform(input_data)
    return features


def model_fn(model_dir):
    """Deserialize fitted model
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor