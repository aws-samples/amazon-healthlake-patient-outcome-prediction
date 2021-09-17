import boto3
from boto3.session import Session
import io
import os
import re
import time
import pandas as pd
import numpy as np
import json
from urllib.parse import urlparse
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tarfile
from tensorflow.keras import backend as K
from tensorflow import keras


def read_csv_s3(bucket, prefix, file_name, header=0, index_col=None):
    """Read csv file from s3 location. 
    This function uses boto3 and s3 client to get the object from S3 bucket, and then use pandas to read in the csv file. Pandas' read_csv function supports using s3 uri, but is not the preferred way of reading csv files.
    
    Args:
        bucket (str): bucket name
        predix (str): path to file
        key (str): path to the file
    Returns:
        pandas.DataFrame: a dataframe that contains the information in the csv file
    """
    s3 = boto3.client('s3') 
    obj = s3.get_object(Bucket= bucket, Key= '{}/{}'.format(prefix, file_name))
    df = pd.read_csv(obj['Body'], header=header, index_col=index_col)
    
    return df

def download_csv_s3(bucket, prefix, file_name, new_file_name=False):
    """Download csv file from s3 location. 
    This function uses boto3 and s3 client to download a file from S3 bucket.
    
    Args:
        bucket (str): bucket name
        predix (str): path to file
        key (str): path to the file
    Returns:
        file_name: str
    """
    s3 = boto3.client('s3') 
    s3.download_file(bucket, '{}/{}'.format(prefix, file_name), new_file_name if new_file_name else file_name)

    return (file_name, new_file_name)

class EpochLogger(CallbackAny2Vec):
    """EpochLogger is used during unsupervised embedding training to monitor the progress. This class also need to be imported when reading the trained embedding matrix"""
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1
        
def load_embedding_matrix_trained_model(model_name='models/w2v_dim100_win100.model', embedding_dim=100):
    """Load embedding matrix from .model file. This function is used for loading the embedding matrix that we trained using gensim.
    
    Args: 
        model_name (str): the path and name of the file
        embedding_dim (int): the dimension of the embedding matrix. Default value: 100, to be consistent with the pre-trained embedding matrix.
        
    Returns: 
        tuple: tuple containing
            embedding_matrix (numpy.array): numpy.array of the embedding matrix\n
            idx_to_code_map (dict): dictionary mapping from index to medical codes\n
            code_to_idx_map (dict): reversed dictionary mapping from medical codes to index\n
            unkown_idx (str): a token that mapps to 'unkown'
    """
    w2v_model = Word2Vec.load(model_name)
    vocab = w2v_model.wv.key_to_index.keys()
    code_to_idx_map = {}
    idx_to_code_map = {}
    embedding_matrix = np.zeros(shape=(len(vocab), embedding_dim))

    for idx, code in enumerate(vocab):
        code_to_idx_map[code] = idx
        idx_to_code_map[idx] = code
        embedding_matrix[idx,:] = w2v_model.wv.key_to_index[code]

    unkown_idx = idx+1
    idx_to_code_map[unkown_idx] = 'UNK'
    code_to_idx_map['UNK'] = unkown_idx
    embedding_matrix = np.append(embedding_matrix, np.array([0]*embedding_matrix.shape[1]).reshape(1, embedding_matrix.shape[1]), axis=0)
    
    return embedding_matrix, idx_to_code_map, code_to_idx_map, unkown_idx

def df2tensor(test_input, test_static_input, maxlen, model_name='models/w2v_dim100_win100.model'):
    """Convert test dataframes to tensors.
    
    Args:
        test_input (str): path to the test file. Use previous batch-transform step's output path if possible. eg: s3://<bucket>/<prefix>/test.csv
        test_static_input (str): path to the static part of the test data. Use previous batch-transform step's output path if possible.
        
    Returns:
        str: a string of input tensor in jsonline format (each line is a json object)
    """
    parsed_url = urlparse(test_input)
    bucket_name = parsed_url.netloc
    prefix = parsed_url.path[1:-9]
    
    df_test = read_csv_s3(bucket_name, prefix, 'test.csv')
    
    parsed_url = urlparse(test_static_input)
    bucket_name = parsed_url.netloc
    prefix = parsed_url.path[1:-16]
    
    df_static_test = read_csv_s3(bucket_name, prefix, 'test-static.csv')
    _, _, code_to_idx_map, unkown_idx = load_embedding_matrix_trained_model(model_name)

    encoded_seq = df_test['events'].apply(lambda x: [code_to_idx_map.get(i, unkown_idx) for i in x.split(' ')])
    X_test = pad_sequences(encoded_seq, maxlen=maxlen, padding='pre')
    X_static_test = df_static_test.values

    input_tensors = ''''''
    for e in zip(X_test, X_static_test):
        input_tensor = {'event_seq':e[0].tolist(),'static':e[1].tolist()}
        input_tensors+=json.dumps(input_tensor)
        input_tensors+='\n'
    input_tensors = input_tensors[:-1] # drop the last '\n' 
    
    return input_tensors


def get_csv_output_from_s3(s3uri, file_name):
    """Read model output file from s3 location. The output file from our model is a json-like object with each predicted probability is usually an element in a list of list, and therefore, there is some customization for making 
    
    Args: 
        s3uri (str): The uri of rhe output file from the model.
        file_name (str): the name of the output file from the model
        
    Returns:
        list: prediction results
    """
    parsed_url = urlparse(s3uri)
    bucket_name = parsed_url.netloc
    prefix = parsed_url.path[1:]
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket_name, '{}/{}'.format(prefix, file_name))
    output=obj.get()["Body"].read().decode('utf-8')
    output = output.replace('}{', '},{')
    output = '['+output+']'
    return json.loads(output)[0]['predictions']

def f1(y_true, y_pred):
    """Calculate f1 score based on precision/recall metrics defined within the function
    
    Args:
        y_true (numpy.array): the true labels of the data
        y_pred (numpy.array): the predicted labels of the data
        
    Returns:
        numpy.array: f1 scores
    """
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'), 
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    f1
]

def load_trained_tf_model(sagemaker_session, job_name):
    model_artifact_path = sagemaker_session.describe_training_job(job_name)['ModelArtifacts']['S3ModelArtifacts']
    bucket = model_artifact_path[5:].split('/')[0]
    prefix = '/'.join(model_artifact_path[5:].split('/')[1:-1])
    file_nm = model_artifact_path[5:].split('/')[-1]
    download_csv_s3(bucket, prefix, file_nm)
    with tarfile.open(file_nm, 'r:*') as f:
        f.extractall()
        f.close()
        tf_model = tf.keras.models.load_model('1', custom_objects={'f1': f1}, compile=False)
        tf_model.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.Adam(), metrics=METRICS)
        
    return tf_model