import boto3
import os
import numpy as np
import pandas as pd
import argparse
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


from config import RANDOM_STATE, MAX_LENGTH, EPOCHS, BATCH_SIZE, EMBEDDING_DIM,  filters, kernel_size
# Import tensorflow lib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, Conv1D, GlobalMaxPool1D, concatenate, MaxPooling1D, Lambda
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import backend as K

SM_CHANNEL_EMBEDDING = '/opt/ml/input/data/embeddings'
SM_CHANNEL_TRAIN_STATIC = '/opt/ml/input/data/train-static'
SM_CHANNEL_VALID_STATIC = '/opt/ml/input/data/valid-static'
filters = 3
kernel_size = 3

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

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=20,
    mode='max',
    restore_best_weights=True)

def _get_class_weight(labels):
    pos = sum(labels)
    total = len(labels)
    neg = total - pos
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    return class_weight

def cnn_model(embedding_matrix, vocab_len, emb_dim, em_input_len, static_feature_dim):
    """Defines the CNN model structure. Please refer to the reports for a visulization of the CNN model.
    This function defines the structure of the CNN model. 
    
    Args:
        embedding_matrix (numpy.array): the embeeding matrix from gensim 
        vocab_len (int): vocabulary length, number of unique medical events available in the embeddings
        emb_dim (int): embedding dimensions, default value: 300
        em_input_length (int): intput length of the medical events, same value as the max_length as we pre-padded all event sequence to that length.
        static_feature_dim (int): number of static features (statistical features), including labs, comorbidities, hospitalizations.
        
    Returns:
        tf.keras.model: model object
    """
    seq_input = keras.Input(shape=(None,), name='event_seq') 
    static_input = keras.Input(shape=(static_feature_dim), name='static')

    # Embed each event in the sequence into a 100-dimensional vector
    seq_features = Embedding(vocab_len, 
                             emb_dim, 
                             weights=[embedding_matrix], 
                             input_length=em_input_len, trainable=False)(seq_input)

    seq_features = Conv1D(filters, kernel_size)(seq_features)
    seq_features = MaxPooling1D(pool_size=em_input_len-filters+1, strides=1)(seq_features)
    seq_features = Lambda(lambda s: K.squeeze(s, axis=1))(seq_features)
    seq_features = Dense(16, activation='relu')(seq_features)
    seq_features = Dropout(0.5)(seq_features)

    # Merge all available features into a single large vector via concatenation
    x = concatenate([seq_features, static_input])
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    # Create model
    model = keras.Model(inputs=[seq_input, static_input],
                        outputs=output)

    return model

def make_model(X_train, X_train_static, y_train, X_valid, X_valid_static, y_valid, embedding_matrix, max_length, static_f_dim, class_weight, epochs, batch_size, learning_rate):
    """Main function to build the model, define optimizer, compile the model and start training the model.
    
    Args:
        X_train (numpy.array): training data features
        X_train_static (numpy.array): training data's required static/statistical features like labs, comorbidities, hospitalizations
        y_train (numpy.array): labels of the training data
        X_valid (numpy.array): validation data features
        X_valid_static(numpy.array): validation data's required static/statistical features like labs, comorbidities, hospitalizations
        y_valid (numpy.array): labels of the validation data
        embedding_matrix(numpy.array): the embeeding matrix generated by gensim
        max_length (int): max length of the sequence/medical events
        static_f_dim (int): number of static features (statistical features), including labs, comorbidities, hospitalizations
        class_weight (dict): dictionary of class weight generated from _get_class_weight(). This is used to deal with the imbalanceness.
        epochs (int): number of epochs of training process, default is 50
        batch_size (int): number of batch size, default is 512
        learning_rate (float): learning rate of the model, default is 0.001
        
    Returns:
        tf.keras.model: a fitted model
    
    """
    model = cnn_model(embedding_matrix, embedding_matrix.shape[0], embedding_matrix.shape[1], max_length, static_f_dim)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=METRICS)
    
    weighted_history = model.fit(
        {'event_seq':X_train, 'static':X_train_static},
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        validation_data = [{'event_seq':X_valid, 'static':X_valid_static}, y_valid],
        class_weight=class_weight)

    return model

def _load_training_data(base_dir, code_to_idx_map, unkown_idx, max_length, target_col):
    train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    encoded_seq = train_df['events'].apply(lambda x: [code_to_idx_map.get(i, unkown_idx) for i in x.split(' ')])
    X_train = pad_sequences(encoded_seq, maxlen=max_length, padding='pre')
    y_train = train_df[target_col].values
    X_train_static = pd.read_csv(os.path.join(base_dir, 'train-static.csv')).values
    
    return X_train, X_train_static, y_train

def _load_validation_data(base_dir, code_to_idx_map, unkown_idx, max_length, target_col):
    valid_df = pd.read_csv(os.path.join(base_dir, 'validation.csv'))
    encoded_seq = valid_df['events'].apply(lambda x: [code_to_idx_map.get(i, unkown_idx) for i in x.split(' ')])
    X_valid = pad_sequences(encoded_seq, maxlen=max_length, padding='pre')
    y_valid = valid_df[target_col].values
    X_valid_static = pd.read_csv(os.path.join(base_dir, 'valid-static.csv')).values
    
    return X_valid, X_valid_static, y_valid
    
    
class EpochLogger(CallbackAny2Vec):
    """EpochLogger is used during unsupervised embedding training to monitor the progress. This class also need to be imported when reading the trained embedding matrix"""
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1
        
def _load_embeddings(base_dir, embedding_dim=300):
    w2v_model = Word2Vec.load(os.path.join(base_dir, 'W2V_event_dim100_win100.model'))
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

def _parse_args():
    parser = argparse.ArgumentParser()
    # Data, model, and output directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--max-length', type=int, default=150)
    parser.add_argument('--static-f-dim', type=int, default=0)
    parser.add_argument('--target-col', type=str, default='target')
    
    return parser.parse_known_args()

if __name__ == "__main__":
    args,_ = _parse_args()
    embedding_matrix, idx_to_code_map, code_to_idx_map, unkown_idx = _load_embeddings(args.training)
    X_train, X_train_static, y_train = _load_training_data(
        args.training, code_to_idx_map, unkown_idx, args.max_length, args.target_col)
    X_valid, X_valid_static, y_valid = _load_validation_data(
        args.validation, code_to_idx_map, unkown_idx, args.max_length, args.target_col)

    class_weight = _get_class_weight(y_train)
    tf_model = make_model(X_train, X_train_static, y_train, 
                          X_valid, X_valid_static, y_valid,
                          embedding_matrix, args.max_length, args.static_f_dim,
                          class_weight,
                          args.epochs, args.batch_size, args.learning_rate)
    
    version = 1
    export_path = os.path.join(args.sm_model_dir, str(version))
    tf_model.save(
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format='tf'
) 