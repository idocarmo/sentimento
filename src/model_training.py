import logging
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data import VOCAB_SIZE
from data import DataPipeline

logging.basicConfig(
    level=logging.INFO,
    format = '%(message)s'
    )
logging.info('\nLoading Variables...')


EMBEDDING_DIM = 2
EPOCHS = 10

def create_model():
    '''Create a embedding model.

    Create a Neural betwork using keras Sequential
    
    Args:
        None.

    Returns:
        a keras Neural Network model. 
    '''
    model = keras.Sequential([
      layers.Embedding(VOCAB_SIZE + 1, EMBEDDING_DIM, input_length=256),
      layers.Flatten(),
      layers.Dropout(rate=0.5),
      layers.Dense(5),
      layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.metrics.BinaryAccuracy(), tf.metrics.Precision(), tf.metrics.Recall()])
    
    model.summary()

    return model


def main():
    '''Model training script
    '''
    logging.info('\nLoadind datasets...\n')
    data_pipeline = DataPipeline()
    train_dataset, validation_dataset, test_dataset = data_pipeline.load_data()

    train_data, train_labels = data_pipeline.split_data_target(train_dataset)
    validation_data, validation_labels = data_pipeline.split_data_target(validation_dataset)

    logging.info('\n\tVectorizing training data...\n')
    X_train = data_pipeline.fit_transform(train_data)
    
    logging.info('\n\tVectorizing validation data...\n')
    X_val = data_pipeline.transform(validation_data)
    logging.info('Done.')

    logging.info('\nCreating embedding model...\n')
    model = create_model()


    logging.info('\nTraining model...\n')
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=1)
    history = model.fit(
        X_train,
        train_labels,
        epochs=EPOCHS,
        batch_size=100,
        validation_data=(X_val, validation_labels),
        callbacks=[callback],
        verbose=1
    )

    history_dict = history.history

    return
    

if __name__ == '__main__':
    main()