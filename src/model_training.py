# -*- coding: utf-8 -*-

import logging
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses

from data import VOCAB_SIZE
from data import DataPipeline

from evaluation import plot_metric, generate_results_str

# Logging configuring
logging.basicConfig(
    level=logging.INFO,
    filename='model/results.log',
    format = '%(asctime)s %(name)s: %(levelname)s %(message)s'
    )
logging.info('\nLoading Variables...')


EMBEDDING_DIM = 2
EPOCHS = 10

def create_model():
    '''Create a embedding model.

    Create a Neural network using keras Sequential.
    
    Args:
        None.

    Returns:
        a keras Neural Network model. 
    '''
    model = keras.Sequential([
      layers.Embedding(VOCAB_SIZE + 1, EMBEDDING_DIM, input_length=512),
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

def compile_final_model(trained_model, data_pipeline):
    '''Create a final embedding model.

    Create a Neural network using the the pre-trained model
    for implementing the whole data processing pipeline on 
    the exported model.
    
    Args:
        trained_model: keras Neural Network trained model.
        data_pipeline: trained DataPipeline object.

    Returns:
        a keras Neural Network model. 
    '''

    final_model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        data_pipeline.vectorizer_layer,
        trained_model,  
    ])

    final_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False), 
        optimizer="adam", 
        metrics=[
            'accuracy', 
            tf.metrics.Precision(), 
            tf.metrics.Recall(), 
            tf.metrics.AUC()
        ]
    )

    return final_model


def main():
    '''Model training and evaluation script.
        
    Main function for the data processing and model training pipeline.

    Args:
        None

    Returns:
        None
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

    logging.info('\nPloting training metrics...\n')
    history_dict = history.history
    plot_metric(history_dict, metric='binary_accuracy', title='Binary Accuracy')
    plot_metric(history_dict, metric='precision', title='Precision')
    plot_metric(history_dict, metric='recall', title='Recall')
    plot_metric(history_dict, metric='loss', title='Loss')


    logging.info('\nCompiling final model...\n')
    final_model = compile_final_model(model, data_pipeline)

    logging.info('\nModel metrics...\n')
    training_results = final_model.evaluate(*data_pipeline.split_data_target(train_dataset), verbose=0)
    validation_results = final_model.evaluate(*data_pipeline.split_data_target(validation_dataset), verbose=0)
    test_results = final_model.evaluate(*data_pipeline.split_data_target(test_dataset), verbose=0)
    
    logging.info('  Training Results' + generate_results_str(training_results))
    logging.info('  Validation Results' + generate_results_str(validation_results))
    logging.info('  Test Results' + generate_results_str(test_results))

    logging.info('\nSaving the model...\n')
    final_model.save('model/trained_model')

    logging.info('Model saved.')
    logging.info('All done.')

    return None
    

if __name__ == '__main__':
    main()