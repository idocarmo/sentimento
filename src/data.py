import re
import numpy as np
import string
from collections.abc import Iterable
from typing import List

from unidecode import unidecode

import datasets
from datasets import load_dataset

import tensorflow as tf
from tensorflow.keras import layers


VOCAB_SIZE = 10000
MAX_TOKENS = 256
BATCH_SIZE = 32

class DataPipeline:
    """Class for loading and processing imdb reviews data.

    Longer class information...
    Longer class information...

    Attributes:
        vectorizer_layer: tensor representation of corpus documents

    Methods:

    """
    def __init__(self) -> None:
        """Initializes the DataPipeline instance.

        Args:
            None.
        """
        self.vectorizer_layer = layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            standardize=self.compose_corpus_tensor,
            split='whitespace',
            ngrams=None,
            output_mode='int',
            output_sequence_length=MAX_TOKENS,
            pad_to_max_tokens=True,
            encoding='utf-8',
        )

    def fit_transform(self, dataset: datasets.arrow_dataset.Dataset):
        ds = dataset.map(self.decode_text, batch_size=BATCH_SIZE)
        corpus_tensor = self.compose_corpus_tensor(ds['text'])
        self.vectorizer_layer.adapt(corpus_tensor, batch_size=BATCH_SIZE)
        return self.vectorizer_layer(ds['text']), np.array(ds['label'])

    def transform(self, dataset: datasets.arrow_dataset.Dataset):
        ds = dataset.map(self.decode_text, batch_size=BATCH_SIZE)
        corpus = ds['text']
        return self.vectorizer_layer(corpus), np.array(ds['label'])

    @staticmethod
    def decode_text(dataset_row):
        dataset_row['text'] = unidecode(dataset_row['text'])
        return dataset_row

    @staticmethod
    def standardize_text(input_data):
        lowercase = tf.strings.lower(input_data)
        standardized = tf.strings.regex_replace(
            lowercase, 
            '[%s]' % re.escape(string.punctuation),
            ''
        )
        return standardized

    @staticmethod
    def compose_corpus_tensor(corpus):
        corpus_tensor = tf.expand_dims(corpus, -1)
        return corpus_tensor

    @staticmethod
    def load_data() -> tuple:
        '''Load portuguese imdb reviews dataset

        Loads train and test portuguese imdb reviews from huggingface
        maritaca AI's repository.
        Splits train dataset in train and validation datasets.  

        Args:
            None.
        Returns:
            tuple of train, validadtion and test datasets.
        '''
        raw_train_dataset, test_dataset = load_dataset('maritaca-ai/imdb_pt', split=['train', 'test'])
        raw_train_dataset = raw_train_dataset.train_test_split(0.2)

        return raw_train_dataset['train'], raw_train_dataset['test'], test_dataset