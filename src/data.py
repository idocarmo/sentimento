import re
import numpy as np
import string
from typing import Iterable, Tuple

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
        """
        self.vectorizer_layer = layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            standardize=self.standardize_text,
            split='whitespace',
            ngrams=None,
            output_mode='int',
            output_sequence_length=MAX_TOKENS,
            pad_to_max_tokens=True,
            encoding='utf-8',
        )

    def fit_transform(self, corpus: Iterable[str]) -> tf.Tensor:
        '''Train vectorizer layer and transform corpus.

        Train vectorizer layer according the input corpus
        and returns the vector representation of its texts 
        as a tensor.

        Args:
            corpus: iterable with texts strings.
        Returns:
            tensor representation of corpus tokens.
        '''
        corpus_tensor = self.compose_corpus_tensor(corpus)
        self.vectorizer_layer.adapt(corpus_tensor, batch_size=BATCH_SIZE)
        return self.vectorizer_layer(corpus_tensor)

    def transform(self, corpus: Iterable[str]) -> tf.Tensor:
        '''Transform data accoding trained layer.

        Transform data to a tensor representation of tokens
        following the pre-trained vectorization layer.

        Args:
            corpus: iterable with texts strings.
        Returns:
            tensor representation of corpus tokens.
        '''
        corpus_tensor = self.compose_corpus_tensor(corpus)
        return self.vectorizer_layer(corpus_tensor)
    
    @staticmethod
    def standardize_text(input_data: Iterable[str]) -> tf.Tensor:
        '''Standartize a string.

        Convert to lower and remove punctuation from text strings
        in a iterable.

        Args:
            input_data: iterable with texts strings.
        Returns:
            tensor representation of the input data with the normalized
            text strings.
        '''
        lowercase = tf.strings.lower(input_data)
        standardized = tf.strings.regex_replace(
            lowercase, 
            '[%s]' % re.escape(string.punctuation),
            ''
        )
        return standardized

    @staticmethod
    def compose_corpus_tensor(corpus: Iterable[str]) -> tf.Tensor:
        '''Compose a tensor from corpus.

        Creates a tensor representation of the corpus.

        Args:
            corpus: iterable with texts that compose the corpus.
        Returns:
            tensor representation of the set of texts.
        '''
        corpus_tensor = tf.expand_dims(corpus, -1)
        return corpus_tensor

    @staticmethod
    def split_data_target(dataset: datasets.arrow_dataset.Dataset)-> Tuple[np.array, np.array]:
        '''Split datataset

        Split input dataset in text item and respective label.

        Args:
            dataset: dataset where items are specified as 'text' and 
            labels as 'label'
        Returns:
            tuple with text and label arrays
        '''
        return np.array(dataset['text']), np.array(dataset['label'])

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