import os
import sys
import random

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from tabulate import tabulate
import matplotlib.pyplot as plt
from progress.bar import Bar

from vectorize import ngram_vectorize, sequence_vectorize
from model import mlp_model, train_ngram_model
from sequence_model import train_sequence_model

def load_csv(file):
    data = []
    with open(file) as f:
        for row in f.readlines():
            data.append(row.split(';'))
    return data

def load_dataset(data_path, seed=123):
    """Loads the Imdb movie reviews sentiment analysis dataset.
    # Arguments
        data_path: string, path to the data directory.
        seed: int, seed for randomizer.
    # Returns
        A tuple of training and validation data.
        Number of training samples: 25000
        Number of test samples: 25000
        Number of categories: 2 (0 - negative, 1 - positive)
    # References
        Mass et al., http://www.aclweb.org/anthology/P11-1015
        Download and uncompress archive from:
        http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    """
    imdb_data_path = os.path.join(data_path, 'aclImdb')

    bar = Bar('Loading the dataset...', max = 50000)
            
    # Load the training data
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(imdb_data_path, 'train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                bar.next()
                with open(os.path.join(train_path, fname)) as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'neg' else 1)             

    # Load the validation data.
    test_texts = []
    test_labels = []
    for category in ['pos', 'neg']:
        test_path = os.path.join(imdb_data_path, 'test', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                bar.next()
                with open(os.path.join(test_path, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)       
    
    # Shuffle the training data and labels.
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)
    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))

def get_num_words_per_sample(sample_texts):
    """Gets the median number of words per sample given corpus.
    # Arguments
        sample_texts: list, sample texts.
    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)

def plot_frequency_distribution_of_ngrams(sample_texts,
                                          ngram_range=(1, 2),
                                          num_ngrams=50):
    """Plots the frequency distribution of n-grams.
    # Arguments
        samples_texts: list, sample texts.
        ngram_range: tuple (min, mplt), The range of n-gram values to consider.
            Min and mplt are the lower and upper bound values for the range.
        num_ngrams: int, number of n-grams to plot.
            Top `num_ngrams` frequent n-grams will be plotted.
    """
    # Create args required for vectorizing.
    kwargs = {
            'ngram_range': (1, 1),
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',  # Split text into word tokens.
    }
    vectorizer = CountVectorizer(**kwargs)

    # This creates a vocabulary (dict, where keys are n-grams and values are
    # idxices). This also converts every text to an array the length of
    # vocabulary, where every element idxicates the count of the n-gram
    # corresponding at that idxex in vocabulary.
    vectorized_texts = vectorizer.fit_transform(sample_texts)

    # This is the list of all n-grams in the index order from the vocabulary.
    all_ngrams = list(vectorizer.get_feature_names())
    num_ngrams = min(num_ngrams, len(all_ngrams))
    # ngrams = all_ngrams[:num_ngrams]

    # Add up the counts per n-gram ie. column-wise
    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
        zip(all_counts, all_ngrams), reverse=True)])
    ngrams = list(all_ngrams)[:num_ngrams]
    counts = list(all_counts)[:num_ngrams]

    idx = np.arange(num_ngrams)
    plt.bar(idx, counts, width=0.8, color='b')
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title('Frequency distribution of n-grams')
    plt.xticks(idx, ngrams, rotation=45)
    plt.show()

def plot_sample_length_distribution(sample_texts):
    """Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()

def plot_accuracy(epochs, val_acc):
    plt.plot(epochs, val_acc, 'bo', label='Validation acc')
    plt.title('Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()

data = load_dataset(os.getcwd(),25000) #load_csv('IMDBdataset.csv')
print()
"""model = load_model('./rotten_tomatoes_sepcnn_model.h5')
examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]

print(model.predict(examples))"""

if len(sys.argv) - 1 == 1:
    if sys.argv[1] == 'stat':
        words_per_sample = get_num_words_per_sample(data[0][0] + data[1][0])

        table = [['Metric name','Metric value'],
            ['Number of samples',len(data[0][0])+len(data[1][0])],
            ['Number of classes',2],
            ['Number of samples per class',(len(data[0][0])+len(data[1][0]))/2],
            ['Number of words per sample',words_per_sample]]

        print(tabulate(table))

        plot_frequency_distribution_of_ngrams(data[0][0])
        plot_sample_length_distribution(data[0][0])
    if sys.argv[1] == 'sequential':
        print('Training...')
        train = train_sequence_model(
            data=data,
            epochs=2,
            batch_size=512
        )
    if sys.argv[1] == 'ngram':
        vector = ngram_vectorize(data[0][0],data[0][1],data[1][0])
        mlp_model(layers=2,
                units=32,
                dropout_rate=0.2,
                input_shape=vector,
                num_classes=2
        )

#plot_accuracy(epochs=2,val_acc=train[0])