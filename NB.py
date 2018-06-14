# -*- coding:utf-8 -*-
import numpy as np
from collections import defaultdict

def readDataSet(filename, frequency = 0, training_set_ratio = 0.7, shuffle = True):
    '''
    read the dataset file, and shuffle, remove all punctuations
    
    Parameters
    ----------
        filename: str, the filename of the data

        frequency: int, you will select the words that appeared more than the frequency you specified
                    for example, if you set frequency equals 1, the program will return all words that they have 
                    appeared more than once.

        training_set_ratio: float, the ratio of training data account for in all data
        
        shuffle: bool, whether to shuffle the data
    
    Returns
    ----------
        train_text: list, each element contains a tuple of words that in each sentence
        
        train_labels: list, each element is the label of the corresponding sentence
        
        test_text: list
        
        test_labels: list
    '''
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().strip().split('\n')
    
    if shuffle:
        np.random.shuffle(text)
    
    import re
    # split all words by space and add them with their labels to the list "dataset"
    dataset = []
    for index, i in enumerate(text):
        t = i.split('\t')
        label = t[0]
        t1 = re.sub("[\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）]+", "", t[1])
        dataset.append((label, re.split(re.compile('\s+'), t1)))
    
    print("dataset's size is", len(dataset))
    
    # split labels and words
    labels, text = zip(*dataset)
    
    split_line = int(len(text) * training_set_ratio)
    train_text = text[:split_line]
    train_labels = labels[:split_line]
    test_text = text[split_line:]
    test_labels = labels[split_line:]
    return train_text, train_labels, test_text, test_labels

def preprocessing_training_data(text, labels):
    '''
    use bag of words to build features for training data
    
    Parameters
    ----------
        text: lists, each element contains a list of words in a sentence
        
        labels: lists, each element is the label of the sample corresponding to the element in text
    
    Returns
    ----------
        trainX: ndarray, training data, the shape of it is (number of samples, number of features)
        
        trainY: ndarray, labels of training data, the shape of it is (number of samples, )
        
        words_table: dict, key is words, value is the index in bag of words
        
        labels_table: dict, key is the label, value is the index that represents the corresponding label
    '''
    bag_of_words = tuple(set(word for words in text for word in words))
    words_table = {i: index for index, i in enumerate(bag_of_words)}
    trainX = np.empty((len(text), len(bag_of_words)))
    for index, words in enumerate(text):
        for word in words:
            trainX[index, words_table[word]] += 1
    labels_table = {i: index for index, i in enumerate(set(labels))}
    trainY = np.array([labels_table[i] for i in labels])
    return trainX, trainY, words_table, labels_table

def preprocessing_testing_data(text, labels, words_table, labels_table):
    '''
    use bag of words to build features for testing data
    
    Parameters
    ----------
        text: lists, each element contains a list of words in a sentence
        
        labels: lists, each element is the label of the sample corresponding to the element in text
        
        words_table: dict, key is words, value is the index in bag of words
        
        labels_table: dict, key is the label, value is the index that represents the corresponding label
    
    Returns
    ----------
        testX: ndarray, testing data, the shape of it is (number of samples, number of features)
        
        testY: ndarray, labels of testing data, the shape of it is (number of samples, )
    '''
    testX = np.empty((len(text), len(words_table)))
    for index, words in enumerate(text):
        for word in words:
            col = words_table.get(word)
            if col is not None:
                testX[index, words_table[word]] += 1
    testY = []
    for i in labels:
        l = labels_table.get(i)
        if l is not None:
            testY.append(l)
        else:
            labels_table[i] = len(labels_table)
            testY.append(labels_table[i])
    testY = np.array(testY)
    return testX, testY

class GaussianNB:
    '''
    Gaussian naive bayes for continous features
    '''
    def __init__(self):
        self.proability_of_y = {}
        self.mean = {}
        self.var = {}
        
    def fit(self, trainX, trainY):
        '''
        use trainX and trainY to compute the prior probability for each class
        and then compute the mean and variance for each features for each class

        Parameters
        ----------
            trainX: ndarray, training data, the shape of it is (number of samples, number of features)
        
            trainY: ndarray, labels of training data, the shape of it is (number of samples, )
        '''
        labels = set(trainY.tolist())
        for y in labels:
            x = trainX[trainY == y, :]
            self.proability_of_y[y] = x.shape[0] / trainX.shape[0]
            self.mean[y] = x.mean(axis = 0)
            var = x.var(axis = 0)
            var[var == 0] += 1e-9 * var.max()
            self.var[y] = var
        
    def predict(self, testX):
        '''
        predict the labels of testX

        Parameters
        ----------
            testX: ndarray, testing data, the shape of it is (number of samples, number of features)
    
        Returns
        ----------
            ndarray: each element is a str variable, which represent the label of corresponding testing data
        '''
        results = np.empty((testX.shape[0], len(self.proability_of_y)))
        labels = []
        for index, (label, py) in enumerate(self.proability_of_y.items()):
            a = np.exp(- ((testX - self.mean[label]) ** 2) / (2 * self.var[label]) ) / np.sqrt(2 * np.pi * self.var[label])
            a[a == 0] += 1e-9 * a.max()
            results[:, index] = np.sum(np.log(a), axis = 1) * py
            labels.append(label)
        return np.array(labels)[np.argmax(results, axis = 1)]

def accuracy(prediction, testY):
    '''
    compute accuracy for prediction

    Parameters
    ----------
        prediction: ndarray, the prediction generated by the classifier
        
        testY: ndarray, true labels

    Returns
    ----------
        float, accuracy
    '''
    return np.sum((prediction - testY) == 0) / testY.shape[0]

def main():
    datadir = 'SMSSpamCollection'
    train_text, train_labels, test_text, test_labels = readDataSet(datadir)
    trainX, trainY, words_table, labels_table = preprocessing_training_data(train_text, train_labels)
    print('training data shape:', trainX.shape, trainY.shape)
    testX, testY = preprocessing_testing_data(test_text, test_labels, words_table, labels_table)
    print('testing data shape:', testX.shape, testY.shape)
    a = GaussianNB()
    a.fit(trainX, trainY)
    r = a.predict(testX)
    print('accuracy:', accuracy(r, testY))

if __name__ == '__main__':
    main()
