# Modified from cs229 pset 2

import numpy as np
import csv
np.random.seed(123)


def train_and_predict_svm(train_matrix, train_labels, test_matrix, radius, test_labels):
    """Train an SVM model and predict the resulting labels on a test set.

    Args: 
        train_matrix: A numpy array containing the word counts for the train set
        train_labels: A numpy array containing the spam or not spam labels for the train set
        test_matrix: A numpy array containing the word counts for the test set
        radius: The RBF kernel radius to use for the SVM

    Return: 
    The predicted labels for each message
	"""
    model = svm_train(train_matrix, train_labels, radius)
    return svm_predict(model, test_matrix, radius, test_labels)


def svm_train(matrix, category, radius):
    state = {}
    M, N = matrix.shape
    Y = 2 * category - 1
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(matrix.T)
    K = np.exp(-(squared.reshape((1, -1)) + squared.reshape((-1, 1)) - 2 * gram) / (2 * (radius ** 2)))

    alpha = np.zeros(M)
    alpha_avg = np.zeros(M)
    L = 1. / (64 * M)
    outer_loops = 10

    alpha_avg = 0
    ii = 0
    while ii < outer_loops * M:
        i = int(np.random.rand() * M)
        margin = Y[i] * np.dot(K[i, :], alpha)
        grad = M * L * K[:, i] * alpha[i]
        if margin < 1:
            grad -= Y[i] * K[:, i]
        alpha -= grad / np.sqrt(ii + 1)
        alpha_avg += alpha
        ii += 1

    alpha_avg /= (ii + 1) * M

    state['alpha'] = alpha
    state['alpha_avg'] = alpha_avg
    state['Xtrain'] = matrix
    state['Sqtrain'] = squared
    return state


def svm_predict(state, matrix, radius, test_labels):
    M, N = matrix.shape

    Xtrain = state['Xtrain']
    Sqtrain = state['Sqtrain']
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(Xtrain.T)
    K = np.exp(-(squared.reshape((-1, 1)) + Sqtrain.reshape((1, -1)) - 2 * gram) / (2 * (radius ** 2)))
    alpha_avg = state['alpha_avg']
    preds = K.dot(alpha_avg)
    output = (1 + np.sign(preds)) // 2

    prediction_accuracy = accuracy(output, test_labels)
    print('radius: ', radius)
    print('true negative: ', prediction_accuracy[0][0])
    print('false positive: ', prediction_accuracy[0][1])
    print('true positive: ', prediction_accuracy[1][1])
    print('false negative: ', prediction_accuracy[1][0])

    print('Positive Precision = ', (prediction_accuracy[1][1] / (prediction_accuracy[0][1] + prediction_accuracy[1][1])))
    print('Positive Recall = ', (prediction_accuracy[1][1] / (prediction_accuracy[1][0] + prediction_accuracy[1][1])))

    print('Negative Precision = ', (prediction_accuracy[0][0] / (prediction_accuracy[0][0] + prediction_accuracy[1][0])))
    print('Negative Recall = ', (prediction_accuracy[0][0] / (prediction_accuracy[0][0] + prediction_accuracy[0][1])))

    print()
    return output, prediction_accuracy

def accuracy(output, test_labels):
    accuracy = np.zeros((2, 2))
    label_freq = np.zeros((2))
    for (predict_class, true_class) in zip(output, test_labels):
        accuracy[int(true_class)][int(predict_class)] += 1.0
        label_freq[int(true_class)] += 1.0
    accuracy[0, :] /= label_freq[0]
    accuracy[1, :] /= label_freq[1]
    return accuracy

def process_file(file_name, N):
    """
    Arguments:
        file_name: name of file to be read
        N: number of features (not including intercept)
    Returns:
        np array with dimension (M, N) containing examples
        np array with dimension (N) containing corresponding labels
    """

    examples = []
    labels = []
    with open(file_name) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            examples.append([1] + row[:N])
            labels.append(row[len(row) - 1])
    examples = np.asarray(examples)
    examples = examples.astype(float)
    labels = np.asarray(labels)
    labels = labels.astype(float)
    return examples, labels

def process_file_features(file_name, feature_indices):
    examples = []
    labels = []
    with open(file_name) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            row_features = []
            for feature_index in feature_indices:
                row_features.append(row[feature_indices])
                row_features = [1] + row_features
            examples.append(row_features)
            labels.append(row[len(row) - 1])
    examples = np.asarray(examples)
    examples = examples.astype(float)
    labels = np.asarray(labels)
    labels = labels.astype(float)
    return examples, labels

train_examples, train_labels = process_file('./Data/train_pca.csv', 19)
test_examples, test_labels = process_file('./Data/test_pca.csv', 19)
train_and_predict_svm(train_examples, train_labels, test_examples, 0.15, test_labels) 
train_and_predict_svm(train_examples, train_labels, test_examples, 0.10, test_labels)
train_and_predict_svm(train_examples, train_labels, test_examples, 0.05, test_labels)
train_and_predict_svm(train_examples, train_labels, test_examples, 0.01, test_labels)
train_and_predict_svm(train_examples, train_labels, test_examples, 0.025, test_labels)
train_and_predict_svm(train_examples, train_labels, test_examples, 0.005, test_labels)
# train_examples, train_labels = process_file('./Data/train/pca.csv', )
# test_examples, test_labels = process_file('./Data/test_pca.csv', 2)
