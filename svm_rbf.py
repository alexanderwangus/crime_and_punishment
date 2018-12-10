# SVM with RBF kernel
# Modified from Stanford cs229 pset 2

import numpy as np
import csv
import utils
np.random.seed(123)


def train_and_predict_svm(train_matrix, train_labels, test_matrix, test_labels, radius):
    """

    Args: 
        train_matrix: A numpy array containing training examples 
        train_labels: A numpy array containing corresponding labels 
        test_matrix: A numpy array containing testing examples 
        test_labels: A numpy array containing corresponding labels
        radius: The RBF kernel radius to use for the SVM

    Return: 
    The predicted labels for each message
	"""
    print('radius = ', radius)
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
    print('Positive Precision = ', (prediction_accuracy[1][1] / (prediction_accuracy[0][1] + prediction_accuracy[1][1])))
    print('Positive Recall = ', (prediction_accuracy[1][1] / (prediction_accuracy[1][0] + prediction_accuracy[1][1])))
    print('Inverse Precision = ', (prediction_accuracy[0][0] / (prediction_accuracy[0][0] + prediction_accuracy[1][0])))
    print('Inverse Recall = ', (prediction_accuracy[0][0] / (prediction_accuracy[0][0] + prediction_accuracy[0][1])))

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

train_examples_rb, train_labels_rb = utils.get_data(utils.TRAIN_PATH, True) 
test_examples_rb, test_labels_rb = utils.get_data(utils.TEST_PATH, True)
train_examples_nrb, train_labels_nrb = utils.get_data(utils.TRAIN_PATH, False) 
test_examples_nrb, test_labels_nrb = utils.get_data(utils.TEST_PATH, False)
print("RACE BLIND")
train_and_predict_svm(train_examples_rb, train_labels_rb, test_examples_rb, test_labels_rb, 0.15) 
print("NOT RACE BLIND")
train_and_predict_svm(train_examples_nrb, train_labels_nrb, test_examples_nrb, test_labels_nrb, 0.15) 
print("RACE BLIND")
train_and_predict_svm(train_examples_rb, train_labels_rb, test_examples_rb, test_labels_rb, 0.10) 
print("NOT RACE BLIND")
train_and_predict_svm(train_examples_nrb, train_labels_nrb, test_examples_nrb, test_labels_nrb, 0.10) 
print("RACE BLIND")
train_and_predict_svm(train_examples_rb, train_labels_rb, test_examples_rb, test_labels_rb, 0.05)
print("NOT RACE BLIND")
train_and_predict_svm(train_examples_nrb, train_labels_nrb, test_examples_nrb, test_labels_nrb, 0.05)
print("RACE BLIND")
train_and_predict_svm(train_examples_rb, train_labels_rb, test_examples_rb, test_labels_rb, 0.01)
print("NOT RACE BLIND")
train_and_predict_svm(train_examples_nrb, train_labels_nrb, test_examples_nrb, test_labels_nrb, 0.01) 
print("RACE BLIND")
train_and_predict_svm(train_examples_rb, train_labels_rb, test_examples_rb, test_labels_rb, 0.025)
print("NOT RACE BLIND")
train_and_predict_svm(train_examples_nrb, train_labels_nrb, test_examples_nrb, test_labels_nrb, 0.025)
print("RACE BLIND")
train_and_predict_svm(train_examples_rb, train_labels_rb, test_examples_rb, test_labels_rb, 0.005)
print("NOT RACE BLIND")
train_and_predict_svm(train_examples_nrb, train_labels_nrb, test_examples_nrb, test_labels_nrb, 0.005) 
