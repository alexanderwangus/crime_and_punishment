# Gaussian kernel SVM
import numpy as np
import csv

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

def get_kernel(train_examples, predict_examples, radius):
    """
    Arguments:
        train_examples: np array with dimension (M_train, N) containing training examples 
        predict_examples: np array with dimension (M_predict, N) containing examples for which predictions will be made
    Returns:
        np array with dimension(M_train, M_predict) containing the kernels for every combination of training, predict examples
    Notes:
        N includes the intercept
    """

    M_train, N = train_examples.shape
    M_predict, N = predict_examples.shape
    kernel = np.zeros((M_train, M_predict))
    for i in range(M_train): 
        for j in range(M_predict):
            kernel[i][j] = np.linalg.norm(train_examples[i] - predict_examples[j]) ** 2
            kernel[i][j] /= (2 * (radius ** 2))
            kernel[i][j] = np.exp(-1 * kernel[i][j])
    return kernel

def train(train_examples, train_labels, radius):
    print('training')
    M_train, N = train_examples.shape
    train_labels = np.subtract(np.multiply(2, train_labels), 1)
    K = get_kernel(train_examples, train_examples, radius)
    LR = 1. / (64 * M_train)
    alpha = np.zeros(M_train)
    alpha_avg = np.zeros(M_train)
    num_iter = 10 * M_train
    ii = 0
    while ii < num_iter:
        i = int(np.random.rand() * M_train)
        margin = train_labels[i] * np.dot(K[i, :], alpha)
        grad = M_train * LR * K[:, i] * alpha[i]
        if margin < 1:
            grad -= train_labels[i] * K[:, i]
        alpha -= grad / np.sqrt(ii + 1)
        alpha_avg += alpha
        ii += 1
    return alpha_avg

def predict(alpha_avg, train_examples, train_labels, test_examples, test_labels, radius):
    label_freq = np.zeros((2))
    accuracy = np.zeros((2, 2))
    M_test = len(test_examples)
    predictions = []
    K = get_kernel(train_examples, test_examples, radius) 
    W = np.multiply(alpha_avg, train_labels)
    for i in range(M_test):
        predict = (np.inner(W, K[:, i]) >= 1)
        predictions.append(predict)
        accuracy[int(test_labels[i])][int(predict)] += 1.0
        label_freq[int(test_labels[i])] += 1.0
    predictions = np.asarray(predictions)
    accuracy[0, :] /= label_freq[0]
    accuracy[1, :] /= label_freq[1]
    return predictions, accuracy

def run_svm(train_file, test_file, radius, N):
    train_examples, train_labels = process_file(train_file, N) 
    test_examples, test_labels = process_file(test_file, N)
    alpha_avg = train(train_examples, train_labels, radius)
    predictions, prediction_accuracy = predict(alpha_avg, train_examples, train_labels, test_examples, test_labels, radius)
    print('Positive Precision = ', (prediction_accuracy[1][1] / (prediction_accuracy[0][1] + prediction_accuracy[1][1])))
    print('Positive Recall = ', (prediction_accuracy[1][1] / (prediction_accuracy[1][0] + prediction_accuracy[1][1])))
    print('Inverse Precision = ', (prediction_accuracy[0][0] / (prediction_accuracy[0][0] + prediction_accuracy[1][0])))
    print('Inverse Recall = ', (prediction_accuracy[0][0] / (prediction_accuracy[0][0] + prediction_accuracy[0][1])))
    print()
    return accuracy

run_svm('./Data/train_pca.csv', './Data/test_pca.csv', 0.15, 2)
run_svm('./Data/train_pca.csv', './Data/test_pca.csv', 0.10, 2)
run_svm('./Data/train_pca.csv', './Data/test_pca.csv', 0.05, 2)
run_svm('./Data/train_pca.csv', './Data/test_pca.csv', 0.01, 2)
run_svm('./Data/train_pca.csv', './Data/test_pca.csv', 0.025, 2)
run_svm('./Data/train_pca.csv', './Data/test_pca.csv', 0.005, 2)
