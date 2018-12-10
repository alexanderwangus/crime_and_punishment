import numpy as np
import scipy.stats
import utils

def process_file(file_name, n):
    """
    Arguments:
        file_name: name of file to be read
        n: number of features per vector 
    Returns:
        examples: np array with dimension (m, n) containing examples
        labels: np array with dimension (n) containing corresponding labels
        m: number of examples in file
        n: number of features per vector
    """

    examples = []
    labels = []
    with open(file_name) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            print(row)
            examples.append(row[:n])
            labels.append(row[len(row) - 1])
    examples = np.asarray(examples)
    examples = examples.astype(float)
    labels = np.asarray(labels)
    labels = labels.astype(float)
    return examples, labels, examples.shape[0], n

def train(examples, labels, m, n, laplace):
    """
    Trains model using the naive bayes algorithm

    Args:
        examples: np matrix with dimension (m, n) containing examples
        labels: np matrix with dimension (m) containing corresponding labels 
        m: number of examples in file
        n: number of features per vector
        laplace: True if user requests laplace smoothing, False otherwise

    Returns:
        x_cond_y_mean: np array with dimension(2, n) containing mean of each feature for each class
        x_cond_y_std: np array with dimension(2, n) containing std dev of each feature for each class
        priors: label probabilties
    """

    x_cond_y = [[], []] 
    for i in range(n):
        x_cond_y[0].append([])
        x_cond_y[1].append([])
    priors = np.zeros((2))
    for i in range(m):
        priors[int(labels[i])] += 1.0
        for j in range(n):
            x_cond_y[int(labels[i])][j].append(float(examples[i][j]))
    x_cond_y_mean = np.zeros((2, n))
    x_cond_y_std = np.zeros((2, n)) 
    if (laplace):
        for i in range(n):
            x_cond_y[0][i].append(1.0)
            x_cond_y[1][i].append(1.0) 
        priors[0] += 1.0
        priors[1] += 1.0
    for i in range(2):
        for j in range(n):
            row = np.asarray(x_cond_y[i][j])
            x_cond_y_mean[i][j] = np.mean(row)
            x_cond_y_std[i][j] = np.std(row)
    priors /= np.sum(priors)
    return x_cond_y_mean, x_cond_y_std, priors

def predict(x, x_cond_y_mean, x_cond_y_std, priors):
    """
    Args:
        x: single training example
        x_cond_y_mean: np array with dimension(2, n) containing mean of each feature for each class
        x_cond_y_std: np array with dimension(2, n) containing std dev of each feature for each class
        priors: label probabilties
     Returns:
        predicted class to which x belongs
    """

    n = len(x)
    p = np.ones((2))
    for i in range(2):
        for j in range(n):
            p[i] *= scipy.stats.norm(x_cond_y_mean[i][j], x_cond_y_std[i][j]).pdf(x[j])
    p = np.multiply(p, priors)
    return p[1] > p[0]

def run_naive_bayes(train_file, test_file, laplace, n):
    """
    Args:
        train: training data file
        test: test data file
        laplace: True if user requests laplace smoothing, False otherwise
        n: number of features
    """

    train_examples, train_labels = utils.get_data(utils.TRAIN_PATH)
    test_examples, test_labels = utils.get_data(utils.TEST_PATH)
    m_train, n = train_examples.shape
    x_cond_y_mean, x_cond_y_std, priors = train(train_examples, train_labels, m_train, n, laplace)
    prediction_actual = np.zeros((2, 2))
    for (example, label) in zip(test_examples, test_labels):
        prediction = predict(example, x_cond_y_mean, x_cond_y_std, priors)
        prediction_actual[int(label)][int(prediction)] += 1.0
    print_accuracy(prediction_actual)
    print()

def print_accuracy(prediction_actual):
    """
    Prints precision and recall information
    Args:
        prediction_actual: np array with dimension(2, 2) where prediction_actual[i][j] is the number of examples with true label i that were predicted to have label j
    """
    print('Positive Precision = ', prediction_actual[1][1] / np.sum(prediction_actual[:, 1]))
    print('Positive Recall = ', prediction_actual[1][1] / np.sum(prediction_actual[1, :]))
    print('Inverse Precision = ', prediction_actual[0][0] / np.sum(prediction_actual[:, 0]))
    print('Inverse Recall = ', prediction_actual[0][0] / np.sum(prediction_actual[0, :]))

print()
print('=== Naive Bayes ===')
run_naive_bayes('./Data/train.csv', './Data/test.csv', True, 19)
print()
