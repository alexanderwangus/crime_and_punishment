import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import math
import utils

def main():
	train_file = open(utils.TRAIN_PATH, 'r')
	train_set = csv.reader(train_file, delimiter=',')
	validate_file = open(utils.VALIDATE_PATH, 'r')
	validate_set = csv.reader(validate_file, delimiter=',')
	test_file = open(utils.TEST_PATH, 'r')
	test_set = csv.reader(test_file, delimiter=',')

	train_array = np.array(list(train_set))
	train_array = train_array[1:]
	validate_array = np.array(list(validate_set))
	validate_array = validate_array[1:]
	test_array = np.array(list(test_set))
	test_array = test_array[1:]

	X_train, y_train = utils.trim_data(train_array)
	X_validate, y_validate = utils.trim_data(validate_array)
	X_test, y_test = utils.trim_data(test_array)

	print(X_train)

if __name__ == '__main__':
	main()
