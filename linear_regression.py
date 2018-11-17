import csv
import numpy as np
from sklearn.linear_model import LinearRegression
import math

TRAIN_PATH = 'data/train.csv'
VALIDATE_PATH = 'data/validate.csv'
TEST_PATH = 'data/test.csv'

FEATURES_TO_IGNORE = ['id', 'first', 'last', 'c_charge_desc', 'r_charge_desc', 'vr_charge_degree', 'vr_charge_desc', 'event', 'age_cat', 'is_recid', 'v_decile_score', 'decile_score', 'vr_offense_date', 'is_violent_recid', 'r_offense_day_from_endjail', 'start', 'end', 'r_charge_degree']

def train(X, y):
	return LinearRegression().fit(X, y)

def predict(reg, x):
	return reg.predict(x)

def evaluate(X, y, reg):
	correct = 0
	false = 0
	for i in range(len(X)):
		prediction = predict(reg, X[i].reshape(1, -1))[0]
		# print(prediction, y[i])
		if abs(prediction - y[i]) < abs(prediction - (1 - y[i])):
			correct += 1
		else:
			false += 1
	return float(false)/(correct + false)

def trim_data(data_set):
	features = data_set[...,len(FEATURES_TO_IGNORE):-1]
	for i in range(len(features)):
		for j in range(len(features[i])):
			if not features[i][j].isdigit():
				features[i][j] = 0
	labels = data_set[...,-1]
	return np.array(features, dtype=int), np.array(labels, dtype=int)

def main():
	train_file = open(TRAIN_PATH, 'r')
	train_set = csv.reader(train_file, delimiter=',')
	validate_file = open(VALIDATE_PATH, 'r')
	validate_set = csv.reader(validate_file, delimiter=',')
	test_file = open(TEST_PATH, 'r')
	test_set = csv.reader(test_file, delimiter=',')

	train_array = np.array(list(train_set))
	train_array = train_array[1:]
	validate_array = np.array(list(validate_set))
	validate_array = validate_array[1:]
	test_array = np.array(list(test_set))
	test_array = test_array[1:]

	X_train, y_train = trim_data(train_array)
	print(X_train)
	X_validate, y_validate = trim_data(validate_array)
	X_test, y_test = trim_data(test_array)

	reg = train(X_train, y_train)
	print("Coefficients:", reg.coef_)

	validation_error = evaluate(X_validate, y_validate, reg)
	test_error = evaluate(X_test, y_test, reg)

	print("Validation error:", validation_error)
	print("Test error:", test_error)

if __name__ == '__main__':
	main()
