import csv
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import utils
from matplotlib import pyplot as plt
from collections import defaultdict

RACE_BLIND = True
PCA = False
NUM_PCA_FEATURES = 12

def train(X, y):
	return LinearRegression().fit(X, y)

def predict(reg, x):
	return reg.predict(x.reshape(1, -1))

def evaluate(X, y, reg):
	correct = 0
	false = 0
	for i in range(len(X)):
		prediction = predict(reg, X[i])[0]
		if abs(prediction - y[i]) < abs(prediction - (1 - y[i])):
			correct += 1
		else:
			false += 1
	return float(correct)/(correct + false)

def evaluate_full(X, y, reg):
	correct_true = 0 # true positive
	incorrect_true = 0 # false positive
	correct_false = 0 # true negative
	incorrect_false = 0 # false negative
	for i in range(len(X)):
		prediction = int(round(predict(reg, X[i])[0]) > 0)
		if y[i] == 1:
			if prediction == 1:
				correct_true += 1
			else:
				incorrect_false += 1
		else:
			if prediction == 1:
				incorrect_true += 1
			else:
				correct_false += 1

	return {'precision': float(correct_true)/(correct_true + incorrect_true), 'recall': float(correct_true)/(correct_true + incorrect_false)}

def evaluate_racial_bias(X, X_race, feature_names_race, y, reg):
	# get indices of race_is_0 (black) and race_is_2 (caucasian)
	indices = [feature_names_race.index("race_is_0"), feature_names_race.index("race_is_1"), feature_names_race.index("race_is_2"), feature_names_race.index("race_is_3"), feature_names_race.index("race_is_4"), feature_names_race.index("race_is_5")]
	accuracies = [defaultdict(int) for _ in range(6)]

	for i in range(len(X)):
		prediction = int(round(predict(reg, X[i])[0]) > 0)
		race = 0
		for j in range(6):
			if X_race[i][indices[j]] == 1:
				race = j
				break

		if y[i] == 1:
			if prediction == 1:
				accuracies[race]["correct_true"] += 1
			else:
				accuracies[race]["incorrect_false"] += 1
		else:
			if prediction == 1:
				accuracies[race]["incorrect_true"] += 1
			else:
				accuracies[race]["correct_false"] += 1

	print(accuracies)

def plot_vs_pca_components():
	max_features = 12
	train_accuracies = []
	validation_accuracies = []
	for i in range(1, max_features+1):
		X_train, y_train = utils.get_data(utils.TRAIN_PATH, race_blind = RACE_BLIND, pca = PCA, num_pca_features = i)
		X_validate, y_validate = utils.get_data(utils.VALIDATE_PATH, race_blind = RACE_BLIND, pca = PCA, num_pca_features = i)
		reg = train(X_train, y_train)
		train_accuracy = evaluate(X_train, y_train, reg)
		validation_accuracy = evaluate(X_validate, y_validate, reg)
		train_accuracies.append(train_accuracy)
		validation_accuracies.append(validation_accuracy)
	plt.plot(range(1, max_features+1), train_accuracies, color='lightblue', label='Train')
	plt.plot(range(1, max_features+1), validation_accuracies, color='indianred', label='Validate')
	plt.title("Linear Regression Train and Validate Accuracies")
	plt.ylabel("Accuracy")
	plt.xlabel("No. PCA Components")
	plt.legend(loc='upper left')
	plt.show()

def plot_full_vs_pca_components():
	max_features = 12
	train_recall = []
	validation_recall = []
	train_precision = []
	validation_precision = []
	for i in range(1, max_features+1):
		X_train, y_train = utils.get_data(utils.TRAIN_PATH, race_blind = RACE_BLIND, pca = PCA, num_pca_features = i)
		X_validate, y_validate = utils.get_data(utils.VALIDATE_PATH, race_blind = RACE_BLIND, pca = PCA, num_pca_features = i)
		reg = train(X_train, y_train)
		results_train = evaluate_full(X_train, y_train, reg)
		results_validate = evaluate_full(X_validate, y_validate, reg)
		train_recall.append(results_train['recall'])
		train_precision.append(results_train['precision'])
		validation_recall.append(results_validate['recall'])
		validation_precision.append(results_validate['precision'])

	plt.plot(range(1, max_features+1), train_recall, color='lightblue', label='Train Recall')
	plt.plot(range(1, max_features+1), train_precision, color='lightblue', label='Train Precision', linestyle='dashed')
	plt.plot(range(1, max_features+1), validation_recall, color='indianred', label='Validate Recall')
	plt.plot(range(1, max_features+1), validation_precision, color='indianred', label='Validate Precision', linestyle='dashed')
	plt.title("Linear Regression Train and Validate Recall and Precision")
	plt.ylabel("Accuracy")
	plt.xlabel("No. PCA Components")
	plt.legend(loc='lower right')
	plt.show()

def main():
	X_train, y_train = utils.get_data(utils.TRAIN_PATH, race_blind = RACE_BLIND, pca = PCA, num_pca_features = NUM_PCA_FEATURES)
	X_validate, y_validate = utils.get_data(utils.VALIDATE_PATH, race_blind = RACE_BLIND, pca = PCA, num_pca_features = NUM_PCA_FEATURES)
	X_test, y_test = utils.get_data(utils.TEST_PATH, race_blind = RACE_BLIND, pca = PCA, num_pca_features = NUM_PCA_FEATURES)
	feature_names = utils.get_feature_names(utils.TRAIN_PATH, race_blind = RACE_BLIND, pca = PCA, num_pca_features = NUM_PCA_FEATURES)

	X_train_race, y_train = utils.get_data(utils.TRAIN_PATH, race_blind = False, pca = PCA, num_pca_features = NUM_PCA_FEATURES)
	X_validate_race, y_validate = utils.get_data(utils.VALIDATE_PATH, race_blind = False, pca = PCA, num_pca_features = NUM_PCA_FEATURES)
	X_test_race, y_test = utils.get_data(utils.TEST_PATH, race_blind = False, pca = PCA, num_pca_features = NUM_PCA_FEATURES)
	feature_names_race = utils.get_feature_names(utils.TRAIN_PATH, race_blind = False, pca = PCA, num_pca_features = NUM_PCA_FEATURES)


	# plot_vs_pca_components()
	# plot_full_vs_pca_components()

	reg = train(X_train, y_train)

	evaluate_racial_bias(X_validate, X_validate_race, feature_names_race, y_validate, reg)

	# print(feature_names)
	# coeff = reg.coef_
	# named_coeffs = []
	# print(len(X_train[0]))
	# for i in range(len(coeff)):
	# 	named_coeffs.append((feature_names[i], coeff[i]))
	#
	# named_coeffs = sorted(named_coeffs, key=lambda pair: abs(pair[1]), reverse=True)
	# print("Coefficients:", named_coeffs)
	#
	# print("Train Precision + Recall: ", evaluate_full(X_train, y_train, reg))
	# print("Train Accuracy: ", evaluate(X_train, y_train, reg))
	# print("Validation Precision + Recall: ", evaluate_full(X_validate, y_validate, reg))
	# print("Validation Accuracy: ", evaluate(X_validate, y_validate, reg))
	# print("Test Accuracy: ", evaluate(X_test, y_test, reg))

if __name__ == '__main__':
	main()
