import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import math
import utils
from matplotlib import pyplot as plt

RACE_BLIND = True
PCA = True
NUM_PCA_FEATURES = 12

def train(X, y, max_depth=2, criterion='gini', max_features=None):
	clf = RandomForestClassifier(n_estimators=10, max_depth=max_depth, criterion=criterion, random_state=0, max_features=max_features)
	clf.fit(X, y)
	return clf

def print_importances(clf, feature_names):
	importances = clf.feature_importances_
	named_importances = []
	for i in range(len(importances)):
		named_importances.append((feature_names[i], importances[i]))

	named_importances = sorted(named_importances, key=lambda pair: abs(pair[1]), reverse=True)
	print("Random Forest Trained. Feature Importances:", named_importances)

def predict(clf, x):
	return clf.predict(x.reshape(1, -1))

def evaluate(X, y, clf):
	correct = 0
	false = 0
	for i in range(len(X)):
		prediction = predict(clf, X[i])[0]
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

def plot_vs_max_depth(X_train, y_train, X_validate, y_validate):
	train_accuracies = []
	validation_accuracies = []
	max_max_depth = 20
	for i in range(1, max_max_depth+1):
		clf = train(X_train, y_train, max_depth=i)
		train_accuracy = evaluate(X_train, y_train, clf)
		validation_accuracy = evaluate(X_validate, y_validate, clf)
		train_accuracies.append(train_accuracy)
		validation_accuracies.append(validation_accuracy)
	plt.plot(range(1, max_max_depth+1), train_accuracies, color='r', label='Train')
	plt.plot(range(1, max_max_depth+1), validation_accuracies, color='g', label='Validate')
	plt.title("Random Forest Train and Validate Accuracies")
	plt.ylabel("Accuracy")
	plt.xlabel("Max Depth of Trees")
	plt.legend(loc='upper left')
	plt.show()

def plot_vs_max_depth_and_max_features(X_train, y_train, X_validate, y_validate):
	max_max_features = len(X_train[0])
	max_max_depth = 6
	min_max_depth = 4
	print(max_max_features)
	colors = ['lightblue', 'indianred', 'darkseagreen', 'papayawhip', 'khaki']
	for i in range(min_max_depth, max_max_depth + 1):
		train_accuracies = []
		validation_accuracies = []
		for j in range(1, max_max_features+1):
			clf = train(X_train, y_train, max_depth=i, max_features = j)
			train_accuracy = evaluate(X_train, y_train, clf)
			validation_accuracy = evaluate(X_validate, y_validate, clf)
			train_accuracies.append(train_accuracy)
			validation_accuracies.append(validation_accuracy)
		train_str = 'Train, Max Depth: ' + str(i)
		validate_str = 'Validate, Max Depth: ' + str(i)
		plt.plot(range(1, max_max_features+1), train_accuracies, color=colors[i-min_max_depth], label=train_str, linestyle='dashed')
		plt.plot(range(1, max_max_features+1), validation_accuracies, color=colors[i-min_max_depth], label=validate_str)
	plt.title("Random Forest Train and Validate Accuracies")
	plt.ylabel("Accuracy")
	plt.xlabel("Max no. features used in splits")
	plt.legend(loc='upper left')
	plt.show()

def plot_vs_pca_components():
	max_features = 12
	train_accuracies = []
	validation_accuracies = []
	for i in range(1, max_features+1):
		X_train, y_train = utils.get_data(utils.TRAIN_PATH, race_blind = RACE_BLIND, pca = PCA, num_pca_features = i)
		X_validate, y_validate = utils.get_data(utils.VALIDATE_PATH, race_blind = RACE_BLIND, pca = PCA, num_pca_features = i)
		clf = train(X_train, y_train, max_depth=5, max_features=None)
		train_accuracy = evaluate(X_train, y_train, clf)
		validation_accuracy = evaluate(X_validate, y_validate, clf)
		train_accuracies.append(train_accuracy)
		validation_accuracies.append(validation_accuracy)
	plt.plot(range(1, max_features+1), train_accuracies, color='lightblue', label='Train')
	plt.plot(range(1, max_features+1), validation_accuracies, color='indianred', label='Validate')
	plt.title("Random Forest Train and Validate Accuracies")
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
		clf = train(X_train, y_train, max_depth=5, max_features=None)
		results_train = evaluate_full(X_train, y_train, clf)
		results_validate = evaluate_full(X_validate, y_validate, clf)
		train_recall.append(results_train['recall'])
		train_precision.append(results_train['precision'])
		validation_recall.append(results_validate['recall'])
		validation_precision.append(results_validate['precision'])

	plt.plot(range(1, max_features+1), train_recall, color='lightblue', label='Train Recall')
	plt.plot(range(1, max_features+1), train_precision, color='lightblue', label='Train Precision', linestyle='dashed')
	plt.plot(range(1, max_features+1), validation_recall, color='indianred', label='Validate Recall')
	plt.plot(range(1, max_features+1), validation_precision, color='indianred', label='Validate Precision', linestyle='dashed')
	plt.title("Random Forest Train and Validate Recall and Precision")
	plt.ylabel("Accuracy")
	plt.xlabel("No. PCA Components")
	plt.legend(loc='lower right')
	plt.show()

def main():
# 	X_train, y_train = utils.get_data(utils.TRAIN_PATH, race_blind = RACE_BLIND, pca = PCA, num_pca_features = NUM_PCA_FEATURES)
# 	X_validate, y_validate = utils.get_data(utils.VALIDATE_PATH, race_blind = RACE_BLIND, pca = PCA, num_pca_features = NUM_PCA_FEATURES)
# 	X_test, y_test = utils.get_data(utils.TEST_PATH, race_blind = RACE_BLIND, pca = PCA, num_pca_features = NUM_PCA_FEATURES)
# 	feature_names = utils.get_feature_names(utils.TRAIN_PATH, race_blind = RACE_BLIND, pca = PCA, num_pca_features = NUM_PCA_FEATURES)
	# plot_vs_max_depth_and_max_features(X_train, y_train, X_validate, y_validate)
	# plot_vs_max_depth(X_train, y_train, X_validate, y_validate)

	plot_vs_pca_components()
	# plot_full_vs_pca_components()

	# clf = train(X_train, y_train, max_depth=5, max_features=None)
	# print_importances(clf, feature_names)
	# print("Train Recall + Precision: ", evaluate_full(X_train, y_train, clf))
	# print("Train Accuracy: ", evaluate(X_train, y_train, clf))
	# print("Validation Recall + Precision: ", evaluate_full(X_validate, y_validate, clf))
	# print("Validation Accuracy: ", evaluate(X_validate, y_validate, clf))
	# print("Test Accuracy: ", evaluate(X_test, y_test, clf))

if __name__ == '__main__':
	main()
