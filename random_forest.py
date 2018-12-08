import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import math
import utils
from matplotlib import pyplot as plt

RACE_BLIND = True

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

def main():
	if RACE_BLIND:
		X_train, y_train = utils.get_data(utils.TRAIN_PATH_RACE_BLIND)
		X_validate, y_validate = utils.get_data(utils.VALIDATE_PATH_RACE_BLIND)
		X_test, y_test = utils.get_data(utils.TEST_PATH_RACE_BLIND)
		feature_names = utils.get_feature_names_race_blind()
	else:
		X_train, y_train = utils.get_data(utils.TRAIN_PATH)
		X_validate, y_validate = utils.get_data(utils.VALIDATE_PATH)
		X_test, y_test = utils.get_data(utils.TEST_PATH)
		feature_names = utils.get_feature_names()

	# plot_vs_max_depth_and_max_features(X_train, y_train, X_validate, y_validate)
	# plot_vs_max_depth(X_train, y_train, X_validate, y_validate)

	clf = train(X_train, y_train, max_depth=5, max_features=None)
	print_importances(clf, feature_names)
	print("Train Accuracy: ", evaluate(X_train, y_train, clf))
	print("Validation Accuracy: ", evaluate(X_validate, y_validate, clf))
	# print("Test Accuracy: ", evaluate(X_test, y_test, clf))

if __name__ == '__main__':
	main()
