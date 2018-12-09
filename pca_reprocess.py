import csv
import numpy as np
from sklearn.decomposition import PCA
import math
import utils

RACE_BLIND = False

def get_principle_components(X, n_components=None):
	pca = PCA(n_components=n_components)
	return pca.fit_transform(X)

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

	new_X = get_principle_components(X_train, n_components=None)
	new_data= []
	data_header = []
	for i in range(len(new_X[0])):
		data_header.append('pca feature ' + str(i))
	data_header.append('y')
	new_data.append(data_header)
	for i in range(len(new_X)):
		new_data.append(np.concatenate((new_X[i], [y_train[i]]), axis=0))

	with open(utils.TRAIN_PATH[:-4] + '_pca' + '.csv', mode='w') as output_file:
		csv_writer = csv.writer(output_file, delimiter=',')
		for row in new_data:
			csv_writer.writerow(row)

if __name__ == '__main__':
	main()
