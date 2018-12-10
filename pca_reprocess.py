import csv
import numpy as np
from sklearn.decomposition import PCA
import math
import utils

RACE_BLIND = True
USE_PCA = False
NUM_PCA_FEATURES = 12

def get_principle_components(X, n_components=None):
	pca = PCA(n_components=n_components)
	return pca.fit_transform(X)

def analyse_pca(X, feature_names, n_components=None):
	pca = PCA(n_components=n_components)
	pca.fit(X)
	for i in range(5):
		component = pca.components_[i]
		named_components = []
		for j in range(len(component)):
			named_components.append((feature_names[j], component[j]))
			named_components = sorted(named_components, key=lambda pair: abs(pair[1]), reverse=True)
		print(named_components)

def main():
	X_train, y_train = utils.get_data(utils.TRAIN_PATH, race_blind = RACE_BLIND, pca = USE_PCA, num_pca_features = NUM_PCA_FEATURES)
	X_validate, y_validate = utils.get_data(utils.VALIDATE_PATH, race_blind = RACE_BLIND, pca = USE_PCA, num_pca_features = NUM_PCA_FEATURES)
	X_test, y_test = utils.get_data(utils.TEST_PATH, race_blind = RACE_BLIND, pca = USE_PCA, num_pca_features = NUM_PCA_FEATURES)
	feature_names = utils.get_feature_names(utils.TRAIN_PATH, race_blind = RACE_BLIND, pca = USE_PCA, num_pca_features = NUM_PCA_FEATURES)


	# new_X = get_principle_components(X_train, n_components=None)
	analyse_pca(X_train, feature_names, n_components=None)
	# new_data= []
	# data_header = []
	# for i in range(len(new_X[0])):
	# 	data_header.append('pca feature ' + str(i))
	# data_header.append('y')
	# new_data.append(data_header)
	# for i in range(len(new_X)):
	# 	new_data.append(np.concatenate((new_X[i], [y_train[i]]), axis=0))

	# with open(utils.TRAIN_PATH[:-4] + '_pca' + '.csv', mode='w') as output_file:
	# 	csv_writer = csv.writer(output_file, delimiter=',')
	# 	for row in new_data:
	# 		csv_writer.writerow(row)

if __name__ == '__main__':
	main()
