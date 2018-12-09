import csv
import math
import numpy as np

TRAIN_PATH = 'data/train.csv'
VALIDATE_PATH = 'data/validate.csv'
TEST_PATH = 'data/test.csv'

TRAIN_PATH_RACE_BLIND = 'data/train_race_blind.csv'
VALIDATE_PATH_RACE_BLIND = 'data/validate_race_blind.csv'
TEST_PATH_RACE_BLIND = 'data/test_race_blind.csv'

FEATURES_TO_IGNORE = ['id', 'first', 'last', 'c_charge_desc', 'r_charge_desc', 'vr_charge_degree', 'vr_charge_desc', 'event', 'age_cat', 'is_recid', 'v_decile_score', 'decile_score', 'vr_offense_date', 'is_violent_recid', 'r_offense_day_from_endjail', 'start', 'end', 'r_charge_degree']

def trim_data(data_set):
	features = data_set[...,len(FEATURES_TO_IGNORE):-1]
	for i in range(len(features)):
		for j in range(len(features[i])):
			if not features[i][j].isdigit():
				features[i][j] = 0
	labels = data_set[...,-1]
	return np.array(features, dtype=int), np.array(labels, dtype=int)

def remove_race_feature(path):
	file = open(path, 'r')
	set = csv.reader(file, delimiter=',')
	array = list(set)
	new_array = []
	for row in array:
		new_array.append(row[:-8] + [row[-1]])
	return new_array

def get_data(path, race_blind = False, pca = False):
	if race_blind:
		path = path[:-4] + '_race_blind.csv'

	if pca:
		path = path[:-4] + '_pca.csv'

	file = open(path, 'r')
	set = csv.reader(file, delimiter=',')
	array = np.array(list(set))[1:] # removes column names
	return trim_data(array)

def get_feature_names(path, race_blind = False, pca = False):
	if race_blind:
		path = path[:-4] + '_race_blind.csv'

	if pca:
		path = path[:-4] + '_pca.csv'

	file = open(TRAIN_PATH, 'r')
	set = csv.reader(file, delimiter=',')
	return list(set)[0][len(FEATURES_TO_IGNORE):-1]

def main():
	train_race_blind = remove_race_feature(TEST_PATH)
	with open('data/test_race_blind.csv', mode='w') as output_file:
		csv_writer = csv.writer(output_file, delimiter=',')
		for row in train_race_blind:
			csv_writer.writerow(row)

if __name__ == '__main__':
	main()
