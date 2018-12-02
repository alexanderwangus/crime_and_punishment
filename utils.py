import csv
import math
import numpy as np

TRAIN_PATH = 'data/train.csv'
VALIDATE_PATH = 'data/validate.csv'
TEST_PATH = 'data/test.csv'

FEATURES_TO_IGNORE = ['id', 'first', 'last', 'c_charge_desc', 'r_charge_desc', 'vr_charge_degree', 'vr_charge_desc', 'event', 'age_cat', 'is_recid', 'v_decile_score', 'decile_score', 'vr_offense_date', 'is_violent_recid', 'r_offense_day_from_endjail', 'start', 'end', 'r_charge_degree']

def trim_data(data_set):
	features = data_set[...,len(FEATURES_TO_IGNORE):-1]
	for i in range(len(features)):
		for j in range(len(features[i])):
			if not features[i][j].isdigit():
				features[i][j] = 0
	labels = data_set[...,-1]
	return np.array(features, dtype=int), np.array(labels, dtype=int)

def get_data(path):
	file = open(path, 'r')
	set = csv.reader(file, delimiter=',')
	array = np.array(list(set))[1:] # removes column names
	return trim_data(array)
