import csv
import random

INPUT_PATH = 'data/Processed_compas-scores-two-years.csv'
OUTPUT_PATH = 'data/Processed_compas-scores-two-years-reprocess.csv'
FEATURES_TO_IGNORE = ['id', 'first', 'last', 'c_charge_desc', 'r_charge_desc', 'vr_charge_degree', 'vr_charge_desc', 'event', 'age_cat', 'is_recid', 'v_decile_score', 'decile_score', 'vr_offense_date', 'is_violent_recid', 'r_offense_day_from_endjail', 'start', 'end', 'r_charge_degree'] # is_recid, is_violent_recid, decile_score and v_delice_score seems to be an unfair features for us to use. r_offense_day_from_endjail we don't use bc only about half rows have this value. Others I remove bc I have idea what they are
# FEATURES_TO_EXPAND = {'race': 6, 'r_charge_degree': 6}
FEATURES_TO_EXPAND = {'race': 6}

def main():
	csv_file = open(INPUT_PATH)
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	rows = []
	column_names = next(csv_reader)

	# Move text features to first columns
	ignore_indices = [column_names.index(column_name) for column_name in FEATURES_TO_IGNORE]
	ignore_indices.sort(reverse=True)

	to_ignore = [column_names[i] for i in ignore_indices]
	to_keep = column_names[:]
	for i in ignore_indices:
		del to_keep[i]
	rows.append(to_ignore + to_keep)

	for row in csv_reader:
		to_ignore = [row[i] for i in ignore_indices]
		to_keep = row
		for i in ignore_indices:
			del to_keep[i]
		rows.append(to_ignore + to_keep)


	# One-hot encoding of categorical features
	csv_file.seek(1)
	column_names = rows[0]
	encoded_rows = rows
	for column_name, num_categories in FEATURES_TO_EXPAND.items():
		index = column_names.index(column_name)
		for i in range(1, len(encoded_rows)):
			to_append = [0 for _ in range(num_categories + 1)]
			if not encoded_rows[i][index].isdigit():
				to_append[num_categories] = 1
			else:
				to_append[int(encoded_rows[i][index])] = 1
			encoded_rows[i] += to_append

	for column_name, num_categories in FEATURES_TO_EXPAND.items():
		index = column_names.index(column_name)
		to_append = [column_name + '_is_' + str(i) for i in range(num_categories + 1)]
		encoded_rows[0] += to_append

	indices_to_delete = [column_names.index(column_name) for column_name in list(FEATURES_TO_EXPAND.keys())]
	indices_to_delete.sort(reverse=True)

	# Delete non one-hot encoding features that have now been encoded
	for index in indices_to_delete:
		for i in range(len(encoded_rows)):
			del encoded_rows[i][index]

	# Move label column to end
	label_index = encoded_rows[0].index('two_year_recid')
	csv_file.seek(0)
	for i in range(len(encoded_rows)):
		encoded_rows[i].append(encoded_rows[i][label_index])
		del encoded_rows[i][label_index]


	with open(OUTPUT_PATH, mode='w') as output_file:
		csv_writer = csv.writer(output_file, delimiter=',')
		for row in encoded_rows:
			csv_writer.writerow(row)



if __name__ == '__main__':
	main()
