import csv
import random

INPUT_PATH = 'data/Processed_compas-scores-two-years-reprocess.csv'
TRAIN_PATH = 'data/train.csv'
VALIDATE_PATH = 'data/validate.csv'
TEST_PATH = 'data/test.csv'


def main():
	with open(INPUT_PATH) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		column_names = []
		rows = []
		for row in csv_reader:
			if line_count == 0:
				column_names = row
			else:
				rows.append(row)
			line_count += 1

		print(column_names)

		random.shuffle(rows)
		train = rows[:int(len(rows)*0.8)]
		validate = rows[int(len(rows)*0.8):int(len(rows)*0.9)]
		test = rows[int(len(rows)*0.9):]

		with open(TRAIN_PATH, mode='w') as file:
			train_writer = csv.writer(file, delimiter=',')
			train_writer.writerow(column_names)
			for row in train:
				train_writer.writerow(row)

		with open(VALIDATE_PATH, mode='w') as file:
			validate_writer = csv.writer(file, delimiter=',')
			validate_writer.writerow(column_names)
			for row in validate:
				validate_writer.writerow(row)

		with open(TEST_PATH, mode='w') as file:
			test_writer = csv.writer(file, delimiter=',')
			test_writer.writerow(column_names)
			for row in test:
				test_writer.writerow(row)


if __name__ == '__main__':
	main()
