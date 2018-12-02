import csv

TRAIN_PATH = 'data/train.csv'
VALIDATE_PATH = 'data/validate.csv'
TEST_PATH = 'data/test.csv'

def train(train_set):
	recid_count = sum([int(row[-1]) for row in train_set])
	if recid_count > sum(1 for row in train_set) - recid_count:
		return 1
	else:
		return 0

def predict(theta):
	return theta

def evaluate(data_set, theta):
	correct = 0
	false = 0
	for row in data_set:
		if predict(theta) == int(row[-1]):
			correct += 1
		else:
			false += 1
	return float(false)/(correct + false)

def main():
	train_file = open(TRAIN_PATH, 'r')
	train_set = list(csv.reader(train_file, delimiter=','))[1:]
	validate_file = open(VALIDATE_PATH, 'r')
	validate_set = list(csv.reader(validate_file, delimiter=','))[1:]
	test_file = open(TEST_PATH, 'r')
	test_set = list(csv.reader(test_file, delimiter=','))[1:]

	theta = train(train_set)
	print("Theta:", theta)

	validation_error = evaluate(validate_set, theta)
	test_error = evaluate(test_set, theta)

	print("Validation error:", validation_error)
	print("Test error:", test_error)

if __name__ == '__main__':
	main()
