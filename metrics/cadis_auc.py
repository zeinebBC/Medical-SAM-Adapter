#!/usr/bin/env python
# FROM: https://cataracts.grand-challenge.org/Evaluation/
"""Evaluation script for the CATARACTS challenge.

Usage:
$ python evaluate.py --truth=<truth directory> --predictions=<prediction directory>

Where :
* the <truth directory> contains one CSV file with ground truth annotations for each video in the test set,
* the <prediction directory> contains one CSV file with automatic predictions for each video in the test set.
In both directories, CSV files should be named 'test<video index>.csv' (test01.csv, test02.csv, ..., test25.csv).

The following dependencies must be installed: pandas, scikit-learn.
They can be installed as follows:
$ pip install pandas scikit-learn
"""

from argparse import ArgumentParser
from math import isnan
from os.path import join
from pandas import read_csv
from sklearn.metrics import roc_curve, auc

num_tools = 21
num_files = 25
file_prefix = 'test'
	

def auc_tool(truth_directory, prediction_directory, tool):
	"""Computes the area under the ROC curve for one tool.
	"""
	filename = ''
	try:
		truth = []
		predictions = []

		# loop on (truth, predictions) file pairs
		for file in range(1, num_files + 1):

			# getting the filenames
			if (file < 10):
				filename = file_prefix + '0{}.csv'.format(file)
			else:
				filename = file_prefix + '{}.csv'.format(file)
			truth_filename = join(truth_directory, filename)
			prediction_filename = join(prediction_directory, filename)

			# parsing the right column for the current tool
			truth_data = read_csv(truth_filename, header = 0, skipinitialspace = True,
								usecols = [tool], squeeze = True, dtype = 'float32').tolist()
			prediction_data = read_csv(prediction_filename, header = None, skipinitialspace = True,
									usecols = [tool], squeeze = True, dtype = 'float32').tolist()
			if len(truth_data) != len(prediction_data):
				raise ValueError('Files {} and {} have different row counts'.
								format(truth_filename, prediction_filename))

			# appending rows with consensual ground truth
			indices = [index for index, value in enumerate(truth_data) if value != 0.5]
			truth += [truth_data[index] for index in indices]
			predictions += [prediction_data[index] for index in indices]

		# computing the area under the ROC curve
		fpr, tpr, _ = roc_curve(truth, predictions)
		score = auc(fpr, tpr)
		return 0. if isnan(score) else score
	except Exception as e:
		print('Error: missing column in {} for tool number {}!'.format(filename, tool)
				if 'Usecols' in str(e) else 'Error: {}!'.format(e))
		return 0.


def main():
	"""Main function.
	"""

	# parsing the command line
	parser = ArgumentParser(description = 'Evaluator for the CATARACTS challenge.')
	parser.add_argument('-t', '--truth', required = True, help = 'directory containing ground truth files')
	parser.add_argument('-p', '--predictions', required = True, help = 'directory containing automatic predictions')
	args = parser.parse_args()

	# computing tool-specific scores
	scores = []
	for tool in range(1, num_tools + 1):
		score = auc_tool(args.truth, args.predictions, tool)
		print('Score tool {0}: {1:.4f}'.format(tool, score))
		scores.append(score)

	# computing the average score
	print('Average: {0:.4f}'.format(sum(scores) / float(len(scores))))


if __name__ == "__main__":
    main()