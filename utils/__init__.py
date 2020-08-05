import os

UTILS_PATH = os.path.split(__file__)[0]

# reference path valid for the data
problem_dir = os.path.join(UTILS_PATH, 'ingestion_program/')
score_dir = os.path.join(UTILS_PATH, 'scoring_program/')
ref_data = os.path.join(UTILS_PATH, 'public_ref/')
ingestion_output = os.path.join(UTILS_PATH, 'logs/')

input_data_check_dir = os.path.join(os.path.split(UTILS_PATH)[0], 'l2rpn_data/')
output_dir = os.path.join(UTILS_PATH, 'output/')
