import os
import sys
import tempfile
import subprocess
import argparse
from zipfile import ZipFile
import pandas as pd

from utils.zip_for_codalab import zip_for_codalab
from utils import input_data_check_dir, problem_dir, score_dir

DEFAULT_MODEL_DIR = 'example_submission/submission'

INFO_ZIP_CREATE = """
INFO: Basic check and creation of the zip file for the folder {}
"""
INFO_UNZIP = """
INFO: Checking the zip file can be unzipped.
"""
INFO_CONTENT = """
INFO: Checking content is valid
"""

INFO_META = """
INFO: metadata found.
"""

INFO_RUNNING = """
INFO: This might take a while..
It will evaluate your agent on a whole lot of scenarios
(10 scenarios, with similar number of timesteps than the validation set)
"""

INFO_RUN_SUCCESS = """
INFO: Your agent could be run correctly. 
You can now check its performance
"""

INFO_RESULT = """
INFO: Check if the results can be read back
"""

INFO_SCORE = """
Your scores are :
(remember these score are not at all an indication of \
what will be used in codalab, as the data it is tested \
on are really different):"
"""

ERR_META = """
ERROR: Submission invalid
There is no file "metadata" in the zip file you provided:
{}
Did you zip it with using "zip_for_codalab" ?
"""

ERR_RUNNING = """
ERROR: Your agent could not be run. 
It will probably fail on codalab.
Here is the information we have:
"""

def main(model_dir): 
    print(INFO_ZIP_CREATE.format(model_dir))
    archive_path = zip_for_codalab(os.path.join(os.path.abspath(model_dir)))
    
    print(INFO_UNZIP)
    tmp_dir = tempfile.TemporaryDirectory()
    sys.path.append(tmp_dir.name)
    with ZipFile(archive_path, 'r') as zipObj:
        # Extract all the contents of zip file in different directory
        zipObj.extractall(tmp_dir.name)
    
    print(INFO_CONTENT)
    if not os.path.exists(os.path.join(tmp_dir.name, "metadata")):
        raise RuntimeError(ERR_META.format(archive_path))
    else:
        print(INFO_META)

    print(INFO_RUNNING)
    run_output_dir = os.path.join("utils", "res")
    li_cmd = [
            sys.executable,
            os.path.join(problem_dir, "ingestion.py"),
            "--dataset_path", input_data_check_dir,
            "--output_path", run_output_dir,
            "--program_path", problem_dir,
            "--submission_path", tmp_dir.name
        ]
    res_ing = subprocess.run(
        li_cmd,
        stdout=subprocess.PIPE
    )
    if res_ing.returncode != 0:
        print(ERR_RUNNING)
        print(res_ing.stdout.decode('utf-8'))
        if res_ing.stderr is not None:
            print("----------")
            print("Error message:")
            print(res_ing.stderr.decode('utf-8'))
        print()
        print()
        print("You can run \n\"{}\"\n for more debug information".format(" ".join(li_cmd)))
        raise RuntimeError("INVALID SUBMISSION")
    else:
        print(INFO_RUN_SUCCESS)
    
    print(INFO_RESULT)
    scoring_output_dir = 'results'
    ingestion_output = os.path.abspath(run_output_dir)
    config_valid = os.path.join(score_dir, "config_0.json")
    res_sc = subprocess.run(
        [
            sys.executable,
            os.path.join(score_dir, "scoring.py"),
            "--logs_in", ingestion_output,
            "--config_in", config_valid,
            "--data_out", scoring_output_dir
        ],
        stdout=subprocess.PIPE
    )

    if res_sc.returncode != 0:
        print(ERR_RUNNING)
        print(res_sc.stdout.decode('utf-8'))
        raise RuntimeError("INVALID SUBMISSION")

    with open(os.path.join(scoring_output_dir, "scores.txt"), "r") as f:
        scores = f.readlines()
    scores = [el.rstrip().lstrip().split(":") for el in scores]
    print(INFO_SCORE)
    res = pd.DataFrame(scores)
    print (res)
    return res

def cli():
    parser = argparse.ArgumentParser(description="Zip and check codalab.")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                        help="Path of the model you want to submit.")
    return parser.parse_args()

if __name__ == "__main__":
    args = cli()
    main(args.model_dir)
    
