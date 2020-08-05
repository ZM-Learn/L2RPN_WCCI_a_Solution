import os
import sys
import zipfile
import tempfile
import datetime
import warnings

import grid2op


def zip_for_codalab(path_agent, dest="."):

    folder = os.path.abspath(path_agent)
    if not os.path.exists(folder):
        raise RuntimeError("The folder \"{}\" is empty and cannot be send to codalab to serve as a submitted agent.")

    root, directory_ = os.path.split(folder)
    sys.path.append(root)
    try:
        exec(f"from {directory_} import make_agent")
    except:
        raise RuntimeError(f"Impossible to import the \"make_agent\" function that is used by codalab to... create "
                           "your agent. Your submission is not valid. Please make sure the {path_agent} module "
                           "exposes the \"make_agent\" function.")

    try:
        exec(f"from {directory_} import reward")
    except:
        print("INFO: No custom reward for the assessment of your agent will be used. If you want to use a custom "
              "reward when your agent is evaluated, make sure to export  \"reward\", which should be a class "
              "inheriting from grid2op.BaseReward in your module (done in __init__.py).")

    try:
        exec(f"from {directory_} import other_rewards")
    except:
        print("INFO: No custom other_rewards for the assessment of your agent will be used. If you want to get "
              "information about other rewards when your agent is evaluated, make sure to export  \"other_rewards\" "
              "dictionnary in your module (you can do it in the __init__.py file)")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env_test = grid2op.make("rte_case5_example", test=True)
    try:
        toto = make_agent(env_test, folder)
    except TypeError:
        print("WARNING: a call to \"make_agent(environment, path_agent)\" raise a TypeError. There are great chances "
              "that your agent will not be valid on codalab. This is a warning but not an error, because the "
              "environment and the path_agent are 'fake' data for the test. If i were you i would double check "
              "though.")
    except:
        pass

    print("Your submission appear to be valid. For more test, we encourage you to run the appropriate notebook to "
          "do these unit testing.")
    the_date = datetime.datetime.now()
    zipped_submission = f'submission_{the_date:%y-%m-%d-%H-%M}.zip'

    dest = os.path.abspath(dest)
    zip_file_name = os.path.join(dest, zipped_submission)
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("metadata", 'This is internal to codalab, do not modify')
        # Do stuff here
        for root, dirs, files in os.walk(folder, topdown=True):
            dirs[:] = [d for d in dirs if d not in {"__pycache__"}]
            for file_ in files:
                if os.path.splitext(file_)[1] == ".pyc":
                    continue
                arc_path = os.path.relpath(os.path.join(root, file_), os.path.join(folder))
                arc_path = os.path.join("submission", arc_path)
                zipf.write(os.path.join(root, file_), arcname=arc_path)

    print(f"The zip file \"{zip_file_name}\" has been created with your submission in it.")
    return zip_file_name
