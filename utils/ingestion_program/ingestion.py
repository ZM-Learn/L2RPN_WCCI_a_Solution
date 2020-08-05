import os
import sys
import warnings
import argparse

import grid2op

from grid2op.Runner import Runner
from grid2op.Chronics import ChangeNothing
from grid2op.Agent import BaseAgent
from grid2op.Reward import BaseReward, RedispReward, L2RPNSandBoxScore
from grid2op.Action import TopologyAndDispatchAction

SUBMISSION_DIR_ERR = """
ERROR: Impossible to find a "submission" package.
Agents should be included in a "submission" directory
A module with a function "make_agent" to load the agent that will be assessed."
"""

MAKE_AGENT_ERR = """
ERROR:  We could NOT find a function name \"make_agent\"
in your \"submission\" package. "
We remind you that this function is mandatory and should have the signature:
 make_agent(environment, path_agent) -> agent 

 - The "agent" is the agent that will be tested.
 - The "environment" is a valid environment provided.
   It will NOT be updated during the scoring (no data are fed to it).
 - The "path_agent" is the path where your agent is located
"""

ENV_TEMPLATE_ERR = """
ERROR: There is no powergrid found for making the template environment. 
Or creating the template environment failed.
The agent will not be created and this will fail.
"""

MAKE_AGENT_ERR2 = """
ERROR: "make_agent" is present in your package, but can NOT be used.

We remind you that this function is mandatory and should have the signature:
 make_agent(environment, path_agent) -> agent

 - The "agent" is the agent that will be tested.
 - The "environment" is a valid environment provided.
   It will NOT be updated during the scoring (no data are fed to it).
 - The "path_agent" is the path where your agent is located
"""

BASEAGENT_ERR = """
ERROR: The "submitted_agent" provided should be a valid Agent. 
It should be of class that inherit "grid2op.Agent.BaseAgent" base class
"""

INFO_CUSTOM_REWARD = """
INFO: No custom reward for the assessment of your agent will be used.
"""

REWARD_ERR = """
ERROR: The "training_reward" provided should be a class.
NOT a instance of a class
"""

REWARD_ERR2 = """
ERROR: The "training_reward" provided is invalid.
It should inherit the "grid2op.Reward.BaseReward" class
"""

INFO_CUSTOM_OTHER = """
INFO: No custom other_rewards for the assessment of your agent will be used.
"""

KEY_OVERLOAD_REWARD = """
WARNING: You provided the key "{0}" in the "other_reward" dictionnary. 
This will be replaced by the score of the competition, as stated in the rules. Your "{0}" key WILL BE erased by this operation.
"""

def cli():
    DEFAULT_KEY_SCORE = "tmp_score_codalab"
    DEFAULT_NB_EPISODE = 10
    
    parser = argparse.ArgumentParser(description="Ingestion program")
    parser.add_argument("--dataset_path", required=True,
                        help="Path to the dataset")
    parser.add_argument("--output_path", required=True,
                        help="Path to the runner logs output dir")
    parser.add_argument("--program_path", required=True,
                        help="Path to the program dir")
    parser.add_argument("--submission_path", required=True,
                        help="Path to the submission dir")
    parser.add_argument("--key_score", required=False,
                        default=DEFAULT_KEY_SCORE, type=str,
                        help="Codalab other_reward name")
    parser.add_argument("--nb_episode", required=False,
                        default=DEFAULT_NB_EPISODE, type=int,
                        help="Number of episodes in the dataset")    
    return parser.parse_args()

def main():
    args = cli()
    
    # read arguments
    input_dir = args.dataset_path
    output_dir = args.output_path
    program_dir = args.program_path
    submission_dir = args.submission_path

    # create output dir if not existing
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("input dir: {}".format(input_dir))
    print("output dir: {}".format(output_dir))
    print("program dir: {}".format(program_dir))
    print("submission dir: {}".format(submission_dir))

    print("input content", os.listdir(input_dir))
    print("output content", os.listdir(output_dir))
    print("program content", os.listdir(program_dir))
    print("submission content", os.listdir(submission_dir))

    submission_location = os.path.join(submission_dir, "submission")
    if not os.path.exists(submission_location):
        print(SUBMISSION_DIR_ERR)
        raise RuntimeError(SUBMISSION_DIR_ERR)

    # add proper directories to path
    sys.path.append(program_dir)
    sys.path.append(submission_dir)

    try:
       from submission import make_agent
    except Exception as e:
        print(e)
        raise RuntimeError(MAKE_AGENT_ERR) from None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_template = grid2op.make(input_dir,
                                        chronics_class=ChangeNothing,
                                        action_class=TopologyAndDispatchAction)
            
    except Exception as e:
        raise RuntimeError(ENV_TEMPLATE_ERR)

    try:
        submitted_agent = make_agent(env_template, submission_location)
    except Exception as e:
        raise RuntimeError(MAKE_AGENT_ERR2)

    if not isinstance(submitted_agent, BaseAgent):
        raise RuntimeError(BASEAGENT_ERR)

    try:
        from submission import reward
    except:
        print(INFO_CUSTOM_REWARD)
        reward = RedispReward

    if not isinstance(reward, type):
        raise RuntimeError(REWARD_ERR)
    if not issubclass(reward, BaseReward):
        raise RuntimeError(REWARD_ERR2)

    try:
        from submission import other_rewards
    except:
        print(INFO_CUSTOM_OTHER)
        other_rewards = {}

    if args.key_score in other_rewards:
        print(KEY_OVERLOAD_WARN.format(args.key_score))
    other_rewards[args.key_score] = L2RPNSandBoxScore
    real_env = grid2op.make(input_dir,
                            reward_class=reward,
                            other_rewards=other_rewards)

    runner = Runner(**real_env.get_params_for_runner(),
                    agentClass=None, agentInstance=submitted_agent)
    path_save = os.path.abspath(output_dir)
    runner.run(nb_episode=args.nb_episode, path_save=path_save)

    print("Done and data saved in : \"{}\"".format(path_save))


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("ERROR: ingestion program failed with error: \n{}".format(e))
        print("Traceback is:")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
