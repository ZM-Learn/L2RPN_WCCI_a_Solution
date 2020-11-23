
from .agents import MyAgent
from grid2op.Reward import L2RPNReward



def make_agent(env, submission_dir):
    """
    This function will be used by codalab to create your agent. It should accept exactly an environment and a path
    to your submission directory and return a valid agent.
    """
    agent = MyAgent(env.action_space)
    return agent

# reward must be a subclass of grid2op.Reward.BaseReward.BaseReward:
reward = L2RPNReward # you can also create your own reward class
