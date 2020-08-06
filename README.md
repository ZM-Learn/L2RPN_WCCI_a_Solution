Learning to Run a Power Network WCCI 2020 Competition - One possible solution
=====
# L2RPN WCCI 2020 Competition
This is a repository for L2RPN WCCI 2020 Competition during **_June 2020 - July 2020_**. The RL agent based on this repository is ranked 3rd in the competition.

https://competitions.codalab.org/competitions/24902#learn_the_details

## Summary
The goal for L2RPN WCCI competition is to train an agent that will be able to learn a policy that could overcome obstacles, such as congestions, while optimizing the overall operation cost. The operation cost to be minimized includes powerlines losses, redispatch cost and blackout cost. The participants are encouraged to operate the grid for as long as possible, and will be penalized for a blackout even after the game is over. Load/RES fluctuations, line matainance and hazards are also considered by the competition.

In the competition grid, substations have a "double busbar layout". Eeach connection between a substation and an object (generator, load or end of a powerline) can be made on the first bus, or the second bus, or not at all if there is a disconnection. Consequently, valid control action includes generation redispatch, line switching and connection of substations' busbar.

This repository presents a policy-based RL agent for the competition. Two agents are trained with different strategies/tricks and (randomly selected) datasets. To improve the control performance, the agents will backup each other during the test phase. The proposed method is inspired by the works of GEIRINA (https://github.com/shidi1985/L2RPN) and Amar (https://github.com/amar-iastate/L2RPN-using-A3C).

## Environment
These codes are programmed in Python 3.7.6. The dependent packages are listed in [requirements.txt](./requirements.txt), and a quick install command is given as follows:

`pip3 install -r requirements.txt`

Compared with the environment provided by L2RPN competition, Keras is changed to 2.1.6, Tensorflow is changed to 1.14.


## Basic Usage

### Runing Test
Make the submission file and test the performance of agent:
```
python make_submission_file.py
```
The test results on local datasets will be shown in "\results\results.html".

Compared with the submitted agent, it seems that the performance of agent can be further improved with more iterations:

![Image text]\results\Results.png

### Try other actions
The actions of our agent include randomly selected topology action (line swithching, substation bus bar connection) and generation redispatch. Due to strict ramping limits, all actions are recongised as discretized actions. If you would like to try other selection of action space, you can change the file "Data_structure_process.py" and run:
```
python Data_structure_process.py
```

### Train the agent
To train the agent, run:
```
python Train_Agent_discrete.py
```
You may try out other tricks for training the agent (experience replay for critic NN, forced exploration, security verification during training, other setting of hyperparameters), these codes are annotated for your reference. The proposed agent was trained with different strategies (based on NN parameters of previous training process).

## License
This Source Code is subject to the terms of the GNU Lesser General Public License v3.0. If a copy of the LGPL-v3 was not distributed with this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.html.