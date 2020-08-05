# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from abc import abstractmethod
import numpy as np
from grid2op.Agent.BaseAgent import BaseAgent
from grid2op.dtypes import dt_float


class MyGreedyAgent(BaseAgent):
    """
    This is a class of "Greedy BaseAgent". Greedy agents are all executing the same kind of algorithm to take action:

      1. They :func:`grid2op.Observation.Observation.simulate` all actions in a given set
      2. They take the action that maximise the simulated reward among all these actions

    This class is an abstract class (object of this class cannot be created). To create "GreedyAgent" one must
    override this class. Examples are provided with :class:`PowerLineSwitch` and :class:`TopologyGreedy`.
    """
    def __init__(self, action_space):
        BaseAgent.__init__(self, action_space)
        self.tested_action = None

    def act(self, observation, done=False):
        """
        By definition, all "greedy" agents are acting the same way. The only thing that can differentiate multiple
        agents is the actions that are tested.

        These actions are defined in the method :func:`._get_tested_action`. This :func:`.act` method implements the
        greedy logic: take the actions that maximizes the instantaneous reward on the simulated action.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The current observation of the :class:`grid2op.Environment.Environment`

        reward: ``float``
            The current reward. This is the reward obtained by the previous action

        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility

        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The action chosen by the bot / controller / agent.

        """
        self.tested_action = self._get_tested_action( observation)
        if len(self.tested_action) > 1:
            all_rewards = np.full(shape=len(self.tested_action), fill_value=np.NaN, dtype=dt_float)
            for i, action in enumerate(self.tested_action):
                simul_obs, simul_reward, simul_has_error, simul_info = observation.simulate(action)
                all_rewards[i] = simul_reward
            
            if np.max(all_rewards)>all_rewards[0]*1.05:
                reward_idx = int(np.argmax(all_rewards))  # rewards.index(max(rewards))
                best_action = self.tested_action[reward_idx]
                print(best_action.to_vect())
            else:
                best_action = self.action_space({})
        else:
            best_action = self.action_space({})
        return best_action

    
    def _get_tested_action(self, observation):
        res = [self.action_space({})]  # add the do nothing
        # better use "get_all_unitary_topologies_set" and not "get_all_unitary_topologies_change"
        # maybe "change" are still "bugged" (in the sens they don't count all topologies exactly once)
        res += self.action_space.get_all_unitary_topologies_set(self.action_space)
        self.li_actions = res
        
        # my added for generation redispatch
        #(gen_idx, ramp)
        # actiongen = env.action_space({"redispatch": [(0, 1),(1, 3),(3, 4)]})
        #res.append(action)
        
        return self.li_actions


