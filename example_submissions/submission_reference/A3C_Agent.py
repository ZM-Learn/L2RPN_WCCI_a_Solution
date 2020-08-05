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


import tensorflow as tf
tf.disable_v2_behavior() # version issues

from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

import grid2op


loaded = np.load('actions_array.npz')
actions_array = np.transpose(loaded['actions_array'])  # this has 157 actions



class A3C_Agent(BaseAgent):
    """
    This is a class of "Greedy BaseAgent". Greedy agents are all executing the same kind of algorithm to take action:

      1. They :func:`grid2op.Observation.Observation.simulate` all actions in a given set
      2. They take the action that maximise the simulated reward among all these actions

    This class is an abstract class (object of this class cannot be created). To create "GreedyAgent" one must
    override this class. Examples are provided with :class:`PowerLineSwitch` and :class:`TopologyGreedy`.
    """
    
    def __init__(self, action_space):
        
        BaseAgent.__init__(self, action_space)
        self.state_size = 590
        self.action_size = 596      
        self.hidden1, self.hidden2 = 800, 600
        self.actor, self.critic = self.build_model()

        
        self.sess = tf.InteractiveSession()
        # TF 1.x - sess = tf.InteractiveSession(); TF 2.X sess=tf.compat.v1.InteractiveSession()
        
        K.set_session(self.sess) # tensorflow 1.X
        #tf.compat.v1.keras.backend.set_session(self.sess) # tensorflow 2.X
        
        
        #tf.compat.v1.disable_eager_execution() # compatibility issues due to tf 2.0
        
        self.sess.run(tf.global_variables_initializer())  # tensorflow 1.X
        
        self.tested_action = None
        
    def build_model(self):
        state = Input(batch_shape=(None,  self.state_size))
        shared = Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(state)

        actor_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(shared)
        action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(actor_hidden)

        value_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(shared)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(value_hidden)

        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    def get_action(self, env, state):
        num_compared_action = 30
        policy_nn = self.actor.predict(np.reshape(self.get_usable_observation(state), [1, self.state_size]))[0] 
        #indx = map(policy_nn.index, heapq.nlargest(num_compared_action, policy_nn))
        #predicted_action = np.argmax([policy_nn])
        #policy_nn_subid_mask = policy_nn * (1 - actions_array.dot((state[-14:]>0).astype(int))) 
        # this masking prevents any illegal operation
        
        #print("policy_nn: ", policy_nn)
        #print("actions_array.dot: ", actions_array.dot((state[-0:]>0).astype(int)))
        
        #policy_nn_subid_mask = policy_nn * (1 - actions_array.dot((state[-0:]>0).astype(int))) 
        

        policy_chosen_list = np.random.choice(self.action_size, num_compared_action, replace=True,
                                              p=policy_nn / sum(policy_nn))
        # sample 4 actions
        
        #print('probability:', policy_nn / sum(policy_nn) )
        #print('policy_chosen_list', policy_chosen_list)
        policy_chosen_list = np.hstack((0, policy_chosen_list)) # adding no action option # comment this line as agent learns...
        
        #print("policy_chosen_list:", policy_chosen_list)
        
        #print("actions_array[policy_chosen_list[0],:] :", actions_array[policy_chosen_list[0],:])
        
        action_asclass = [None]*num_compared_action
        reward_simu = [None]*num_compared_action
        for i in range(num_compared_action):
            action_asclass[i] = env.action_space({})
            action_asclass[i].from_vect(actions_array[policy_chosen_list[i],:])
            state_temp = state
            obs_0, reward_simu[i], done_0, infos0  = state_temp.simulate(action_asclass[i])

        #print("np.argmax([rw_0,rw_1,rw_2,rw_3]):",np.argmax([rw_0,rw_1,rw_2,rw_3]))
        return policy_chosen_list[np.argmax([reward_simu])] # origin
    

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")


    def est_reward_update(self,obs,rw,done): # penalizing overloaded lines
        #obs = observation_space.array_to_observation(obs) if not done else 0
        
        state_obs = obs
        rw_0 = rw - 500 * sum(((( state_obs.rho - 1) )[
                        state_obs.rho > 1])) if not done else -100
        return rw_0

    
    def get_usable_observation(self,obs): # penalizing overloaded lines
        #obs = observation_space.array_to_observation(obs) if not done else 0
        
        prod_p = obs.prod_p /10
        prod_q = obs.prod_q /100
        prod_v = obs.prod_v /100
        load_p = obs.load_p /10
        load_q = obs.load_q /100
        load_v = obs.load_v /100 # voltage setpoint of the loads
        rho = obs.rho
        # topology_information = obs.connectivity_matrix()
        topo_vect = obs.topo_vect
        
        line_status = obs.line_status
        
        time_before_cooldown_line = obs.time_before_cooldown_line
        time_before_cooldown_sub = obs.time_before_cooldown_sub
        time_next_maintenance = obs.time_next_maintenance
        duration_next_maintenance = obs.duration_next_maintenance
        
        usable_observation = np.hstack((prod_p,prod_q,prod_v,load_p,load_q,load_v,rho,topo_vect,line_status,time_before_cooldown_line,time_next_maintenance))
        
        return usable_observation




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
        env = grid2op.make("l2rpn_wcci_2020")
        action = self.get_action(env, observation)
                        
        action_asvector = actions_array[action,:]
        this_action = env.action_space({})
        this_action.from_vect(action_asvector)
        
        simul_obs, simul_reward, simul_has_error, simul_info = observation.simulate(this_action)
        simul_obs, reward_nonaction, simul_has_error, simul_info = observation.simulate(self.action_space({}))
        
        if np.max(simul_reward)> reward_nonaction+10:
            best_action = self.tested_action[reward_idx]
            print(best_action.to_vect())
        else:
            best_action = self.action_space({})

        return best_action

    


