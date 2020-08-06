
#from grid2op.Agent import BaseAgent
#from grid2op.Reward import L2RPNReward

from grid2op.Agent import BaseAgent

import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras.layers import Dense, Input
from keras.models import Model
from .Backup_agent import Backup_agent
import os

loaded = np.load(os.path.split(os.path.realpath(__file__))[0]+'/'+ 'actions_array.npz')
actions_array = np.transpose(loaded['actions_array'])  # this has 157 actions

loaded1 = np.load(os.path.split(os.path.realpath(__file__))[0]+'/'+ 'actions_array_backup.npz')
actions_array_backup = np.transpose(loaded1['actions_array'])  # this has 157 actions

class MyAgent(BaseAgent):
    """
    The template to be used to create an agent: any controller of the power grid is expected to be a subclass of this
    grid2op.Agent.BaseAgent.
    """
    def __init__(self, action_space):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=action_space)
        self.state_size = 590+177
        self.action_size = 596      
        self.hidden1, self.hidden2, self.hidden3 = 600, 400, 400
        
        self.actor = self.build_model()
        
        self.load_model('pypow_wcci_a3c')
        print("Loaded saved NN model parameters \n")
        tf.get_default_graph()
        self.backup_agent = Backup_agent(action_space)
        #self.sess = tf.InteractiveSession()
        #K.set_session(self.sess)

        #self.sess = tf.InteractiveSession()
        # TF 1.x - sess = tf.InteractiveSession(); TF 2.X sess=tf.compat.v1.InteractiveSession()
        #K.set_session(self.sess) # tensorflow 1.X
        #tf.compat.v1.keras.backend.set_session(self.sess) # tensorflow 2.X
        #tf.compat.v1.disable_eager_execution() # compatibility issues due to tf 2.0
        #self.sess.run(tf.global_variables_initializer())  # tensorflow 1.X
        self.tested_action = None



    def build_model(self):
        state = Input(batch_shape=(None,  self.state_size))
        shared = Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(state)

        actor_hidden1 = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(shared)
        actor_hidden2 = Dense(self.hidden3, activation='relu', kernel_initializer='he_uniform')(actor_hidden1)
        action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(actor_hidden2)


        actor = Model(inputs=state, outputs=action_prob)

        actor._make_predict_function()

        return actor

    def get_action(self, state):
        num_compared_action = 2
        policy_nn = self.actor.predict(np.reshape(self.get_usable_observation(state), [1, self.state_size]))[0] 
        
        #indx = map(policy_nn.index, heapq.nlargest(num_compared_action, policy_nn))
        policy_chosen_list = np.argsort(policy_nn)[-1: -num_compared_action-1: -1]
        policy_chosen_list[num_compared_action-1] = 0 # force no action inside decisions

        
        action_asclass = [None]*num_compared_action
        obs_0 = [None]*num_compared_action
        done_0 = [None]*num_compared_action
        reward_simu = [0]*num_compared_action
        reward_simu[num_compared_action-1] = 0.1
        for i in range(num_compared_action):
            action_asclass[i] = self.action_space({})
            action_asclass[i].from_vect(actions_array[policy_chosen_list[i],:])
            obs_0[i], reward_simu[i], done_0[i], infos0  = state.simulate(action_asclass[i])
            reward_simu[i] = reward_simu[i]+self.est_reward_update(obs_0[i],reward_simu[i],done_0[i])

        if np.max(reward_simu)< 0: # contingency
            print(np.max(reward_simu))
            additional_action = 80
            policy_chosen_list = np.argsort(policy_nn)[-1: -additional_action-num_compared_action-1: -1]
            
            action_asclass = [None]*additional_action
            reward_simu2 = [0]*additional_action
            for i in range(additional_action):
                action_asclass[i] = self.action_space({})
                action_asclass[i].from_vect(actions_array[policy_chosen_list[num_compared_action+i],:])
                obs_1, reward_simu2[i], done_1, _  = state.simulate(action_asclass[i])
                reward_simu2[i] = self.est_reward_update(obs_1,reward_simu2[i],done_1)
                if reward_simu2[i] > 0:
                    return obs_1, policy_chosen_list[num_compared_action+i],reward_simu2[i],done_1 # origin
                
           
        #print("np.argmax([rw_0,rw_1,rw_2,rw_3]):",np.argmax([rw_0,rw_1,rw_2,rw_3]))
        return obs_0[np.argmax([reward_simu])] , policy_chosen_list[np.argmax([reward_simu])],np.max([reward_simu]),done_0[np.argmax([reward_simu])] # origin
    

    def load_model(self, name):
        self.actor.load_weights(os.path.split(os.path.realpath(__file__))[0]+'/'+ name + "_actor.h5")

    def est_reward_update(self,obs,rw,done): # penalizing overloaded lines
        #obs = observation_space.array_to_observation(obs) if not done else 0
        
        state_obs = obs
        rw_0 = rw - 50 * sum(((( state_obs.rho - 0.99) )[
                        state_obs.rho > 1])) if not done else -200
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
        
        usable_observation = np.hstack((prod_p,prod_q,prod_v,load_p,load_q,load_v,rho,topo_vect, topo_vect-1,line_status,time_before_cooldown_line,time_next_maintenance))
        
        return usable_observation




    def act(self, observation,reward, done=False):
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

        
        if min(observation.rho < 0.8): # seems 0.8 is the best
            this_action = self.action_space({})
        else:
            obs_0, action, rewd_simu_0, done_0 = self.get_action(observation)
            action_asvector = actions_array[action,:]
            
            if done_0 or rewd_simu_0<0:
                reward_simu_2, policy_chosen_backup = self.backup_agent.get_action(observation)
                if reward_simu_2>rewd_simu_0:
                    action_asvector = actions_array_backup[policy_chosen_backup,:]
                
            this_action = self.action_space({})
            this_action.from_vect(action_asvector)
        #best_action = this_action
        
        '''
        # OPF is after topology decisions
        '''        

        return this_action

