
#from grid2op.Agent import BaseAgent
#from grid2op.Reward import L2RPNReward

from grid2op.Agent import BaseAgent

import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras.layers import Dense, Input
from keras.models import Model

import os

loaded = np.load(os.path.split(os.path.realpath(__file__))[0]+'/'+ 'actions_array_backup.npz')
actions_array = np.transpose(loaded['actions_array'])  # this has 157 actions


class Backup_agent2(BaseAgent):
    """
    The template to be used to create an agent: any controller of the power grid is expected to be a subclass of this
    grid2op.Agent.BaseAgent.
    """
    def __init__(self, action_space):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=action_space)
        self.state_size = 943+177
        self.action_size = 987+21      
        self.hidden1, self.hidden2, self.hidden3 = 1000, 1000, 1000
        
        self.actor = self.build_model()
            
        self.load_model('pypow_wcci_a3c_backup2')
        print("Loaded saved NN model parameters \n")
        tf.get_default_graph()
        
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
        action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(actor_hidden1)
        
        actor = Model(inputs=state, outputs=action_prob)

        actor._make_predict_function()

        return actor

    def get_action(self, state):
        policy_nn = self.actor.predict(np.reshape(self.get_usable_observation(state), [1, self.state_size]))[0] 
        policy_chosen = np.argsort(policy_nn)[0]
        
        action_asclass = self.action_space({})
        action_asclass.from_vect(actions_array[policy_chosen,:])
        obs_, reward_simu_, done_, infos  = state.simulate(action_asclass)
        reward_simu_ = self.est_reward_update(obs_, reward_simu_, done_)
        
        if done_ or sum(((( obs_.rho - 1) )[obs_.rho > 1.1]))>0:
            #reward_simu_ = 30-reward_simu_/500
            
            additional_action = 2
            policy_chosen_list = np.argsort(policy_nn)[-1: -additional_action-1-1: -1]
            
            action_asclass = [None]*additional_action
            reward_simu1 = [0]*additional_action
            for i in range(additional_action):
                action_asclass[i] = self.action_space({})
                action_asclass[i].from_vect(actions_array[policy_chosen_list[1+i],:])
                obs_0, reward_simu1[i], done_0, _  = state.simulate(action_asclass[i])
                #reward_simu1[i] = 30-reward_simu1[i]/500
                reward_simu1[i] = self.est_reward_update(obs_0,reward_simu1[i],done_0)
                if (not done_0) and (sum(((( obs_0.rho - 1) )[obs_0.rho > 1.1]))==0):
                    #print('this one not done', np.max(reward_simu1),reward_simu1[i], i,sum(((( obs_0.rho - 1) )[obs_0.rho > 1.01])) )
                    
                    return reward_simu1[i]+5, policy_chosen_list[1+i] # origin
                
            if np.max(reward_simu1)>reward_simu_:
                #print(np.max(reward_simu1))
                return np.max(reward_simu1), policy_chosen_list[1+np.argmax([reward_simu1])] # origin
        
        
        
        
        return reward_simu_, policy_chosen
        
        
        
        '''
        # compare based
        num_compared_action = 2
        policy_nn = self.actor.predict(np.reshape(self.get_usable_observation(state), [1, self.state_size]))[0] 
        
        #indx = map(policy_nn.index, heapq.nlargest(num_compared_action, policy_nn))
        policy_chosen_list = np.argsort(policy_nn)[-1: -num_compared_action-1: -1]
        policy_chosen_list[num_compared_action-1] = 0 # force no action inside decisions

        
        action_asclass = [None]*num_compared_action
        reward_simu = [0]*num_compared_action
        reward_simu[num_compared_action-1] = 0.01
        for i in range(num_compared_action):
            action_asclass[i] = self.action_space({})
            action_asclass[i].from_vect(actions_array[policy_chosen_list[i],:])
            obs_0, reward_simu[i], done_0, infos0  = state.simulate(action_asclass[i])
            #reward_simu[i] = 500-reward_simu[i]/50
            reward_simu[i] = self.est_reward_update(obs_0,reward_simu[i],done_0)

        
        if np.max(reward_simu)< 0: # contingency
            print(np.max(reward_simu))
            additional_action = 1000
            policy_chosen_list = np.argsort(policy_nn)[-1: -additional_action-num_compared_action-1: -1]
            
            action_asclass = [None]*additional_action
            reward_simu1 = [0]*additional_action
            for i in range(additional_action):
                action_asclass[i] = self.action_space({})
                action_asclass[i].from_vect(actions_array[policy_chosen_list[num_compared_action+i],:])
                obs_0, reward_simu1[i], done_0, _  = state.simulate(action_asclass[i])
                #reward_simu[i] = 500-reward_simu[i]/50
                reward_simu1[i] = self.est_reward_update(obs_0,reward_simu1[i],done_0)
                if reward_simu1[i] > 0:
                    return policy_chosen_list[num_compared_action+i] # origin
                if np.max(reward_simu1)>np.max(reward_simu):
                    return policy_chosen_list[num_compared_action+np.argmax([reward_simu1])] # origin
                    
        #print("np.argmax([rw_0,rw_1,rw_2,rw_3]):",np.argmax([rw_0,rw_1,rw_2,rw_3]))
        return policy_chosen_list[np.argmax([reward_simu])] # origin
        '''
    

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
        
        all_observation = obs.to_vect()
        obs_numerical = all_observation[6:713]
        obs_numerical[649:706] = obs_numerical[649:706]*100
        topo_vect = obs.topo_vect
        line_status = obs.line_status
        usable_observation = np.hstack((obs_numerical,topo_vect,topo_vect-1,line_status))
        
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

        
        if min(observation.rho < 0.6): # seems 0.8 is the best
            this_action = self.action_space({})
        else:
            action = self.get_action(observation)
            action_asvector = actions_array[action,:]
            this_action = self.action_space({})
            this_action.from_vect(action_asvector)
        
        #best_action = this_action
        
        '''
        # OPF is after topology decisions
        '''        

        return this_action

