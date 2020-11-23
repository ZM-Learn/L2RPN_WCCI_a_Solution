
#from grid2op.Agent import BaseAgent
#from grid2op.Reward import L2RPNReward

from grid2op.Agent import BaseAgent
import networkx as nx
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras.layers import Dense, Input
from keras.models import Model
#from .Backup_agent import Backup_agent
import os


loaded = np.load(os.path.split(os.path.realpath(__file__))[0]+'/'+ 'actions_array_useful.npz')
actions_array = np.transpose(loaded['actions_array_useful'])  # this has 157 actions

loaded1 = np.load(os.path.split(os.path.realpath(__file__))[0]+'/'+ 'actions_array_backup.npz')
actions_array_backup = np.transpose(loaded1['actions_array_useful'])  # this has 157 actions

#loaded1 = np.load('state_mean.npz')
#state_mean = loaded1['state_mean']
#state_mean = state_mean[1:]

loaded2 = np.load(os.path.split(os.path.realpath(__file__))[0]+'/'+ 'norm_range.npz')
norm_range = loaded2['norm_range']
norm_range = norm_range[1:]

loaded2 = np.load(os.path.split(os.path.realpath(__file__))[0]+'/'+ 'norm_min.npz')
norm_min = loaded2['norm_min']
norm_min = norm_min[1:]


class MyAgent(BaseAgent):
    """
    The template to be used to create an agent: any controller of the power grid is expected to be a subclass of this
    grid2op.Agent.BaseAgent.
    """
    def __init__(self, action_space):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=action_space)
        self.state_size = 1040
        self.action_size = 1163  
        self.hidden1, self.hidden2 = 1000, 800
        
        self.actor = self.build_model()
            
        self.load_model('pypow_neurips_a3c')
        print("Loaded saved NN model parameters \n")
        
        #self.backup_agent2 = Backup_agent2(action_space)
        
        
        tf.get_default_graph()
        
        #self.backup_agent = Backup_agent(action_space)
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
        #state_norm = BatchNormalization()(state)
        shared = Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(state)

        actor_hidden1 = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(shared)
        action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(actor_hidden1)
        
        actor = Model(inputs=state, outputs=action_prob)

        actor._make_predict_function()

        return actor

    '''
    def get_action(self, state):
        policy_nn = self.actor.predict(np.reshape(self.get_observation_vector(state), [1, self.state_size]))[0] 
        policy_chosen = np.argsort(policy_nn)[0]
        
        action_asclass = self.action_space({})
        action_asclass.from_vect(actions_array[policy_chosen,:])
        obs_0, reward_simu_0, done_0, infos  = state.simulate(self.action_space({}))
        reward_simu_0 = self.est_reward_update(obs_0, reward_simu_0, done_0)
        obs_, reward_simu_, done_, infos  = state.simulate(action_asclass)
        reward_simu_ = self.est_reward_update(obs_, reward_simu_, done_)
        
        if reward_simu_>=reward_simu_0:
                
            if np.max(reward_simu_)< 0: # contingency
                print(np.max(reward_simu_))
                additional_action = 300
                num_compared_action = 1
                policy_chosen_list = np.argsort(policy_nn)[-1: -additional_action-num_compared_action-1: -1]
            
                action_asclass = [None]*additional_action
                reward_simu1 = [0]*additional_action
                for i in range(additional_action):
                    action_asclass[i] = self.action_space({})
                    action_asclass[i].from_vect(actions_array[policy_chosen_list[num_compared_action+i],:])
                    obs_0, reward_simu1[i], done_0, _  = state.simulate(action_asclass[i])
                    #reward_simu1[i] = 30-reward_simu[i]/500
                    reward_simu1[i] = self.est_reward_update(obs_0,reward_simu1[i],done_0)
                    if reward_simu1[i] > 0:
                        return reward_simu1[i], policy_chosen_list[num_compared_action+i] # origin
                if np.max(reward_simu1)>np.max(reward_simu_):
                    return np.max(reward_simu1), policy_chosen_list[num_compared_action+np.argmax([reward_simu1])] # origin
                
                return np.max(reward_simu_), policy_chosen_list[np.argmax([reward_simu_])]
    
                return reward_simu_, policy_chosen
            
            
            return reward_simu_, policy_chosen
        
        return reward_simu_0,0
    '''
    def get_action(self, state):
        policy_nn = self.actor.predict(np.reshape(self.get_observation_vector(state), [1, self.state_size]))[0] 
        policy_chosen = np.argsort(policy_nn)[0]
        
        action_asclass = self.action_space({})
        action_asclass.from_vect(actions_array[policy_chosen,:])
        obs_, reward_simu_, done_, infos  = state.simulate(action_asclass)
        reward_simu_ = self.est_reward_update(obs_, reward_simu_, done_)
        
        if done_ or sum(((( obs_.rho - 1) )[obs_.rho > 1.0]))>0:
            #reward_simu_ = 30-reward_simu_/500
            
            #print(reward_simu_)
            additional_action = 1162
            policy_chosen_list = np.argsort(policy_nn)[-1: -additional_action-1-1: -1]

            action_asclass = [None]*additional_action
            reward_simu1 = [0]*additional_action
            for i in range(additional_action):
                action_asclass[i] = self.action_space({})
                action_asclass[i].from_vect(actions_array[policy_chosen_list[1+i],:])
                obs_0, reward_simu1[i], done_0, _  = state.simulate(action_asclass[i])
                #reward_simu1[i] = 30-reward_simu1[i]/500
                reward_simu1[i] = self.est_reward_update(obs_0,reward_simu1[i],done_0)
                if (not done_0) and (sum(((( obs_0.rho - 1) )[obs_0.rho > 1.01]))==0):
                    #print('this one not done', np.max(reward_simu1),reward_simu1[i], i,sum(((( obs_0.rho - 1) )[obs_0.rho > 1.01])) )
                    return reward_simu1[i]+5, policy_chosen_list[1+i] # origin

            
            #reward_simu_2, policy_chosen_backup = self.backup_agent2.get_action(state)
            
            #if np.max(reward_simu1)>np.max([reward_simu_,reward_simu_2]):
                #print(np.max(reward_simu1))
            #    return np.max(reward_simu1), policy_chosen_list[1+np.argmax([reward_simu1])] # origin            
            
            if  np.max(reward_simu1)>reward_simu_:
                #print(np.max(reward_simu1))
                return np.max(reward_simu1), policy_chosen_list[1+np.argmax([reward_simu1])] # origin     

            return reward_simu_, policy_chosen
        
        
        return reward_simu_, policy_chosen
 

    def load_model(self, name):
        self.actor.load_weights(os.path.split(os.path.realpath(__file__))[0]+'/'+ name + "_actor.h5")


    def est_reward_update(self,obs,rw,done): # penalizing overloaded lines
        #obs = observation_space.array_to_observation(obs) if not done else 0
        
        state_obs = obs
        rw_0 = rw - 10 * sum(((( state_obs.rho - 1) )[
                        state_obs.rho > 1])) if not done else -200
        return rw_0
    '''
    def est_reward_update(self,obs,rw,done): # penalizing overloaded lines
        #obs = observation_space.array_to_observation(obs) if not done else 0
        
        state_obs = obs
        if not done:
            rw_0 = rw/2 - 2 * sum(((( state_obs.rho - 0.99) )[state_obs.rho > 1])) - 0.1 * sum(((( state_obs.rho - 0.8) )[state_obs.rho > 0.8]))  
            if rw_0<-5:
                rw_0 = -5
        else:
            rw_0 = -10
        return rw_0
    '''

    
    def get_observation_vector(self,OBS_0):
        
    
        connectivity_ = OBS_0.connectivity_matrix()
        
        graph = nx.from_numpy_matrix(connectivity_)
        
        #obs_graph = ObservationSpaceGraph(observation_space)
        #obs_graph.get_state_with_graph_features(observation)
        
        # The result is a ranking of the substations in the grid taking into account their connectedness & weight (flows)
        pagerank_ = nx.pagerank(graph)
        
        # This is a potential measure of how critical a substation is for connecting the grid
        betweenness_centrality_ = nx.betweenness_centrality(graph)
    
        # Measure of how central a substation is in the grid
        degree_centrality_ = nx.degree_centrality(graph)
    
        # Measure of how central a substation is as a producer (uses a directed graph)
        #out_degree_centrality_ = nx.out_degree_centrality(graph)
    
        # Measure of how central a substation is as a consumer (uses a directed graph)
        #in_degree_centrality_ = nx.in_degree_centrality(graph)
    
        
        pagerank = []
        betweenness_centrality = []
        degree_centrality = []
        #out_degree_centrality = []
        #in_degree_centrality = []
    
        for k in sorted(pagerank_.keys()):
            pagerank.append(pagerank_[k])
            betweenness_centrality.append(betweenness_centrality_[k])
            degree_centrality.append(degree_centrality_[k])
            #out_degree_centrality.append(out_degree_centrality_[k])
            #in_degree_centrality.append(in_degree_centrality_[k])
    
        graph_features =  np.hstack((
            pagerank,
            betweenness_centrality,
            degree_centrality,
            #out_degree_centrality,
            #in_degree_centrality
        ))
        
        
        forcasted_obs = OBS_0.get_forecasted_inj()
        prod_p_f = forcasted_obs[0]
        #prod_v_f = forcasted_obs[1]
        load_p_f = forcasted_obs[2]
        load_q_f = forcasted_obs[3]
        
        numeric_features = np.hstack((prod_p_f, load_p_f, load_q_f))
        
        # all_observation = OBS_0.to_vect()
        prod_p = OBS_0.prod_p
        prod_q = OBS_0.prod_q
        #prod_v = OBS_0.prod_v
        load_p = OBS_0.load_p
        load_q = OBS_0.load_q
        #load_v = OBS_0.load_v
        rho_margin = 1 - OBS_0.rho
        
        numeric_features = np.hstack((numeric_features, prod_p, prod_q, load_p, load_q, rho_margin))
        
        topo_vect = OBS_0.topo_vect-1
        line_status = OBS_0.line_status*1
        
        selected_features = np.hstack((numeric_features, graph_features, topo_vect, line_status))
        
        return (selected_features - norm_min)/norm_range




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

        
        if min(observation.rho < 1): # seems 0.8 is the best
            this_action = self.action_space({})
        else:

            if (np.sum(observation.line_status)< np.size(observation.line_status)):
                line_to_recover = np.where(observation.line_status==0)
                for i_line in range(np.size(line_to_recover)):
                    # print(line_to_recover)
                    if (observation.time_before_cooldown_line[(line_to_recover[0][i_line])]) == 0:
                        line_action = self.action_space({"set_line_status": [(line_to_recover[0][i_line], 1)]})
                        obs_line, reward_simu_line, done_line, infos  = observation.simulate(line_action)
                        reward_simu_line = self.est_reward_update(obs_line, reward_simu_line, done_line)
                        if not done_line:
                            return line_action
            
            reward_simu_1, action = self.get_action(observation)
            action_asvector = actions_array[action,:]
            
            '''
            #if reward_simu_1<-0:
            if reward_simu_1<30:
                reward_simu_2, policy_chosen_backup = self.backup_agent.get_action(observation)
                if reward_simu_2>reward_simu_1:
                    action_asvector = actions_array_backup[policy_chosen_backup,:]
            '''
            this_action = self.action_space({})
            this_action.from_vect(action_asvector)
            
                
        #best_action = this_action
        
        '''
        # OPF is after topology decisions
        '''        

        return this_action

