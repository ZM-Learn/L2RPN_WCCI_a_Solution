

import threading
import numpy as np

import tensorflow as tf
tf.disable_v2_behavior()

import time
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import heapq

import os
import grid2op
from grid2op.Runner import Runner
from grid2op import make
from grid2op.Reward import L2RPNReward


EPISODES_train = 10000
epsilon = 0.9
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class A3CAgent:
    def __init__(self, state_size, action_size, env_name):
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # get gym environment name
        self.env_name = env_name

        # these are hyper parameters for the A3C
        self.actor_lr = 0.001 # previously 0.0001
        self.critic_lr = 0.05 # previously 0.0001
        self.discount_factor = .7
        self.hidden1, self.hidden2, self.hidden3 = 1000, 1000, 1000
        self.threads = 2 # 48 or 16 or 32 - corresponds to parallel agents

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]
        
        tf.get_default_graph()

        self.sess = tf.InteractiveSession()
        # TF 1.x - sess = tf.InteractiveSession(); TF 2.X sess=tf.compat.v1.InteractiveSession()
        
        K.set_session(self.sess) # tensorflow 1.X
        #tf.compat.v1.keras.backend.set_session(self.sess) # tensorflow 2.X
        
        
        #tf.compat.v1.disable_eager_execution() # compatibility issues due to tf 2.0

        
        self.sess.run(tf.global_variables_initializer())  # tensorflow 1.X
        

        #self.sess.run(tf.compat.v1.global_variables_initializer())  # tensorflow 2.X

    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_model(self):
        state = Input(batch_shape=(None,  self.state_size))
        shared = Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(state)

        actor_hidden1 = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(shared)
        action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(actor_hidden1)

        value_hidden1 = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(shared)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(value_hidden1)

        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic
    # make loss function for Policy Gradient
    # [log(action probability) * advantages] will be input for the back prop
    # we add entropy of action probability to loss
    def actor_optimizer(self):
        action = K.placeholder(shape=(None, self.action_size))
        advantages = K.placeholder(shape=(None, ))

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages) # 1e-10 to 1e-8
        loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)  # 1e-10 to 1e-8

        actor_loss = loss + 0.01*entropy

        optimizer = Adam(lr=self.actor_lr)
        #updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        #train = K.function([self.actor.input, action, advantages], [], updates=updates)
        
        updates = optimizer.get_updates(params=self.actor.trainable_weights,loss=actor_loss)
        train = K.function([self.actor.input, action, advantages],[], updates=updates)
        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        

        # my debug
        #tempout = K.placeholder(shape=(None, ))

        #updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        #train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        
        updates = optimizer.get_updates(params=self.critic.trainable_weights , loss=loss)
        train = K.function([self.critic.input, discounted_reward],[], updates=updates)
        return train

    # make agents(local) and start training
    def train(self):
        
        print("Start training...")

        try:
            self.load_model('pypow_wcci_a3c')
            print("Loaded saved NN model parameters \n")
        except:
            print("No existing model is found or saved model sizes do not match - initializing random NN weights \n")
        agents = [Agent(i, self.actor, self.critic, self.optimizer, self.env_name, self.discount_factor,
                        self.action_size, self.state_size) for i in range(self.threads)]

        for agent in agents:
            agent.start()


        while (len(scores) < EPISODES_train ):
            time.sleep(300) # main thread saves the model every 200 sec
            print("Current score list: ",len(scores))
            if (len(scores)>1):
                self.save_model('pypow_wcci_a3c')
                print("saved NN model at episode", episode, "\n")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

# This is Agent(local) class for threading
class Agent(threading.Thread):
    def __init__(self, index, actor, critic, optimizer, env_name, discount_factor, action_size, state_size):
        threading.Thread.__init__(self)

        self.states = []
        self.rewards = []
        self.actions = []

        self.index = index
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.env_name = env_name
        self.discount_factor = discount_factor
        self.action_size = action_size
        self.state_size = state_size
        
        

    # Thread interactive with environment
    def run(self):
        global episode
        global episode_test
        #env = grid2op.make() # env_name not used

        episode = 0
        print("Running an agent...")
        env = grid2op.make("l2rpn_wcci_2020", reward_class=L2RPNReward, difficulty="competition")
        training_batch_size = 128

        #time_step_end = 10000
        
                
        while episode < EPISODES_train:
            #env.set_id(episode)
            env.reset()
            #env.fast_forward_chronics(500)
            state = env.reset()
            #state_obs = observation_space.array_to_observation(state)
            #state = self.useful_state(state)
            time_step_end = env.chronics_handler.max_timestep()-2
            print('time step', time_step_end)
            #all_observation = state.to_vect()
            time_hour = 0
            score = 0
            
            time_step = 0
            non_zero_actions = 0
            while True:
                action = self.get_action(env,state)
                # print("get action:", action)
                
                if min(state.rho < 0.8):# in program this is set 0.8
                    action = 0
                    action_asvector = actions_array[action,:]
                else:
                    action_asvector = actions_array[action,:]
                    #print(action)
                    #print('current action:', np.sum(action_asvector))
                
                this_action = env.action_space({})
                this_action.from_vect(action_asvector)
                
                next_state, reward, done, flag = env.step(this_action)
                
                #reward = 100 - reward/10
                
                '''
                else:
                    action = self.get_action(env,state.to_vect())
                    next_state, reward, done, flag = env.step(actions_array[action,:])
                '''
                    
                if done:
                    score += -120 # this is the penalty for grid failure.
                    self.memory( self.get_usable_observation(state), action, -120)
                    print("done at episode:", episode)
                    print(env.time_stamp)
                    print('power deficiency: ', np.sum(state.prod_p)-np.sum(state.load_p))
                else:
                    #state_obs = env(next_state)
                    state_obs = next_state
                    #time_hour = state_obs.date_day*10000 + state_obs.date_hour * 100+ state_obs.date_minute
                    time_hour = state_obs.hour_of_day + 24*state_obs.day
                    #current_lim_factor = 0.85
                    over_current = 50 * sum(((state_obs.rho -1 ) )[
                        state_obs.rho > 1]) # # penalizing lines close to the limit
                   
                    score += (reward-over_current)
                    self.memory( self.get_usable_observation(state), action, (reward - over_current))
                    #print('power deficiency: ', np.sum(state.prod_p)-np.sum(state.load_p))
                    
                non_zero_actions += 0 if action==0 else 1
                state = next_state if not done else np.zeros([1, state_size])
                time_step += 1
                
                if done or time_step > time_step_end:
                    if done:
                        print("----STOPPED Thread:", self.index, "/ train episode: ", episode, "/ average score : ", int(score/time_step),
                              "/ with final time:", time_step, "/ with final action", action,
                              "/ number of non-zero actions", non_zero_actions, "/ day_hour_min:", time_hour)

                    if time_step > time_step_end:
                        print("End Thread:", self.index, "/ train episode: ", episode, "/ average score : ", int(score/time_step),
                              "/ with final time:", time_step, "/ with final action", action,
                              "/ number of non-zero actions", non_zero_actions, "/ day_hour_min:", time_hour)
                    scores.append(score)
                    episode += 1
                    print('exploration probability this episode', epsilon)
                    print('time window length: ',env.chronics_handler.max_timestep())
                    if episode%10==0:
                        env.render()
                    
                    self.train_episode(score < 2000000) # max score = 80000
                    break
                
                
                if time_step % training_batch_size ==0:
                    print("Continue Thread:", self.index, "/ train episode: ", episode, "/ average score : ", int(score/time_step),
                          "/ with recent time:", time_step, "/ with recent action", action,"/ number of non-zero actions", non_zero_actions, "/ day_hour_min:", time_hour)
                    self.train_episode(score < 2000000) # max score = 80000
                    print('saving memories...')
                    with open('self.states.txt', 'w') as filehandle:
                        for listitem in self.states:
                            filehandle.write('%s\n' % listitem)
                            
                    with open('self.actions.txt', 'w') as filehandle:
                        for listitem in self.actions:
                            filehandle.write('%s\n' % listitem)

                    with open('self.rewards.txt', 'w') as filehandle:
                        for listitem in self.rewards:
                            filehandle.write('%s\n' % listitem)


    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.reshape(self.states[-1], (1, self.state_size)))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    # update policy network and value network every episode
    def train_episode(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)

        values = self.critic.predict(np.array(self.states))
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values
        
        print('Training critic with prediction difference:', np.mean(np.abs(advantages)))
        self.optimizer[0]([self.states, self.actions, advantages])
        self.optimizer[1]([self.states, discounted_rewards])
        
        self.states, self.actions, self.rewards = [], [], []


    def get_action(self, env, state):
        roll = np.random.uniform()
        global epsilon
        if roll < epsilon:
            num_random_explor = 15
            random_action_s = np.random.randint(self.action_size, size = num_random_explor)
            epsilon = np.max([0.1,epsilon*0.995])
            obs_0, reward_nonaction, done_0, _  = state.simulate(env.action_space({}))
            
            reward_nonaction = 0.01+self.est_reward_update(obs_0, reward_nonaction, done_0)
            
            action_asclass = [None]*num_random_explor
            reward_simu = [None]*num_random_explor
            for i in range(num_random_explor):
                action_asclass[i] = env.action_space({})
                action_asclass[i].from_vect(actions_array[random_action_s[i],:])
                obs_0, reward_simu[i], done_0, _  = state.simulate(action_asclass[i])
                reward_simu[i] = self.est_reward_update(obs_0, reward_simu[i], done_0)
            
            if np.max(reward_simu)<reward_nonaction:
                random_action = 0
            else:
                random_action = random_action_s[np.argmax([reward_simu])]
            
            return random_action # origin
        
        
        num_compared_action = 4
        policy_nn = self.actor.predict(np.reshape(self.get_usable_observation(state), [1, self.state_size]))[0] 
        #indx = map(policy_nn.index, heapq.nlargest(num_compared_action, policy_nn))
        policy_chosen_list = np.argsort(policy_nn)[-1: -num_compared_action-1: -1]
        policy_chosen_list[num_compared_action-1] = 0 # force no action inside decisions
        #policy_nn_subid_mask = policy_nn * (1 - actions_array.dot((state[-14:]>0).astype(int))) 
        # this masking prevents any illegal operation
        
        #print("policy_nn: ", policy_nn)
        #print("actions_array.dot: ", actions_array.dot((state[-0:]>0).astype(int)))
        
        #policy_nn_subid_mask = policy_nn * (1 - actions_array.dot((state[-0:]>0).astype(int))) 
        
        

        #policy_chosen_list = np.random.choice(self.action_size, num_compared_action, replace=True,
        #                                      p=policy_nn / sum(policy_nn))
        
        # sample 4 actions
        
        #print('probability:', policy_nn / sum(policy_nn) )
        #print('policy_chosen_list', policy_chosen_list)
        #policy_chosen_list = np.hstack((0, policy_chosen_list)) # adding no action option # comment this line as agent learns...
        
        #print("policy_chosen_list:", policy_chosen_list)
        
        #print("actions_array[policy_chosen_list[0],:] :", actions_array[policy_chosen_list[0],:])
        
        action_asclass = [None]*num_compared_action
        reward_simu = [None]*num_compared_action
        for i in range(num_compared_action):
            action_asclass[i] = env.action_space({})
            action_asclass[i].from_vect(actions_array[policy_chosen_list[i],:])
            state_temp = state
            obs_0, reward_simu[i], done_0, infos0  = state_temp.simulate(action_asclass[i])
            #reward_simu[i] = 100 - reward_simu[i]/10
            reward_simu[i] = self.est_reward_update(obs_0, reward_simu[i], done_0)
            if i == num_compared_action-1:
                reward_simu[i] += 0.1
                
        
        #policy_chosen = np.argsort(policy_nn)[0]
        
        
        #policy_nn = self.actor.predict(np.reshape(self.get_usable_observation(state), [1, self.state_size]))[0] 
        #policy_chosen = np.argsort(policy_nn)[0]
        #return policy_chosen

        #print("np.argmax([rw_0,rw_1,rw_2,rw_3]):",np.argmax([rw_0,rw_1,rw_2,rw_3]))
        return policy_chosen_list[np.argmax([reward_simu])] # origin
        #return predicted_action


    def est_reward_update(self,obs,rw,done): # penalizing overloaded lines
        #obs = observation_space.array_to_observation(obs) if not done else 0
        
        state_obs = obs
        rw_0 = rw - 50 * sum(((( state_obs.rho - 1) )[
                        state_obs.rho > 1])) if not done else -120
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




if __name__ == '__main__':
    
    state_size = 943+177  # all_observartion
    action_size = 987+21 # all actions
    generation_control_num = 22
    scores = []
    #actions_array
    loaded = np.load('actions_array.npz')
    actions_array = np.transpose(loaded['actions_array'])  # this has 157 actions

    env_name = "l2rpn_wcci_2020"
    global_agent = A3CAgent(state_size, action_size, env_name)
    global_agent.train()
