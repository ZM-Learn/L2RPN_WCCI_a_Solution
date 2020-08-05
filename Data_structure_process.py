

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from grid2op.PlotGrid import PlotMatplot

import example_submissions.submission.BaseAgent as BaseAgent
#import example_submissions.submission.GreedyAgent as GreedyAgent

import os
import grid2op
from grid2op.Runner import Runner
from grid2op import make

import heapq
from random import sample as sample_action
from grid2op.Reward import L2RPNSandBoxScore

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


if __name__ == '__main__':
    
    total_episodes = 10
    all_test_reward_history = []
    
    env = grid2op.make("l2rpn_wcci_2020", reward_class=L2RPNSandBoxScore, difficulty="competition")

    action_space = env.action_space
    observation_space = env.observation_space
    
    
    ###################### observation space #################
    obs_as_object, reward, done, info = env.step(env.action_space({})) # use no action initialize
    #obs_as_vect = obs_as_object.to_vect()
    #observation_vector = observation_space.to_vect()
    obs = obs_as_object
    
    #agent = GreedyAgent.MyGreedyAgent
    #agent.__init__(agent,action_space)
    
    #  The shapes of all the components of the actio
    #  array([20, 56, 56, 20,  5])
    prod_p = obs.prod_p /100
    prod_q = obs.prod_q /100
    prod_v = obs.prod_v /100
    load_p = obs.load_p /100
    load_q = obs.load_q /100
    load_v = obs.load_v /100 # voltage setpoint of the loads
    rho = obs.rho
    topology_information = obs.connectivity_matrix()
    
    line_status = obs.line_status
    timestep_overflow = obs.timestep_overflow
    time_before_cooldown_line = obs.time_before_cooldown_line
    time_before_cooldown_sub = obs.time_before_cooldown_sub
    time_next_maintenance = obs.time_next_maintenance
    duration_next_maintenance = obs.duration_next_maintenance
    target_dispatch = obs.target_dispatch
    actual_dispatch = obs.actual_dispatch
    

    #obs.get_forecasted_inj(1)  #This function allows you to retrieve directly the “planned” injections for the timestep time_step
    
    all_observation = obs.to_vect()
    obs_numerical = all_observation[6:713]
    obs_numerical[649:706] = obs_numerical[649:706]*100
    
    ################# action space ######################

    shape_action = action_space.shape
    no_action = env.action_space({})
    no_action_vector = no_action.to_vect()
    
    topolo = action_space.get_all_unitary_topologies_set(env.action_space)

    lineset = action_space.get_all_unitary_line_set(env.action_space)
    
    actions_array = no_action_vector
    
    
    #env.set_id(1) # 2012-02-23
    env.set_id(2) # 2012-01-23 
    #env.set_id(3) # 2012-06-08
    #env.set_id(4) # 2012-11-07
    #env.set_id(5) # 2012-10-02
    #env.set_id(6) # 2012-12-10
    
    env.reset() 
    print(env.time_stamp)
    next_state, reward_nonaction, done, flag = env.step(no_action) # simulating which bus to connect performs better results
    reward_nonaction = 50 - reward_nonaction/10
    # sub_id_set = [1, 4, 5, 7, 12, 13, 16, 18, 23, 26, 28, 29, 31, 32, 33, 34] 
    # In our work, reconfiguration of substations 1, 4, 5, 7, 12, 13, 16, 18, 23, 26, 28, 29, 31, 32, 33, 34 is considered
    # Size = 66525, still impossible
    
    #sub_id_set = [4, 12, 16, 18, 23, 26, 31, 33]
    sub_id_set=[16]
    #sub_id_set = [1, 4, 5, 7, 8, 9, 12, 13, 14, 18, 21,22, 23, 26, 27, 28, 29, 31, 32, 33, 34,35] # 16 excluded
    #exclude_topology = ['load','generator']
    exclude_topology = []
    
    # set substation status
    num_reduced_action = 0
    record_recording = []
    
    record_recording.append(reward_nonaction)
    for i, action in enumerate(topolo):
        dict_topolo = topolo[i].as_dict()
        # if int(dict_topolo.get("set_bus_vect").get("modif_subs_id")[0]) in sub_id_set: # for set 
        if int(dict_topolo.get("set_bus_vect").get("modif_subs_id")[0]) in sub_id_set:
            
            id_this_sub = dict_topolo.get("set_bus_vect").get("modif_subs_id")[0]
            flag_exclude = 0
            for (action_key, action_type) in dict_topolo.get("set_bus_vect").get(id_this_sub).items():
                if action_type.get('type') in exclude_topology:
                    flag_exclude = 1
                    
            if flag_exclude:
                continue # skip this action
            else:
                env.reset()
                next_state, reward, done, flag = env.step(topolo[i]) # simulating which bus to connect performs better results
                reward = 50 - reward/10
                if reward > 0: # if reward > reward_nonaction:
                    added_select_action = topolo[i].to_vect()
                    actions_array = np.column_stack((actions_array, added_select_action))
                    num_reduced_action += 1
                    record_recording.append(reward)
                    print(i)
                    print(id_this_sub)
   
            
    '''
    # change substation status
    topolo = action_space.get_all_unitary_topologies_change(env.action_space)
    
    num_reduced_action = 0
    record_recording = []
    for i, action in enumerate(topolo):
        dict_topolo = topolo[i].as_dict()
        if dict_topolo: # non-empty
            if int(dict_topolo.get("change_bus_vect").get("modif_subs_id")[0]) in sub_id_set:
                id_this_sub = dict_topolo.get("change_bus_vect").get("modif_subs_id")[0]
                flag_exclude = 0
                for (action_key, action_type) in dict_topolo.get("change_bus_vect").get(id_this_sub).items():
                    if action_type.get('type') in exclude_topology:
                        flag_exclude = 1
                    
            if flag_exclude:
                continue # skip this action
            else:
                env.reset()
                next_state, reward, done, flag = env.step(topolo[i]) # simulating which bus to connect performs better results
                if reward > reward_nonaction*0.95:
                    added_select_action = topolo[i].to_vect()
                    actions_array = np.column_stack((actions_array,added_select_action))
                    num_reduced_action += 1
                    record_recording.append(reward)
                    print(i)
                    print(id_this_sub)
    
    '''
    ################ Store the actions #################
    

    
    max_num_index_list = map(record_recording.index, heapq.nlargest(200, record_recording))
    idx_best = np.asarray(list(max_num_index_list))
    best_selected_a = actions_array[ : , idx_best]
    
    
    max_num_index_list2 = map(record_recording.index, heapq.nlargest(10000, record_recording))
    index_maximumreward = list(max_num_index_list2)
    idxs_of_idx = np.random.randint(0, len(index_maximumreward), size=100)
    idx_random = []
    for i in range(len(idxs_of_idx)):
        idx_random.append(index_maximumreward[idxs_of_idx[i]])
    
    idx_random = np.asarray(idx_random)
    
    random_selected_a = actions_array[ : , idx_random]
    
    uniques_index.append(0)
    uniques_index = np.unique(index_maximumreward)
    
    uniques_action = np.unique(actions_array,axis=1)
    
    
    selected_actions = no_action_vector
    selected_actions = np.column_stack((selected_actions, best_selected_a))
    selected_actions = np.column_stack((selected_actions, random_selected_a))
    
    
    ################ Add lines setting to action array #################
    
    for i, action in enumerate(lineset):
        dict_lineset = lineset[i].as_dict()
        # if int(dict_topolo.get("set_bus_vect").get("modif_subs_id")[0]) in sub_id_set: # for set 
        
        added_select_action = lineset[i].to_vect()
        selected_actions = np.column_stack((selected_actions, added_select_action))

    

    
    
    
    # 1 st NN: determine the topology
    # 2 nd OPF: determine redispatch decision
    # Add generation redispatch: t
    
    
    #remaining_action_settings = int(actions_array.size/len(no_action_vector))
    
    #generators_id = action_space.get_generators_id(env.action_space)
    #actiongen = env.action_space({"redispatch": [(0, 1),(1, 3),(3, 4)]})
    
    
    
    actions_array = selected_actions
    np.savez_compressed('actions_array.npz', actions_array=actions_array)
    
    #np.savez_compressed('valid_actions_array_uniq.npz', valid_actions_array_uniq=valid_actions_array_uniq)
    
    #
    
    no_action = env.action_space({})
    no_action_vector = no_action.to_vect()
    redispatch_array = no_action_vector.copy()
    action_asvector = no_action_vector.copy()
    #redispatch actions
    action_asvector[472] = 1.3 # ramp 1.3, maximum  50
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[474] = 1.3 # ramp 1.3, maximum  50
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[475] = 2.7 # ramp 2.7, maximum  250
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[476] = 1.3 # ramp 1.3, maximum  50
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[482] = 2.7 # 2.7 ,100
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[485] = 2.7 # 2.7 ,100
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[488] = 4.2 # 4.2 ,150
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[491] = 2.7 # 2.7 ,400
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[492] = 8.4 # 8.4 ,300
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[493] = 9.8 # 9.8, 350
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[472] = -1.3 # ramp 1.3, maximum  50
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[474] = -1.3 # ramp 1.3, maximum  50
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[475] = -2.7 # ramp 2.7, maximum  250
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[476] = -1.3 # ramp 1.3, maximum  50
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[482] = -2.7 # 2.7 ,100
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[485] = -2.7 # 2.7 ,100
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[488] = -4.2 # 4.2 ,150
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[491] = -2.7 # 2.7 ,400
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[492] = -8.4 # 8.4 ,300
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    action_asvector[493] = -9.8 # 9.8, 350
    redispatch_array = np.column_stack((redispatch_array, action_asvector))
    action_asvector = no_action_vector.copy()
    
    np.savez_compressed('redispatch_array.npz', redispatch_array=redispatch_array)
