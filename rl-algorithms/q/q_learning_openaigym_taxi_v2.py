#https://www.youtube.com/watch?v=q2ZOEFAaaI0&index=2&t=0s&list=LLHkR-ZpIbPoYkS3kmiYxkFw

import numpy as np
import gym
import random
from tqdm import tqdm

total_episodes = 50000
total_test_episodes = 100
max_steps = 99

learning_rate = 0.7
gamma = 0.68        # discounting rate

# exploration params
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

env = gym.make('Taxi-v2')

action_space_size = env.action_space.n # all possible actions that can be taken for this env
observation_space_size = env.observation_space.n # number of possible states

qtable = np.zeros((observation_space_size, action_space_size))


for episode in tqdm(range(total_episodes)):
    # reset the env
    state = env.reset()
    step = 0
    done = False
    
    for step in range(max_steps):
        # chose an action in the current world state
        exp_exp_tradeoff = random.uniform(0, 1)
        # if bigger than epsilon (exploration rate) select the action with the biggest Q for this tate
        if exp_exp_tradeoff > epsilon:
            # action is an index [6 possible represented by Q value]
            action = np.argmax(qtable[state, :]) # (state, action) => (int, int)
        else:
            action = env.action_space.sample()
            
        new_state, reward, done, info = env.step(action)
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma *
                                    np.max(qtable[new_state, :]) - qtable[state, action])
        
        state = new_state
        
        if done == True:
            break
        
    episode += 1
    
    #reduce epsilon like with simmulated annealing
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)


    env.reset()

# test the agent    
rewards = []

for episode in tqdm(range(total_test_episodes)):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        env.render()
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        total_rewards += reward
        
        if done:
            rewards.append(total_rewards)
            print ("Score", total_rewards)
            break
            
        state = new_state
        
env.close()
print ("Score over time: " +  str(sum(rewards) / total_test_episodes))

