import numpy as np
import gym
import random
from tqdm import tqdm

env = gym.make('FrozenLake-v0')

action_size = env.action_space.n # number of possible actions
state_size = env.observation_space.n # number of possible states

qtable = np.zeros((state_size, action_size)) # qtable[state][action] - q score for taking [action] in the given [state]

total_episodes = 15000
learning_rate = 0.8
max_steps = 99
gamma = 0.95

# exploration params
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

rewards = [] # for logging or wtv # YOU ONLY RECEIVE 1 if you reach the goal 0 otherwise https://gym.openai.com/envs/FrozenLake-v0/
for episode in tqdm(range(total_episodes)):
    # reset the env
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0 # for logging or wtv
    
    for step in range(max_steps):
        # chose an action in the current word state(s)
        # randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)
        # if this number > epsilon ----> exploration (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state])
        # else doing a random choice ---> exploration
        else:
            action = env.action_space.sample()

        # take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # update Q(S, a) + Lr[R(S, a) + gamma * max Q(s', a') - Q(s, a)]
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        total_rewards += reward # stupid in this case - you only get the reward when you reach the goal

        state = new_state

        # we died
        if done == True:
            break

        # reduce epsilon (decrease exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        rewards.append(total_rewards)

print('Score over time: {}'.format(sum(rewards) / total_episodes))
print(qtable)



# play the game using the formed qtable

env.reset()

for episode in range(10):
    state = env.reset()
    step = 0
    done = False
    print('======================================================')
    print('Episode: {}'.format(episode))

    for step in range(max_steps):
        action = np.argmax(qtable[state])
        new_state, reward, done, info = env.step(action)

        if done:
            env.render()
            print('Number of steps: {}'.format(step))
            break
        state = new_state
    env.close()