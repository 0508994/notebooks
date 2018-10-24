import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random
import time
from vizdoom import *
from skimage import transform
from collections import deque
from tqdm import tqdm

warnings.filterwarnings('ignore')
STACK_SIZE = 4

def create_environment():
    # load configuration
    game = DoomGame()
    game.load_config('basic.cfg')
    game.set_doom_scenario_path('basic.wad')
    game.init()
    # possible actionons
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    return game, possible_actions

def test_environment():
    game, actions = create_environment()

    episodes = 5
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            reward = game.make_action(action)
            print(reward, action)
            time.sleep(0.02)
        print('Result: {}'.format(game.get_total_reward()))
        time.sleep(2)
    game.close()


def preprocess_frame(frame):
    '''
        preprocess_frame:
        Take a frame.
        Resize it.
            __________________
            |                 |
            |                 |
            |                 |
            |                 |
            |_________________|
            
            to
            _____________
            |            |
            |            |
            |            |
            |____________|
        Normalize it.

        return preprocessed_frame
    '''
    # Greyscale frame already done in our vizdoom config
    # x = np.mean(frame, - 1)

    # Crop the screen (remove the roof becouse it contains no information)
    cropped_frame = frame[30:-10, 30:-30]
    # Normalize pixel values
    normalized_frame = cropped_frame / 255.0
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])

    return preprocessed_frame

def stack_frames(stacked_frames, state, is_new_episode, stack_size=STACK_SIZE):
    '''
        https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        Stacking frames is really important because it helps us to give have a sense of motion to our Neural Network.
            1) First we preprocess frame
            2) Then we append the frame to the deque that automatically removes the oldest frame
            3) Finally we build the stacked state
        This is how work stack:
            1) For the first frame, we feed 4 frames
            2) At each timestep, we add the new frame to deque and then we stack them to form a new stacked frame
            3) And so on stack
            4) If we're done, we create a new stack with 4 new frames (because we are in a new episode)

        ---M
         Alternativa ovome je motion-tracer: --> 
         https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/16_Reinforcement_Learning.ipynb
    '''
    frame = preprocess_frame(state)

    if is_new_episode:
        # clear the stack by copying the same frame 4 tmes
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for _ in range(stack_size)], maxlen=stack_size)
        for _ in range(stack_size):
            stacked_frames.append(frame)
    else:
        # append frame to deque, automatically removing the oldest frame
        stacked_frames.append(frame)
    
    stacked_state = np.stack(stacked_frames, axis=2)

    # stacked_state is 84x84x4 so it fits nn input_ (and it's a np.array now)
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.stack.html
    return stacked_state, stacked_frames


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name='inputs')
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name='actions_')
            self.target_Q = tf.placeholder(tf.float32, [None], name='target')

            #----> CONV LAYER 1
            # Input is 84x84x4
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_, \
                                          filters = 32,\
                                          kernel_size = [8, 8], \
                                          strides = [4, 4], \
                                          padding = 'VALID', \
                                          kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), \
                                          name = 'conv1')

            self.conv1_batchnorm = tf.layers.batch_normalization(\
                                          self.conv1, \
                                          training = True, \
                                          epsilon = 1e-5, \
                                          name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name='conv1_out') 
            # --> [20, 20, 32]

            #----> CONV LAYER 2
            # Input is conv1_out
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out, \
                                          filters = 64,\
                                          kernel_size = [4, 4], \
                                          strides = [2, 2], \
                                          padding = 'VALID', \
                                          kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), \
                                          name = 'conv2')

            self.conv2_batchnorm = tf.layers.batch_normalization(\
                                          self.conv2, \
                                          training = True, \
                                          epsilon = 1e-5, \
                                          name = 'batch_norm2')
            
            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name='conv2_out') 
            # --> [9, 9, 64]

            #----> CONV LAYER 3
            # Input is conv2_out
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out, \
                                          filters = 128,\
                                          kernel_size = [4, 4], \
                                          strides = [2, 2], \
                                          padding = 'VALID', \
                                          kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), \
                                          name = 'conv3')

            self.conv3_batchnorm = tf.layers.batch_normalization(\
                                          self.conv3, \
                                          training = True, \
                                          epsilon = 1e-5, \
                                          name = 'batch_norm3')
            
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name='conv3_out') 
            # --> [3, 3, 128]

            # Flatten layer -->
            # Input is conv3_out
            self.flatten = tf.layers.flatten(self.conv3_out)
            ## --> [1152]
            
            # Dense layer
            # Input is flattened conv3_out
            self.fc =  tf.layers.dense(inputs = self.flatten, \
                                       units = 512, \
                                       activation = tf.nn.elu, \
                                       kernel_initializer = tf.contrib.layers.xavier_initializer(), \
                                       name = 'fc1' )

            # Output layer [Q values for each action]
            self.output =  tf.layers.dense(\
                                       inputs = self.fc, \
                                       units = 3, \
                                       activation = None, \
                                       kernel_initializer = tf.contrib.layers.xavier_initializer())  

            # Our predicted Q-value
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            # >>> np.sum([1, 2, 3], axis=1)
            # array([6])
            # >>> np.sum([1, 2, 3], axis=0)
            # array([1, 2, 3])
            # >>> np.sum([1, 2, 3], axis=None) # default
            # 6

            # The loss is the difference between our predicted Q-values and Q_target
            # Sum(Q-Target - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

# https://github.com/udacity/deep-learning/blob/master/reinforcement/Q-learning-cart.ipynb
class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]

def instantiate_memory(max_size, game, pretrain_length, possible_actions, stacked_frames):
    '''
        Here we'll deal with the empty memory problem:
        we pre-populate our memory by taking random actions and storing the experience (state, action, reward, new_state).
    '''
    memory = Memory(max_size=max_size)
    game.new_episode()

    for i in range(pretrain_length):
        if i == 0:
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)

        # Random action
        action = random.choice(possible_actions)
        reward = game.make_action(action)
        done = game.is_episode_finished()

        if done:
            # Create a blank state
            next_state = np.zeros(state.shape)
            # Add exp to memory
            memory.add((state, action, reward, next_state, done))
            # Start a new episode
            game.new_episode()
            # Get a new state
            state = game.get_state().screen_buffer
            # Stack the frames
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        else:
            # Get the next state
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            # Add exp to memory
            memory.add((state, action, reward, next_state, done))
            # Set the next state
            state = next_state

    return memory

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions, dqn, sess):
    # This function will do the part with epsilon select a random action, or action = argmaxQ(st, a)
    # Epsilon greedy strategy
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if explore_probability > exp_exp_tradeoff:
        action = random.choice(possible_actions)
    else:
        # https://www.aiworkbox.com/lessons/use-feed_dict-to-feed-values-to-tensorflow-placeholders
        Qs = sess.run(dqn.output, feed_dict={dqn.inputs_:state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability
    


if __name__ == '__main__':
    stack_size = STACK_SIZE
    stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for _ in range(stack_size)], maxlen=stack_size)
    game, possible_actions = create_environment()

    # Model hyperparameters
    state_size = [84, 84, 4]                        # Input is a stack of 4 frames (84x84x4 - width, height, channels)
    action_size = game.get_available_buttons_size() # 3 possible actions: left, right, shoot
    learning_rate = 0.0002                          # Alpha aka learning rate

    # Training hyperparameters
    total_episodes = 501                           # Total training episodes
    max_steps = 100                                 # Maximum steps inside of each episode
    batch_size = 64

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0
    explore_stop = 0.01
    decay_rate = 0.0001

    # Q learning hyperparameters
    gamma = 0.95
    
    # Memory hyperparameters
    pretrain_length = batch_size                    # Number of experiences stored in the memory when initiated for the first time
    memory_size = 5000

    training = False
    episode_render = False

    tf.reset_default_graph()
    DQNetwork = DQNetwork(state_size, action_size, learning_rate)

    # writer = tf.summary.FileWriter('/tensorboard/dqn/1')
    # tf.summary.scalar('Loss', DQNetwork.loss)
    # write_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if training:
        memory = instantiate_memory(memory_size, game, pretrain_length, possible_actions, stacked_frames)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            decay_step = 0
            game.init()
        
            for episode in range(total_episodes): # tqdm(range(total_episodes)):
                step = 0
                episode_rewards = []
                game.new_episode()
                state = game.get_state().screen_buffer
                state, stacked_frames = stack_frames(stacked_frames, state, True)

                while step < max_steps:
                    step += 1
                    decay_step += 1
                    action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions, DQNetwork, sess)
                    reward = game.make_action(action)
                    done = game.is_episode_finished()
                    episode_rewards.append(reward)

                    if done:
                        # The episode ends so there is no next state
                        next_state = np.zeros((84, 84), dtype=np.int)
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                        
                        # Set steps to max to end the episode!
                        step = max_steps
                        
                        total_reward = np.sum(episode_rewards)
                        print('Episode: {}\tLoss: {:.4f}\tTotalReward: {}\tExplore P: {:.4f}'\
                            .format(episode, loss, total_reward, explore_probability))
                        
                        memory.add((state, action, reward, next_state, done))

                    else:
                        # Get the next state
                        next_state = game.get_state().screen_buffer
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                        # Add exp to mmemory
                        memory.add((state, action, reward, next_state, done))

                        # Update state
                        state = next_state

                    # LEARNING PART
                    batch = memory.sample(batch_size)
                    states_mb = np.array([each[0] for each in batch], ndmin=3)
                    actions_mb = np.array([each[1] for each in batch])
                    rewards_mb = np.array([each[2] for each in batch])
                    next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                    dones_mb = np.array([each[4] for each in batch])

                    target_Qs_batch = []

                    # Get Q values for next_state
                    Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_:next_states_mb})

                    # Set Q_target - r if te episode ends at s + 1 otherwhise Q_target = r + gamma*maxQ(s', a')

                    for i in range(len(batch)):
                        terminal = dones_mb[i]
                        if terminal:
                            # If we are at the terminal state only equals r
                            target_Qs_batch.append(rewards_mb[i])
                        else:
                            target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                            target_Qs_batch.append(target)

                    targets_mb = np.array([each for each in target_Qs_batch])


                    loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],\
                                        feed_dict = {DQNetwork.inputs_:states_mb,\
                                                    DQNetwork.target_Q: targets_mb,\
                                                    DQNetwork.actions_:actions_mb})

                    # summary = sess.run(write_op, feed_dict = {DQNetwork.inputs:states_mb,\
                    #                                           DQNetwork.target_Q:targets_mb,\
                    #                                           DQNetwork.actions_:actions_mb})
                    # writer.add_summary(summary, episode)
                    # writer.flush()
                        
                if episode % 50 == 0:
                    save_path = saver.save(sess, './models/deepq_doom_model.ckpt')
                    print('Model saved at {}.'.format(save_path))
        
                # \while
            # \for
        # \with
        game.close()
    # \if
    else:
        with tf.Session(config=config) as sess:
            #game, possible_actions = create_environment()
            #total_score = 0

            # Load the model
            saver.restore(sess, './models/deepq_doom_model.ckpt')
            for i in range(1):
                done = False
                game.new_episode()
                state = game.get_state().screen_buffer
                state, stacked_frames = stack_frames(stacked_frames, state, True)

                while not game.is_episode_finished():
                    # Take the biggest Q value (= the best action)
                    Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_:state.reshape((1, *state.shape))})

                    choice = np.argmax(Qs)
                    action = possible_actions[int(choice)]

                    game.make_action(action)
                    done = game.is_episode_finished()
                    score = game.get_total_reward()

                    if done:
                        break
                    else:
                        time.sleep(0.2)
                        print('Goin...')
                        next_state = game.get_state().screen_buffer
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                        state = next_state
                    
                score = game.get_total_reward()
                print('Score: {}'.format(score))
        game.close()
# \main           

        
