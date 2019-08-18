import tensorflow as tf         
import numpy as np              
from vizdoom import *           
import random                   
import time                     
from skimage import transform   

from collections import deque
import matplotlib.pyplot as plt 

import warnings 
warnings.filterwarnings("ignore")

from pgnet import PGNetwork

IMGW = 84
IMGH = 84
STACK_SIZE = 4
STATE_SIZE = (IMGH, IMGW, STACK_SIZE)
ACTION_SIZE = 3
LEARNING_RATE = 0.002
NUM_EPOCHS = 201
BATCH_SIZE = 2500
GAMMA = 0.95
TRAINING = True

def create_env():
    """
        Creates doom health gathering env.
        params: none
        returs: env, all possible actions
    """
    game = DoomGame()   
    game.load_config("health_gathering.cfg")
    game.set_doom_scenario_path("health_gathering.wad")
    game.init()

    return game, np.identity(3, dtype=int).tolist()


def preprocess_frame(frame):
    """
        Preprocesses the input frame using IMGH and IMGW
        Image is first cropped removing the useless information
        then normalised and resized in the end
        params: image
        returns: preprocessed image
    """
    
    cropped_frame = frame[80:, :] # crop [UP:DOWN, LEFT:RIGHT]
    normalized_frame = cropped_frame / 255.0

    return transform.resize(normalized_frame, [IMGH, IMGW])

def stack_frames(stacked_frames, state, is_new_episode):
    """
        Creates a deque of stacked frames and a stacked_state
        to be used as input to the neural net
        params: deque of stacked frakes
                state (frame)
                indicator that tells if this state represents the start of the new episode
        returns: updated frame stack
                 stacked state to be used as input of the neural net 
    """
    
    frame = preprocess_frame(state)
    if is_new_episode:
        # reset the stack
        stacked_frames = deque([np.zeros((IMGH, IMGW), dtype=np.int) for _ in range(STACK_SIZE)], maxlen=STACK_SIZE)

        # stack the same frame STACK_SIZE times
        for _ in range(STACK_SIZE):
            stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        # append the frame and remove the oldest entry
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


def discount_and_normalize_rewards(episode_rewards, gamma=GAMMA):
    """
        Discounts the episode reward using some strange transformation ---
        params: numpy array of rewards the agent has received in this episode
                discount factor
        returns: numpy array of discounted rewards
    """

    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0  
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)

    return (discounted_episode_rewards - mean) / std


if __name__ == "__main__":
    stacked_frames = deque([np.zeros((IMGH, IMGW), dtype=np.int) for _ in range(STACK_SIZE)], maxlen=STACK_SIZE)
    game, possible_actions = create_env()
   
    tf.reset_default_graph()

    network = PGNetwork(STATE_SIZE, ACTION_SIZE, LEARNING_RATE)

    writer = tf.summary.FileWriter("./tensorboard/pg/test")
    tf.summary.scalar("Loss", network.loss)
    tf.summary.scalar("Reward_mean", network.mean_reward_ )
    write_op = tf.summary.merge_all()

    def make_batch(batch_size):
        global stacked_frames

        states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []

        episode_num = 1
        game.new_episode()

        # Get a new state
        frame = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, frame, True)

        while True:
            action_probability_distribution = sess.run(network.action_distribution,\
                                                       feed_dict={network.inputs_:state.reshape(1, *STATE_SIZE)})
        
            action = np.random.choice(range(action_probability_distribution.shape[1]),\
                                            p=action_probability_distribution[0])

            action = possible_actions[action]
            reward = game.make_action(action)
            done = game.is_episode_finished()

            states.append(state)
            actions.append(action)
            rewards_of_episode.append(reward)

            if done:
                next_frame = np.zeros((IMGH, IMGW), dtype=int)
                next_state, stacked_frames = stack_frames(stacked_frames, next_frame, False)

                rewards_of_batch.append(rewards_of_episode)
                discounted_rewards.append(discount_and_normalize_rewards(rewards_of_episode))

                if len(np.concatenate(rewards_of_batch)) > batch_size:
                    break

                rewards_of_episode = []
                episode_num += 1
                game.new_episode()
                
                frame = game.get_state().screen_buffer
                state, stacked_frames = stack_frames(stacked_frames, frame, True)

            else:
                next_frame = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_frame, False)
                state = next_state


        return np.stack(np.array(states)),\
               np.stack(np.array(actions)),\
               np.concatenate(rewards_of_batch),\
               np.concatenate(discounted_rewards),\
               episode_num 

    all_rewards = []
    total_rewards = 0
    maximum_reward_recorded = 0
    epoch = 1
    mean_reward_total = []
    average_reward = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        try:
            if TRAINING:
                init = tf.global_variables_initializer()
                sess.run(init)
                
                # Load the model
                saver.restore(sess, "./models/model.ckpt")
                print("Model restored.")

                while epoch < NUM_EPOCHS + 1:
                    # Gather data
                    states_mb, actions_mb, rewards_of_batch, discounted_rewards_mb, nb_episodes_mb =\
                        make_batch(BATCH_SIZE)

                    total_reward_of_that_batch = np.sum(rewards_of_batch)
                    all_rewards.append(total_reward_of_that_batch)

                    mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, nb_episodes_mb)
                    mean_reward_total.append(mean_reward_of_that_batch)

                    average_reward_of_all_training = np.divide(np.sum(mean_reward_total), epoch)
                    maximum_reward_recorded = np.amax(all_rewards)


                    print("==========================================")
                    print("Epoch: ", epoch, "/", NUM_EPOCHS)
                    print("-----------")
                    print("Number of training episodes: {}".format(nb_episodes_mb))
                    print("Total reward: {}".format(total_reward_of_that_batch))
                    print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
                    print("Average Reward of all training: {}".format(average_reward_of_all_training))
                    print("Max reward for a batch so far: {}".format(maximum_reward_recorded))

                    loss_, _ = sess.run([network.loss, network.train_opt],
                                        feed_dict={network.inputs_: states_mb.reshape((len(states_mb), IMGH ,IMGW, STACK_SIZE)),\
                                                    network.actions: actions_mb,\
                                                    network.discounted_episode_rewards_: discounted_rewards_mb })
                                                                                

                    print("Training Loss: {}".format(loss_))

                    summary = sess.run(write_op,\
                                        feed_dict={network.inputs_: states_mb.reshape((len(states_mb), IMGH, IMGW, STACK_SIZE)),\
                                                    network.actions: actions_mb,\
                                                    network.discounted_episode_rewards_: discounted_rewards_mb, \
                                                    network.mean_reward_: mean_reward_of_that_batch })
                    
                    writer.add_summary(summary, epoch)
                    writer.flush()

                    if epoch % 10 == 0:
                        saver.save(sess, "./models/model.ckpt")
                        print("Model saved")
                    epoch += 1
            else:
                
                saver.restore(sess, "./models/model.ckpt")

                for i in range(10):
                    game.new_episode()
                    frame = game.get_state().screen_buffer
                    state, stacked_frames = stack_frames(stacked_frames, frame, True)

                    while not game.is_episode_finished():
                    
                        action_probability_distribution = sess.run(network.action_distribution, \
                                                                    feed_dict={network.inputs_: state.reshape(1, *STATE_SIZE)})


                        action = np.random.choice(range(action_probability_distribution.shape[1]),\
                                                        p=action_probability_distribution[0]) 
                                                 
                        action = possible_actions[action]
                        reward = game.make_action(action)
                        done = game.is_episode_finished()
                        
                        if done:
                            break
                        else:
                            next_frame = game.get_state().screen_buffer
                            next_state, stacked_frames = stack_frames(stacked_frames, next_frame, False)
                            state = next_state
        

                    print("Score for episode ", i, " :", game.get_total_reward())
                game.close()

        except KeyboardInterrupt:
            print("Manual interrupt has occurred.")
