import tensorflow as tf


class PGNetwork:
    def __init__(self, state_size, action_size, learning_rate, name="PGNetwork"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            with tf.name_scope("inputs"):
                self.inputs_ = tf.placeholder(
                    tf.float32, [None, *state_size], name="inputs_"
                )
                self.actions = tf.placeholder(
                    tf.int32, [None, action_size], name="actions"
                )
                self.discounted_episode_rewards_ = tf.placeholder(
                    tf.float32,
                    [
                        None,
                    ],
                    name="discounted_episode_rewards_",
                )
                # Tensorboard
                self.mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")

            ########################################################################################################################

            with tf.name_scope("conv1"):
                # Input is 84x84x4
                self.conv1 = tf.layers.conv2d(
                    inputs=self.inputs_,
                    filters=32,
                    kernel_size=[8, 8],
                    strides=[4, 4],
                    padding="VALID",
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    name="conv1",
                )

                self.conv1_batchnorm = tf.layers.batch_normalization(
                    self.conv1, training=True, epsilon=1e-5, name="batch_norm1"
                )

                self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
                # ## [20, 20, 32]

            ########################################################################################################################
            with tf.name_scope("conv2"):
                self.conv2 = tf.layers.conv2d(
                    inputs=self.conv1_out,
                    filters=64,
                    kernel_size=[4, 4],
                    strides=[2, 2],
                    padding="VALID",
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    name="conv2",
                )

                self.conv2_batchnorm = tf.layers.batch_normalization(
                    self.conv2, training=True, epsilon=1e-5, name="batch_norm2"
                )

                self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")

            ########################################################################################################################
            with tf.name_scope("conv3"):
                self.conv3 = tf.layers.conv2d(
                    inputs=self.conv2_out,
                    filters=128,
                    kernel_size=[4, 4],
                    strides=[2, 2],
                    padding="VALID",
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    name="conv3",
                )

                self.conv3_batchnorm = tf.layers.batch_normalization(
                    self.conv3, training=True, epsilon=1e-5, name="batch_norm3"
                )

                self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
                # --> [3, 3, 128]
            ########################################################################################################################
            with tf.name_scope("flatten"):
                self.flatten = tf.layers.flatten(self.conv3_out)
                # -> [1152]
            ########################################################################################################################
            with tf.name_scope("fc"):
                self.fc = tf.layers.dense(
                    inputs=self.flatten,
                    units=512,
                    activation=tf.nn.elu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name="fc",
                )

            ########################################################################################################################
            with tf.name_scope("logits"):
                self.logits = tf.layers.dense(
                    inputs=self.fc,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    units=self.action_size,
                    activation=None,
                )

            ########################################################################################################################
            with tf.name_scope("softmax"):
                self.action_distribution = tf.nn.softmax(self.logits)

            ########################################################################################################################
            with tf.name_scope("loss"):
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.logits, labels=self.actions
                )
                self.loss = tf.reduce_mean(
                    self.neg_log_prob * self.discounted_episode_rewards_
                )

            ########################################################################################################################
            with tf.name_scope("train"):
                self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(
                    self.loss
                )


if __name__ == "__main__":
    print("Testing for indent errors...")