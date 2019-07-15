import numpy as np
import tensorflow as tf

class KMeans:
    def __init__(self, sess):
        self.sess = sess
        self.DIMENSIONS = 2
        self.CLUSTERS = 10
        self.TRAINING_STEPS = 1000
        self.TOLERANCE = 0

    def construct_model(self, data):
        self.in_tensor = in_tensor = tf.placeholder(tf.float32, [None, self.DIMENSIONS])
        # trainable variables: clusters means
        random_point_ids = tf.squeeze(tf.multinomial(tf.ones([1, tf.shape(in_tensor)[0]]), self.CLUSTERS))
        means = tf.Variable(tf.gather(in_tensor, random_point_ids), dtype=tf.float32)

        # E-step: recomputing cluster assignments according to the current means
        inputs_ex, means_ex = tf.expand_dims(in_tensor, 0), tf.expand_dims(means, 1)
        distances = tf.reduce_sum(tf.squared_difference(inputs_ex, means_ex), 2)
        self.assignments = assignments = tf.argmin(distances, 0)

        # M-step: relocating cluster means according to the computed assignments
        sums = tf.unsorted_segment_sum(in_tensor, assignments, self.CLUSTERS)
        counts = tf.reduce_sum(tf.one_hot(assignments, self.CLUSTERS), 0)
        self.means_ = means_ = tf.divide(sums, tf.expand_dims(counts, 1))
        self.distortion = distortion = tf.reduce_sum(tf.reduce_min(distances, 0))
        self.train_step = train_step = means.assign(means_)
        self.sess.run(tf.global_variables_initializer(), feed_dict={self.in_tensor: data})
        return self

    def train(self, data):
        prev_assignments = None
        for step in range(self.TRAINING_STEPS):
            c_distortion, c_means, c_assignments, _ = self.sess.run(
                [self.distortion, self.means_, self.assignments, self.train_step], feed_dict={self.in_tensor: data})
            if step > 0:
                # computing the number of re-assignments during the step
                re_assignments = (c_assignments != prev_assignments).sum()
                print("{0}:\tdistortion {1:.2f}\tre-assignments {2}".format(
                    step, c_distortion, re_assignments))
                if re_assignments <= self.TOLERANCE:
                    break
            prev_assignments = c_assignments
        self.c_distortion = c_distortion
        self.c_means = c_means
        self.c_assignments = c_assignments
        return self