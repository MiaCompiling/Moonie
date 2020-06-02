#!/usr/bin/env python
from __future__ import print_function
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

import argparse
# import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import tensorflow.compat.v1 as tf

# tf.compat.v1.disable_resource_variables()
tf.disable_eager_execution()
try:
    import hfo
except ImportError:
    print('Failed to import hfo. To install hfo, in the HFO directory, run: \"pip install .\"')
    exit()

tf.reset_default_graph()

# ___________________________Actor Net____________________________


class Actor(object):
    def __init__(self, sess, a_dim, p_dim, learning_rate, tau):
        self.sess = sess
        self.a_dim = a_dim
        # self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_counter = 0
        self.tau = tau
        self.params_dim = p_dim

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a, self.params = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_, self.params_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        # print("actor evaluate parameters: ", len(self.e_params))  # len = 12
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        self.soft_replace = [tf.assign(t, (1 - self.tau) * t + self.tau * e)
                             for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            init_weights = tf.random_normal_initializer(0, 0.01)
            init_bias = tf.constant_initializer(0.1)

            net_1 = tf.layers.dense(s, 1024, kernel_initializer=init_weights, bias_initializer=init_bias,
                                    name='l1', trainable=trainable)
            net_1 = tf.nn.leaky_relu(net_1, alpha=0.01)
            net_2 = tf.layers.dense(net_1, 512, kernel_initializer=init_weights, bias_initializer=init_bias,
                                    name='l2', trainable=trainable)
            net_2 = tf.nn.leaky_relu(net_2, alpha=0.01)
            net_3 = tf.layers.dense(net_2, 256, kernel_initializer=init_weights, bias_initializer=init_bias,
                                    name='l3', trainable=trainable)
            net_3 = tf.nn.leaky_relu(net_3, alpha=0.01)
            net_4 = tf.layers.dense(net_3, 128, kernel_initializer=init_weights, bias_initializer=init_bias,
                                    name='l4', trainable=trainable)
            net_4 = tf.nn.leaky_relu(net_4, alpha=0.01)

            act_value = tf.layers.dense(net_4, self.a_dim, kernel_initializer=init_weights,
                                           bias_initializer=init_bias,
                                           name='action', trainable=trainable)
            # action_value = tf.keras.layers.Dense(self.a_dim, kernel_initializer=init_weights,
            #                                      bias_initializer=init_bias,
            #                                      name='action', trainable=trainable)(
            #     net_4)
            params = tf.layers.dense(net_4, self.params_dim, kernel_initializer=init_weights,
                                     bias_initializer=init_bias,
                                     name='params', trainable=trainable)
            # params = tf.keras.layers.Dense(self.params_dim, kernel_initializer=init_weights, bias_initializer=init_bias,
            #                                name='params', trainable=trainable)(net_4)

        return act_value, params

    def learn(self, s):  # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        self.sess.run(self.soft_replace)

    def choose_action(self, s, epsi):
        if epsi == 0 or np.random.uniform() > epsilon:  # use epsilon-greedy
            # print("~~~~~~~~~~~~~~~~~~~~~~~choose max action~~~~~~~~~~~~~~~~~~~~~~~~~~")
            s = s[np.newaxis, :]
            a_value, a_params = self.sess.run((self.a, self.params), feed_dict={S: s})  # this is questionable
            act = np.argmax(a_value, 1)[0]
        else:
            act = np.random.choice([0, 1, 2])  # 用np. 还是tf.
            s = s[np.newaxis, :]                                   # important!!!!!
            a_value = sess.run(self.a, feed_dict={S: s})
            a_params = [np.random.uniform(0, 100), np.random.uniform(-180, 180),
                        np.random.uniform(-180, 180),
                        np.random.uniform(-180, 180),
                        np.random.uniform(0, 100), np.random.uniform(-180, 180)]
        a_value = np.squeeze(a_value)
        a_params = np.squeeze(a_params)
        # print("6 parameters: ", a_params)
        return act, a_value, a_params  # the action index

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy
            # xs = policy's parameters
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            # print("ys: ", tf.concat([self.a, self.params], axis=1)) shape -------- [None, 10]
            # print('e_params: ', self.e_params)
            self.policy_grads = tf.gradients(ys=tf.concat([self.a, self.params], axis=1), xs=self.e_params, grad_ys=a_grads)
            # print("policy_grads: ", self.policy_grads)
            # print(len(self.policy_grads)) ------ 12

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


# '___________________Critic Net_____________________'


class Critic(object):
    def __init__(self, sess, s_dim, a_dim, learning_rate, gamma, a, a_, tau):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim  # 4+6 = 10, a contains action_value and parameters
        self.lr = learning_rate
        self.gamma = gamma
        self.tau = tau

        with tf.variable_scope('Critic', reuse=tf.AUTO_REUSE):
            # Input (s, a), output q
            self.a = tf.stop_gradient(a)  # stop critic update flows to actor
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net',
                                      trainable=False)  # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

            with tf.variable_scope('target_q'):
                self.target_q = R + self.gamma * self.q_

            with tf.variable_scope('TD_error'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

            with tf.variable_scope('C_train'):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            with tf.variable_scope('a_grad'):
                self.a_grad = tf.gradients(self.q, self.a)
                # print('a_grad: ', self.a_grad)  # shape = [?, 10]

            self.soft_replacement = [tf.assign(t, (1 - self.tau) * t + self.tau * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            init_weights = tf.random_normal_initializer(0., 0.01)
            init_bias = tf.constant_initializer(0.1)

            n_l1 = 1024
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_weights, trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_weights, trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=init_bias, trainable=trainable)
            net_1 = tf.nn.leaky_relu((tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1), alpha=0.01)

            net_2 = tf.layers.dense(net_1, 512, kernel_initializer=init_weights, bias_initializer=init_bias,
                                    name='l2', trainable=trainable)
            net_2 = tf.nn.leaky_relu(net_2, alpha=0.01)
            net_3 = tf.layers.dense(net_2, 256, kernel_initializer=init_weights, bias_initializer=init_bias,
                                    name='l3', trainable=trainable)
            net_3 = tf.nn.leaky_relu(net_3, alpha=0.01)
            net_4 = tf.layers.dense(net_3, 128, kernel_initializer=init_weights, bias_initializer=init_bias,
                                    name='l4', trainable=trainable)
            net_4 = tf.nn.leaky_relu(net_4, alpha=0.01)

        with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
            q = tf.layers.dense(net_4, 1, kernel_initializer=init_weights, bias_initializer=init_bias,
                                trainable=trainable)

        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        self.sess.run(self.soft_replacement)


# '_______________Memory______________'


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, p, r, s_):
        transition = np.hstack((s, a, p, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


# "____________________Reward_____________________"


# get the distance between the ball and the goal
def get_ball_dist_goal(sta):
    ball_proximity = sta[53]
    goal_proximity = sta[15]
    ball_dist = 1.0 - ball_proximity
    goal_dist = 1.0 - goal_proximity
    ball_ang_sin_rad = sta[51]
    ball_ang_cos_rad = sta[52]
    ball_ang_rad = math.acos(ball_ang_cos_rad)
    if ball_ang_sin_rad < 0:
        ball_ang_rad *= -1.
    goal_ang_sin_rad = sta[13]
    goal_ang_cos_rad = sta[14]
    goal_ang_rad = math.acos(goal_ang_cos_rad)
    if goal_ang_sin_rad < 0:
        goal_ang_rad *= -1.
    alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad, goal_ang_rad)
    ball_dist_goal = math.sqrt(
        ball_dist * ball_dist + goal_dist * goal_dist - 2. * ball_dist * goal_dist * math.cos(alpha))
    return ball_dist_goal


# there's explicit codes about reward in src/hfo_game.cpp
def getReward(old_state, current_state, get_kickable_reward, status):
    r = 0
    kickable = current_state[12]
    old_kickable = old_state[12]

    # NOTE: the closer agent gets towards the ball, the bigger this state[53] is
    ball_prox_delta = current_state[53] - old_state[53]  # ball_proximity - old_ball_prox
    kickable_delta = kickable - old_kickable
    ball_dist_goal_delta = get_ball_dist_goal(current_state) - get_ball_dist_goal(old_state)
    player_on_ball = hfo_env.playerOnBall()
    our_unum = hfo_env.getUnum()

    # move to ball reward
    if player_on_ball.unum < 0 or player_on_ball.unum == our_unum:
        r += ball_prox_delta
    # if kickable_delta >= 1 and (not get_kickable_reward):
    if kickable_delta >= 1:
        r += 1.0
        get_kickable_reward = True

    # kick to goal reward
    if player_on_ball.unum == our_unum:
        r -= 3 * ball_dist_goal_delta
    # elif get_kickable_reward:  # we have passed to teammate
    #     r -= 3 * 0.2 * ball_dist_goal_delta

    # EOT reward
    if status == hfo.GOAL:
        if player_on_ball.unum == our_unum:
            r += 5
        else:
            r += 1
    elif status == hfo.CAPTURED_BY_DEFENSE:
        r += 0

    return r, get_kickable_reward


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=6000,
                    help="Server port")
parser.add_argument('--seed', type=int, default=None,
                    help="Python randomization seed; uses python default if 0 or not given")
parser.add_argument('--no-reorient', action='store_true',
                    help="Do not use the new Reorient action")
parser.add_argument('--record', action='store_true',
                    help="Doing HFO --record")
parser.add_argument('--rdir', type=str, default='log/',
                    help="Set directory to use if doing HFO --record")
parser.add_argument('--MEM_CAPACITY', type=int, default=2000)
parser.add_argument('--MAX_EPISODES', type=int, default=2000)
parser.add_argument('--LR_A', type=int, default=0.001)
parser.add_argument('--LR_C', type=int, default=0.001)
parser.add_argument('--GAMMA', type=int, default=0.9)  # reward discount
parser.add_argument('--BATCH_SIZE', type=int, default=32)
parser.add_argument('--tau', type=int, default=0.001)  # I changed from 0.0001 to 0.001
args = parser.parse_args()
if args.seed:
    random.seed(args.seed)
# Create the HFO Environment
hfo_env = hfo.HFOEnvironment()
# Connect to the server with the specified
# feature set. See feature sets in hfo.py/hfo.hpp.
if args.record:
    hfo_env.connectToServer(hfo.LOW_LEVEL_FEATURE_SET,
                            'bin/teams/base/config/formations-dt', args.port,
                            'localhost', 'base_left', False,
                            record_dir=args.rdir)
else:
    hfo_env.connectToServer(hfo.LOW_LEVEL_FEATURE_SET,
                            'bin/teams/base/config/formations-dt', args.port,
                            'localhost', 'base_left', False)
if args.seed:
    print("Python randomization seed: {0:d}".format(args.seed))

state_dim = hfo_env.getStateSize()  # 59
action_dim = 3
param_dim = 5

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

sess = tf.Session()

# Create actor and critic.
# They are actually connected to each other
actor = Actor(sess, action_dim, param_dim, args.LR_A, args.tau)
# print("actor_a", actor.a) --------  shape = [None, 4]
# print("actor_params", actor.params) ----------- shape = [None, 6]
critic = Critic(sess, state_dim, (action_dim + param_dim), args.LR_C, args.GAMMA,
                tf.concat([actor.a, actor.params], axis=1), tf.concat([actor.a_, actor.params_], axis=1), args.tau)
# print("~~~~~~~~a_grad: ", critic.a_grad)  shape --------- [None, 10]
actor.add_grad_to_graph(critic.a_grad)

sess.run(tf.global_variables_initializer())

M = Memory(args.MEM_CAPACITY, dims=2 * state_dim + (action_dim + param_dim) + 1)

epsilon = 1

x_list = []
reward_list = []

t1 = time.time()
for i in range(args.MAX_EPISODES):
    ep_reward = 0
    count = 0
    status = hfo.IN_GAME
    if epsilon > 0.1:
        epsilon *= 0.995

    while status == hfo.IN_GAME:
        count += 1
        state = hfo_env.getState()
        # print("state shape", state.shape) ------ (59,)
        action_index, action_value, action_params = actor.choose_action(state, epsilon)
        # print("the action chosen: ", action_index)
        if action_index == 0:
            hfo_env.act(hfo.DASH, action_params[0], action_params[1])
        elif action_index == 1:
            hfo_env.act(hfo.TURN, action_params[2])
        elif action_index == 2:
            hfo_env.act(hfo.KICK, action_params[4], action_params[5])

        status = hfo_env.step()
        state_ = hfo_env.getState()
        got_kickable_r = False
        reward, got_kickable_r = getReward(state, state_, got_kickable_r, status)

        # print('reward: ', reward)
        # print('state: ', state)
        # print("action_value: ", action_value)
        # print("action_params: ", action_params)
        M.store_transition(state, action_value, action_params, reward, state_)

        if M.pointer > args.MEM_CAPACITY:
            if M.pointer == args.MEM_CAPACITY + 1:
                print("####################################reach learn###################################")
            b_M = M.sample(args.BATCH_SIZE)
            b_s = b_M[:, :state_dim]
            b_a = b_M[:, state_dim: state_dim + action_dim + param_dim]  # action_value along with 6 parameters
            b_r = b_M[:, -state_dim - 1: -state_dim]
            b_s_ = b_M[:, -state_dim:]

            critic.learn(b_s, b_a, b_r, b_s_)
            actor.learn(b_s)

        state = state_
        ep_reward += reward

    x_list.append(i)
    reward_list.append(ep_reward)
    plt.figure("reward figure")
    plt.clf()
    plt.title('reward figure')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.scatter(np.array(x_list), np.array(reward_list))
    plt.pause(0.001)

    print("Episode: ", i, "episode reward: ", ep_reward, 'explore: ', epsilon)

    if status == hfo.SERVER_DOWN:
        hfo_env.act(hfo.QUIT)
        exit()


