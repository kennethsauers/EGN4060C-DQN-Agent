import random
import gym
import time
import numpy as np
from collections import deque

from dataLogger import Logger
from gifMaker import GifMaker

import datetime
import os

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

TensorBoard(log_dir='Graph', histogram_freq=0,write_graph=True, write_images=True)


ENV_NAME = "CartPole-v1"
MODEL_SAVE_DIR = 'Model/'

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

DROPOUT_RATE = 0.4
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

logger = Logger()


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(32, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dropout(DROPOUT_RATE))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dropout(DROPOUT_RATE))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        self.model.summary()
        #self.tensorboard = TensorBoard(log_dir='Graph', histogram_freq=0, write_graph=True, write_images=True)

    def save(self, filename ='test'):
        self.model.save(MODEL_SAVE_DIR + str(filename) + '.h5')

    def load(self, filename='test'):
        self.model = load_model(MODEL_SAVE_DIR + filename + '.h5')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
            #self.model.fit(state, q_values, verbose=0, callbacks = [self.tensorboard])
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

# REVIEW:
# logger.reward not logging data`
def cartpole():
    maker = GifMaker()
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    dqn_solver.load()
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            env.render()
            maker.log(env.unwrapped.viewer.get_array())
            if terminal:
                logger.log(step)
                maker.makeGif(filename = str(run))

                if run % 10 is 0:
                    dqn_solver.save()
                    logger.plot(file = 'testing')
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                break
            dqn_solver.experience_replay()


if __name__ == "__main__":
    cartpole()
