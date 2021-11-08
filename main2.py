import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

import os



class Model:
    def __init__(self, stateSize, actionSize):
        self.stateSize = stateSize
        self.actionSize = actionSize


        self.memory = deque(maxlen=2000)
        self.batch_size = 16
        self.learning_rate = 0.001

        self.alpha = 0.9

        self.model = self.createModel()

        self.filepath = 'training1/cp.ckpt'
        self.cp_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=self.filepath, save_weight_only=True, verbose=1)
        


    def createModel(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.stateSize, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.actionSize, activation='linear'))

        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='mse' )
        return model
        
    def setModel(self):
        self.createModel()
        self.addDense()

    def training(self):
        batch = np.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.alpha * np.amax(self.model.predict(next_state)[0])
            target_outputs = self.model.predict(state)
            target_outputs[0][action] = target
            print("model fit")
                
            self.modelClass.model.fit(
                epochs=1,
                callbacks=[self.modelClass.cp_callbacks]
            )

    def action(self, state):
        predict = self.model.predict(state)
        return np.argmax(predict)

    def getModelWeight(self):
        self.model.load_weight(self.filepath)

    def memorize(self, data):
        self.memory.append(data)




class Acrobot:
    def __init__(self):
        self.episodes = 1000
        self.timesteps = 200
        self.score = []

        self.env = gym.make('Acrobot-v1')

        self.stateSize = self.env.observation_space.shape[0]
        self.actionSize = self.env.action_space.n

        self.modelClass = Model(self.stateSize, self.actionSize)
        
        

    def running(self):
        for e in range(self.episodes):
            state = np.reshape(self.env.reset(), [1, self.stateSize])

            for t in range(self.timesteps):
                self.env.render()
                print(state)
                action = self.modelClass.model.action(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.stateSize])

                self.modelClass.memorize((state, action, reward, next_state, done))

                if done or t == self.timesteps-1:
                    print('Episode :', e, ', Score :', t+1)
                    self.score.append(t + 1)
                    break

            self.modelClass.training()

        self.env.close()

    def showGraph(self):
        plt.plot(range(len(self.score)), self.score)
        plt.show()


    




acrobot = Acrobot()
acrobot.running()





