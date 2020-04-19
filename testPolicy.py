import tensorflow as tf
import numpy as np
import random


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0
        self.learning_rate = 0.001
        self.model = self._buildDNN()

    def _buildDNN(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            400, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(300, activation='relu'))
        model.add(tf.keras.layers.Dense(
            self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def chooseAction(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        action_index_max = np.argmax(action_values[0])
        return action_index_max

    def getStateValue(self, state):
        action_values = self.model.predict(state)
        state_value = np.amax(action_values[0])
        return state_value

    def getMeanActionValues(self, state):
        action_values = self.model.predict(state)
        state_value = np.mean(action_values[0])
        return state_value

    def getQ(self, state):
        action_values = self.model.predict(state)
        return action_values[0]

    def load(self, name):
        self.model.load_weights(name)

    def __call__(self, state):
        action_values = self.model.predict(state)
        return action_values


if __name__ == '__main__':
    statesListInit = [[10, 10, 0, 0], [20, 20, 0, 0]]
    speedList = [8, 4, 4, 4, 4, 4]
    sheepIdentity = 0

    numOfActions = 16
    actionAnglesList = [i * (360 / numOfActions)
                        for i in range(1, numOfActions + 1)]
    sheepActionList = [np.array((speedList[sheepIdentity] * np.cos(actionAngles * np.pi / 180),
                                 speedList[sheepIdentity] * np.sin(actionAngles * np.pi / 180))) for actionAngles in actionAnglesList]

    state_size = 8
    action_size = numOfActions

    agent = DQNAgent(state_size, action_size)
    agent.load("./save/SingleWolf_episode_3000.h5")

    agentState = np.asarray(statesListInit)
    agentState = np.reshape(agentState, [1, state_size])

    actionValues = agent(agentState)

    action_index_max = agent.chooseAction(agentState)
    sheepAction = sheepActionList[action_index_max]
    print (actionValues)
    print (sheepAction)
