import tensorflow as tf
import numpy as np

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 200


class PolicyNetwork:
    def __init__(self, state_size, action_size):
        self.model = self.create_actor_network(state_size, action_size)

    def create_actor_network(self, state_size, action_size):
        S = tf.keras.layers.Input(shape=[state_size])
        h0 = tf.keras.layers.Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation='relu')(h0)
        out = tf.keras.layers.Dense(action_size, activation='softmax')(h1)
        model = tf.keras.models.Model(inputs=[S], outputs=[out])
        return model

    def __call__(self, state):
        action_values = self.model.predict(state)[0]
        action_index_max = np.argmax(action_values)
        return action_index_max


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

    actor = PolicyNetwork(state_size, action_size)

    actor.model.load_weights('./SingleWolf_DDPG_episode_800-actormodel.h5')

    agentState = np.asarray(statesListInit)
    agentState = np.reshape(agentState, [1, state_size])

    action_index_max = actor(agentState)
    sheepAction = sheepActionList[action_index_max]

    print (sheepAction)
