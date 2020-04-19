
import tensorflow as tf


class NN():
    def __init__(self, HIDDEN1_UNITS, HIDDEN2_UNITS, LEARNING_RATE):
        self.HIDDEN1_UNITS = HIDDEN1_UNITS
        self.HIDDEN2_UNITS = HIDDEN2_UNITS
        self.learning_rate = LEARNING_RATE

    def __call__(self, state_size, action_size):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            HIDDEN1_UNITS, input_dim=state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(HIDDEN2_UNITS, activation='relu'))
        model.add(tf.keras.layers.Dense(
            action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        weights = model.get_weights()
        return model, weights


class DQN:
    def __init__(self, state_size, action_size, BATCH_SIZE, LEARNING_RATE, EPSILON, numOfOpisodes):
        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE
        self.epsilon = EPSILON
        self.memory = deque(BUFFER_SIZE)
        self.state_size = state_size
        self.action_size = action_size
        self.numOfOpisodes = numOfOpisodes

    def __call__(self, state, model):
        for e in range(self.numOfOpisodes):
            target_model = model

            action = policy(state, model, self.action_size)
            next_state, reward, done = transition(state, action)

            memory = self.memory.append(
                (state, action, reward, next_state, done))

            state = next_state

            loss, model, weights = trainModel(memory, model, target_model)
            target_model, target_weights = updateTargetModel(
                target_model, weights, TAU)

        return weights

    def trainModel(self, memory, model, target_model):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        minibatch = random.sample(memory, self.batch_size)
        states = np.asarray([e[0] for e in minibatch])
        actions = np.asarray([e[1] for e in minibatch])
        rewards = np.asarray([e[2] for e in minibatch])
        new_states = np.asarray([e[3] for e in minibatch])
        dones = np.asarray([e[4] for e in minibatch])
        y_t = np.asarray([e[1] for e in minibatch])

        if len(memory) > REPLAY_START_SIZE:
            target_q_values = target_model.predict(new_states)
            for k in range(len(minibatch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            _, loss = sess.run(optimize, feed_dict={self.state: states})
            # loss += model.train_on_batch(states, y_t)

            weights = model.get_weights()

        return loss, model, weights


def updateTargetModel(target_model, weights, TAU):
    target_weights = target_model.get_weights()
    for i in range(len(weights)):
        target_weights[i] = TAU * weights[i] + \
            (1 - TAU) * target_weights[i]
    target_model.set_weights(target_weights)

    return target_model, target_weights


def policy(state, model, action_size):
    if np.random.rand() <= EPSILON:
        action = random.randrange(action_size)
    else:
        action_values = model.predict(state)
        action = np.argmax(action_values[0])

    return action


def main():
    BUFFER_SIZE = 100000
    BATCH_SIZE = 64
    TAU = 0.001
    HIDDEN1_UNITS = 32
    HIDDEN2_UNITS = 32
    LEARNING_RATE = 0.0001
    GAMMA = 0.99
    EPSILON = 0.1
    REPLAY_START_SIZE = 100
    numOfOpisodes = 100000

    initState, state_size, action_size = gameInit()

    nn = NN(HIDDEN1_UNITS, HIDDEN2_UNITS)
    model = nn(state_size, action_size)

    agent = DQN(state_size, action_size, BATCH_SIZE,
                LEARNING_RATE, EPSILON, numOfOpisodes)

    weights = agent(initState, model)


if __name__ == '__main__':
    main()
