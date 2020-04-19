class DRQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._buildRNN()
        self.target_model = self._buildRNN()
        self.updateTargetModel()

    def _buildRNN(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(
            1, self.state_size), return_sequences=False))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        action = np.argmax(action_values[0])
        return action

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

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)

                states.append(state[0])
                targets.append(target[0])

        states_mb = np.array(states)
        targets_mb = np.array(targets)
        return states_mb, targets_mb

    def train(self, states_mb, targets_mb):
        # tensorboard = TensorBoard(log_dir='./logs')
        # history = self.model.fit(states_mb, targets_mb, epochs=1, verbose=0,callbacks=[tensorboard])
        history = self.model.fit(states_mb, targets_mb, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return history.history['loss']

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def __call__(self, state_img):
        action_values = self.model.predict(state_img)
        action_index_max = np.argmax(action_values[0])
        return action_index_max


if __name__ == '__main__':
    statesListInit = [[10, 10, 0, 0], [10, 5, 0, 0], [15, 15, 0, 0]]
    speedList = [8, 4, 4, 4, 4, 4]
    movingRange = [0, 0, 364, 364]
    assumeWolfPrecisionList = [50, 11, 3.3, 1.83, 0.92, 0.31]
    circleR = 10
    sheepIdentity = 0
    wolfIdentity = 1

    distractorPrecision = 0.5 / 3.14
    maxDistanceToFixation = movingRange[3]
    minDistanceEachOther = 50
    maxDistanceEachOther = 180
    minDistanceWolfSheep = 120

    numberObjects = 2

    PureAttentionModel = 0
    HybridModel = 0
    IdealObserveModel = 1

    if PureAttentionModel:
        attentionLimitation = 2
        precisionPerSlot = 8.0
        precisionForUntracked = 0
        memoryratePerSlot = 0.7
        memoryrateForUntracked = 0

    if HybridModel:
        attentionLimitation = 2
        precisionPerSlot = 8
        precisionForUntracked = 2.5
        memoryratePerSlot = 0.7
        memoryrateForUntracked = 0.45

    if IdealObserveModel:
        attentionLimitation = 100
        precisionPerSlot = 50
        precisionForUntracked = 50
        memoryratePerSlot = 0.99
        memoryrateForUntracked = 0.99

    attentionSwitchFrequency = 12

    initialPosition = InitialPosition(
        movingRange, maxDistanceToFixation, minDistanceEachOther, maxDistanceEachOther, minDistanceWolfSheep)
    transState = Transition.Transition(movingRange, speedList)
    computePrecisionAndDecay = Attention.AttentionToPrecisionAndDecay(
        precisionPerSlot, precisionForUntracked, memoryratePerSlot, memoryrateForUntracked)
    switchAttention = Attention.AttentionSwitch(attentionLimitation)
    updateBelief = BeliefUpdateWithAttention(
        computePrecisionAndDecay, switchAttention, attentionSwitchFrequency, sheepIdentity)

    takeWolfAction = WolfPolicy(
        sheepIdentity, wolfIdentity, speedList[wolfIdentity])

    numOfActions = 16
    actionAnglesList = [i * (360 / numOfActions)
                        for i in range(1, numOfActions + 1)]

    sheepActionList = [np.array((speedList[sheepIdentity] * np.cos(actionAngles * np.pi / 180),
                                 speedList[sheepIdentity] * np.sin(actionAngles * np.pi / 180))) for actionAngles in actionAnglesList]

    state_size = numberObjects * 4
    action_size = numOfActions
    agent = DQNAgent(state_size, action_size)

    # agent.load("./save/IdealObserveModel_episode_9850.h5")

    loss_log = []
    score_log = []

    batch_size = 64
    replay_start_size = 1000
    num_opisodes = 100001

    for e in range(num_opisodes):
        score = 0

        init_positionList = initialPosition(numberObjects)
        # print(init_positionList)
        if init_positionList == False:
            continue

        statesList = []
        initVelocity = [0, 0]
        for initPosition in init_positionList:
            statesList.append(initPosition + initVelocity)

        # print (statesList)

        oldStates = pd.DataFrame(statesList, index=list(range(numberObjects)),
                                 columns=['positionX', 'positionY', 'velocityX', 'velocityY'])

        # wolfPrecision = random.choice(assumeWolfPrecisionList)
        wolfPrecision = 50
        done = False

        for time in range(1000):
            oldStates_array = np.asarray(oldStates).flatten()

            state_series = np.array(
                [trace[-1] for trace in agent.memory[-agent.trace_length:]])
            state_series = np.expand_dims(state_series, axis=0)

            action = agent.act(oldStates_input)
            sheepAction = sheepActionList[action]

            wolfAction = takeWolfAction(oldStates, wolfPrecision)
            currentActions = [sheepAction, wolfAction]

            currentStates = transState(oldStates, currentActions)
            currentStates_array = np.asarray(currentStates).flatten()

            reward = stateReward(currentStates_array,
                                 currentActions, movingRange)

            if isTerminals(currentStates_array):
                done = 1
            else:
                done = 0

            agent.remember(oldStates_input, action, reward,
                           currentStates_input, done)

            oldStates = currentStates

            if len(agent.memory) > replay_start_size:
                states_mb, targets_mb = agent.replay(batch_size)
                loss = agent.train(states_mb, targets_mb)

            if done:
                # agent.updateTargetModel()
                score = time
                break

        if e % 3 == 0:
            agent.updateTargetModel()
