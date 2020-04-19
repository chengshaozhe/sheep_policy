
import pandas as pd
import numpy as np


def renormalVector(rawVector, targetLength):
    rawLength = np.power(np.power(rawVector, 2).sum(), 0.5)
    changeRate = np.divide(targetLength, rawLength)
    return np.multiply(rawVector, changeRate)


class Transition():
    def __init__(self, movingRange, speedList):
        self.movingRange = movingRange
        self.speedList = speedList

    def __call__(self, currentStates, currentActions):
        currentPositions = currentStates.loc[:][[
            'positionX', 'positionY']].values
        currentVelocities = currentStates.loc[:][[
            'velocityX', 'velocityY']].values
        numberObjects = len(currentStates.index)

        newVelocities = [renormalVector(np.add(currentVelocities[i], np.divide(
            currentActions[i], 2.0)), self.speedList[i]) for i in range(numberObjects)]
        newPositions = [np.add(currentPositions[i], newVelocities[i])
                        for i in range(numberObjects)]

        for i in range(numberObjects):
            if newPositions[i][0] > self.movingRange[2]:
                newPositions[i][0] = 2 * \
                    self.movingRange[2] - newPositions[i][0]
            if newPositions[i][0] < self.movingRange[0]:
                newPositions[i][0] = 2 * \
                    self.movingRange[0] - newPositions[i][0]
            if newPositions[i][1] > self.movingRange[3]:
                newPositions[i][1] = 2 * \
                    self.movingRange[3] - newPositions[i][1]
            if newPositions[i][1] < self.movingRange[1]:
                newPositions[i][1] = 2 * \
                    self.movingRange[1] - newPositions[i][1]

        newVelocities = [newPositions[i] - currentPositions[i]
                         for i in range(numberObjects)]
        newStatesList = [list(newPositions[i]) + list(newVelocities[i])
                         for i in range(numberObjects)]
        newStates = pd.DataFrame(
            newStatesList, index=currentStates.index, columns=currentStates.columns)
        return newStates


if __name__ == "__main__":
    statesList = [[10, 10, 0, 0], [10, 5, 0, 0], [15, 15, 0, 0]]
    currentStates = pd.DataFrame(statesList, index=[0, 1, 2], columns=[
                                 'positionX', 'positionY', 'velocityX', 'velocityY'])
    speedList = [5, 3, 3]
    currentActions = [[0, 3], [0, 3], [3, 0]]
    movingRange = [0, 0, 15, 15]

    transState = Transition(movingRange, speedList)

    newStates = transState(currentStates, currentActions)
    print(currentStates)
    print(currentActions)
    print(newStates)

    print(renormalVector((5,5),8))