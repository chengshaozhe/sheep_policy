
import numpy as np
import pandas as pd
import itertools as it


def computeDistance(position1, position2):
    distance = np.sqrt(
        np.sum(np.power(np.array(position1) - np.array(position2), 2)))
    return distance


def samplePosition(positionRange):
    positionX = np.random.uniform(positionRange[0], positionRange[2])
    positionY = np.random.uniform(positionRange[1], positionRange[3])
    position = [positionX, positionY]
    return position


class InitialPosition():
    def __init__(self, movingRange, maxDistanceToFixation, minDistanceEachOther, maxDistanceEachOther, minDistanceWolfSheep):
        self.movingRange = movingRange
        self.maxDistanceToFixation = maxDistanceToFixation
        self.minDistanceEachOther = minDistanceEachOther
        self.maxDistanceEachOther = maxDistanceEachOther
        self.minDistanceWolfSheep = minDistanceWolfSheep

    def __call__(self, numberObjects):
        positionList = [samplePosition(self.movingRange)
                        for i in range(numberObjects)]
        pairList = list(it.combinations(range(numberObjects), 2))
        fixationPosition = [(self.movingRange[0] + self.movingRange[2]) / 2,
                            (self.movingRange[1] + self.movingRange[3]) / 2]
        sampleCount = 1
        while sampleCount < 100000:
            distanceToFixationArray = np.array(
                [computeDistance(position, fixationPosition) for position in positionList])
            distanceEachOtherArray = np.array([computeDistance(
                positionList[index[0]], positionList[index[1]]) for index in pairList])
            distanceWolfSheep = computeDistance(
                positionList[0], positionList[1])
            if (distanceWolfSheep > self.minDistanceWolfSheep) & np.all(distanceToFixationArray < self.maxDistanceToFixation) & np.all(distanceEachOtherArray > self.minDistanceEachOther) & np.all(distanceEachOtherArray < self.maxDistanceEachOther):
                break
            else:
                positionList = [samplePosition(
                    self.movingRange) for i in range(numberObjects)]
                sampleCount = sampleCount + 1
        if sampleCount == 100000:
            print("unable to initial correct positionList")
            return False
        else:
            # print(sampleCount)
            return positionList


if __name__ == "__main__":
    movingRange = [0, 0, 364, 364]
    maxDistanceToFixation = movingRange[3]
    minDistanceEachOther = 50
    maxDistanceEachOther = 180
    minDistanceWolfSheep = 120 
    numberObjects = 6

    initialPosition = InitialPosition(
        movingRange, maxDistanceToFixation, minDistanceEachOther, maxDistanceEachOther, minDistanceWolfSheep)

    positionList = initialPosition(numberObjects)
    print(positionList)
