import glob
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def movingaverage(y, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(y, window, 'same')


def plot_file(filename, type='loss'):
    with open(f, 'r') as csvfile:
        reader = csv.reader(csvfile)
        y = []
        for row in reader:
            if type == 'loss':
                y.append(float(row[0]))

        if type == 'loss':
            window = 100
        y_av = movingaverage(y, window)

        arr = np.array(y_av)
        print("%f\t%f\n" % (arr.min(), arr.mean()))

        plt.clf()  # Clear.
        plt.title(f)
        plt.plot(y_av[:-50])
        plt.ylabel('Smoothed Loss')
        plt.ylim(0, 100)
        plt.xlim(0, 250000)

        plt.savefig(f + '.png', bbox_inches='tight')


if __name__ == "__main__":
    os.chdir("analysis/")
    score = []
    for f in glob.glob("score*.csv"):
        with open(f, 'r') as csvfile:
            reader = csv.reader(csvfile)
            y = []
            for row in reader:
                y.append(float(row[0]))
            x = np.mean(y)
            score.append(x)

    plt.plot(score)
    plt.ylim(0, 1000)
    plt.xlim(0, 6000)

    new_ticks = np.linspace(0, 6000, 100)
    plt.xticks(new_ticks)
    # plt.show()
    plt.savefig('score' + '.png', bbox_inches='tight')

    # plt.pause(100)
