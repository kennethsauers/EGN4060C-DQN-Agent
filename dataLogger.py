import matplotlib.pyplot as plt
import datetime

class Logger():
    def __init__(self):
        self.reward = []

    def log(self, x):
        self.reward.append(x)

    def plot(self, save = True, show = False, file = 'false'):
        if file is 'false':
            filename = datetime.datetime.now()
        else:
            filename = file
        plt.plot(self.reward)
        plt.ylabel('Reward')
        plt.xlabel('Epoch')
        plt.title('Reward over time')
        if show:
            plt.show()
        if save:
            plt.savefig('{}.png'.format(filename))
