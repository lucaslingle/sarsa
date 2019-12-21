import os
import matplotlib.pyplot as plt

class PlotMaker:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def save(self, xys):
        xs, ys = zip(*xys)
        fig, ax = plt.subplots()
        ax.set(xlabel='total timesteps', ylabel='episodes completed')
        ax.plot(xs, ys)
        ax.grid()

        fp = os.path.join(self.output_dir, "progress.png")
        fig.savefig(fp)

        print("\nPlot of trained agent's progress over time")
        print('\t' + fp)