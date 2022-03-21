import pickle
import matplotlib.pyplot as plt

# averaged over n episodes, plot data saved by append_data in each step

class RealtimePlot:
    def __init__(self, title = "Title", y_label = "y", n_average = 20, x_label = "Episodes", loglog = True):
        self.n_average = n_average
        self.y_label = y_label
        self.x_label = x_label
        self.y_data_step = []
        self.x_data = []
        self.y_data = []
        self.title = title
        self.loglog = loglog

        self.figure = plt.figure()
        self.line, = plt.plot(self.x_data, self.y_data, '-')


    def append(self, y_data_step):
        self.y_data_step.append(y_data_step)
        # if average is reached:
        if len(self.y_data_step) == self.n_average:


            self.y_data.append(sum(self.y_data_step) / self.n_average)
            self.y_data_step = []

            if len(self.x_data) == 0:
                self.x_data.append(self.n_average)
            else:
                last_x = self.x_data[len(self.x_data)-1]
                self.x_data.append(self.n_average + last_x)
        if len(self.y_data) == 0:
            self.plot() # for some reason, it is blacked out after the first plot, this is a workaround.
        self.plot()

    def plot(self, loglog = None):
        self.line.set_data(self.x_data, self.y_data)
        self.figure.gca().relim()
        self.figure.gca().autoscale_view()
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        if loglog is None:
            if self.loglog:
                plt.yscale("log")
                plt.xscale("log")
            else:
                plt.yscale("linear")
                plt.xscale("linear")
        elif loglog:
            plt.yscale("log")
            plt.xscale("log")
        else:
            plt.yscale("linear")
            plt.xscale("linear")
        self.figure.show()
        plt.pause(0.0001)

    def store(self, name = None):
        if name is None:
            name = "models/" + self.title
        with open(name + ".pt", "wb") as file:
            pickle.dump(self, file)


class Recorder:
    def __init__(self, title = "Title", y_label = ["y"], n_average = 20):
        self.n_average = n_average
        self.y_label = y_label
        self.num_data = len(y_label)
        self.y_data_step = [[]]*self.num_data
        self.x_data = []
        self.y_data = [[]]*self.num_data
        self.title = title


    def append(self, y_data_step):
        for i in range (self.num_data):
            self.y_data_step[i].append(y_data_step[i])
        # if average is reached:
        if len(self.y_data_step[0]) == self.n_average:
            for i in range (self.num_data):
                self.y_data[i].append(sum(self.y_data_step[i]) / self.n_average)
                self.y_data_step = [[]]*self.num_data

            if len(self.x_data) == 0:
                self.x_data.append(self.n_average)
            else:
                last_x = self.x_data[len(self.x_data)-1]
                self.x_data.append(self.n_average + last_x)


    def store(self, name = None):
        if name is None:
            name = "models/" + self.title
        with open(name + ".pt", "wb") as file:
            pickle.dump(self, file)
