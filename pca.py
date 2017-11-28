import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import arff
from collections import OrderedDict
import pprint


class Pca:
    def __init__(self):
        # Data for classification
        self.x = []
        # Classes of samples
        self.y = []
        # Classes of samples without repetition
        self.classes = []
        self.pcaa = np.array(0)
        self.sorted_eigen_values = {}
        # 90% of most important eigen valus
        self._90p_eigen_values = {}

    def load_arff(self, file_path):
        data, meta = arff.loadarff(file_path)
        for w in range(len(data)):
            self.x.append([])
            for k in range(len(data[0])):
                if k == (len(data[0]) - 1):
                    self.y.append(data[w][k])
                else:
                    self.x[w].append(data[w][k])
        self.classes = list(set(self.y))

    def pca(self):
        # Data matrix without the class column
        X = np.array(self.x).transpose()
        covariance = np.cov(X)
        [eigen_values, eigen_vectors] = np.linalg.eig(covariance)
        # Sorted eigen values with their indexes in eigen values array
        self.sorted_eigen_values = {i: list(eigen_values).index(i) for i in sorted(eigen_values, reverse=True)}

        # 90% of most important eigen valus
        eigen_values_sum = sum(self.sorted_eigen_values.keys())
        for i in self.sorted_eigen_values:
            if sum(self._90p_eigen_values.keys()) < 0.9:
                self._90p_eigen_values[i / eigen_values_sum] = self.sorted_eigen_values[i]

        # Principal Component Analysis
        self.pcaa = np.dot(X.transpose(), eigen_vectors).transpose()

    def print2d(self, plot_ratio=1):
        # Assuming that there are only 3 classes (like in iris and waveform5000)

        # PCAs importance list
        importance = list(self.sorted_eigen_values.values())

        for i in range(int(len(self.y) * plot_ratio)):
            if self.y[i] == self.classes[0]:
                plt.scatter(self.pcaa[importance[0]][i], self.pcaa[importance[1]][i], s=4, c='r')  # , label='Klasa 1')
            elif self.y[i] == self.classes[1]:
                plt.scatter(self.pcaa[importance[0]][i], self.pcaa[importance[1]][i], s=4, c='g')  # , label='Klasa 2')
            else:
                plt.scatter(self.pcaa[importance[0]][i], self.pcaa[importance[1]][i], s=4, c='b')  # , label='Klasa 3')
            plt.title("Wizulizacja danych w rzucie na 2 pierwsze składowe główne")
            plt.xlabel('PCA1')
            plt.ylabel('PCA2')

            # # Handling multiple class labels
            # handles, labels = plt.gca().get_legend_handles_labels()
            # by_label = OrderedDict(zip(labels, handles))
            # plt.legend(by_label.values(), by_label.keys())
        plt.show()

    def print3d(self, plot_ratio=1):
        # Assuming that there are only 3 classes (like in iris and waveform5000)
        fig = plt.figure()
        ax = Axes3D(fig)

        # PCAs importance list
        importance = list(self.sorted_eigen_values.values())
        for i in range(int(len(self.y) * plot_ratio)):
            if self.y[i] == self.classes[0]:
                ax.scatter(self.pcaa[importance[0]][i], self.pcaa[importance[1]][i], self.pcaa[importance[2]][i], s=4,
                           c='r')  # , label='Klasa 1')
            elif self.y[i] == self.classes[1]:
                ax.scatter(self.pcaa[importance[0]][i], self.pcaa[importance[1]][i], self.pcaa[importance[2]][i], s=4,
                           c='g')  # , label='Klasa 2')
            else:
                ax.scatter(self.pcaa[importance[0]][i], self.pcaa[importance[1]][i], self.pcaa[importance[2]][i], s=4,
                           c='b')  # , label='Klasa 3')

            plt.title("Wizulizacja danych w rzucie na 3 pierwsze składowe główne")
            ax.set_xlabel('PCA1')
            ax.set_ylabel('PCA2')
            ax.set_zlabel('PCA3')

            ax.zaxis.set_rotate_label(True)
            ax.yaxis.set_rotate_label(True)

            # # Handling multiple class labels
            # handles, labels = plt.gca().get_legend_handles_labels()
            # by_label = OrderedDict(zip(labels, handles))
            # plt.legend(by_label.values(), by_label.keys())

        plt.show()


iris_pca = Pca()
iris_pca.load_arff('iris.arff')
iris_pca.pca()
print(
    "Zbiór iris - składowe główne wyjaśniające 90% zmiennych oryginalnych [stosunek wyjaśniania: numer składowej głównej]:")
print(iris_pca._90p_eigen_values)
iris_pca.print2d()
iris_pca.print3d()

wave_pca = Pca()
wave_pca.load_arff('waveform5000.arff')
wave_pca.pca()
print(
    "Zbiór waveform5000 - składowe główne wyjaśniające 90% zmiennych oryginalnych [stosunek wyjaśniania: numer składowej głównej]:")
pprint.pprint(wave_pca._90p_eigen_values)

# To increase plot generation speed, only 10% of data is plotted
# If no argument is given all data will be plotted.
wave_pca.print2d(0.1)
wave_pca.print3d(0.1)
