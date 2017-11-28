from pca import *
import sys

# Python version check is necessary as dictionary expression
# used in this code work differently in python version below 3.6
if sys.version_info < (3, 6):
    raise "must use python 3.6"


iris_pca = Pca()
iris_pca.load_arff('iris.arff')
iris_pca.pca()
print(50*"#")
print(
    "Zbiór iris - składowe główne wyjaśniające 90% zmiennych oryginalnych [procent wyjaśnienia: numer składowej głównej]:")
print(iris_pca._90p_eigen_values)
print("Posortowany rosnąco wektor wartości własnych [wartość własna: nr wartości własnej]")
pprint.pprint(iris_pca.sorted_eigen_values)
print(50*"#")
iris_pca.print2d_pca()
iris_pca.print2d_rand()
iris_pca.print3d_pca()
iris_pca.print3d_rand()


wave_pca = Pca()
wave_pca.load_arff('waveform5000.arff')
wave_pca.pca()
print(
    "Zbiór waveform5000 - składowe główne wyjaśniające 90% zmiennych oryginalnych [procent wyjaśnienia: numer składowej głównej]:")
pprint.pprint(wave_pca._90p_eigen_values)
print("Posortowany rosnąco wektor wartości własnych [wartość własna: nr wartości własnej]")
pprint.pprint(wave_pca.sorted_eigen_values)
print(50*"#")

# To increase plot generation speed, only 10% of data is plotted
# If no argument is given all data will be plotted.
wave_pca.print2d_pca(0.1)
wave_pca.print2d_rand(0.1)
wave_pca.print3d_pca(0.1)
wave_pca.print3d_rand(0.1)
