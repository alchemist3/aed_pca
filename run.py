from pca import *

iris_pca = Pca()
iris_pca.load_arff('iris.arff')
iris_pca.pca()
print(
    "Zbiór iris - składowe główne wyjaśniające 90% zmiennych oryginalnych [stosunek wyjaśniania: numer składowej głównej]:")
print(iris_pca._90p_eigen_values)
iris_pca.print2d_pca()
iris_pca.print2d_rand()
iris_pca.print3d_pca()
iris_pca.print3d_rand()

wave_pca = Pca()
wave_pca.load_arff('waveform5000.arff')
wave_pca.pca()
print(
    "Zbiór waveform5000 - składowe główne wyjaśniające 90% zmiennych oryginalnych [stosunek wyjaśniania: numer składowej głównej]:")
pprint.pprint(wave_pca._90p_eigen_values)

# To increase plot generation speed, only 10% of data is plotted
# If no argument is given all data will be plotted.
wave_pca.print2d_pca(0.1)
wave_pca.print2d_rand(0.1)
wave_pca.print3d_pca(0.1)
wave_pca.print3d_rand(0.1)
