import numpy as np
import matplotlib.pyplot as plt
from csv import reader

def loadCsv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(list(map(int, row)))

	return dataset

class Perceptron(object):

    def __init__(self, noOfInputs, threshold=1000, learningRate=0.01):
        self.threshold = threshold  # ilosc przejsc przez zestaw treningowy
        self.learningRate = learningRate  # wielkosc zmiany wag po przejsciu
        self.weights = np.random.rand(noOfInputs + 1)  # losowanie wag w ilosci : ilosc imputow + 1 waga biasu)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # f(x) = 1 if w · x + b > 0 : 0 otherwise
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation # jesli zwroci 1 to od output odejmujemy jeden w train, czyli zmieniamy wagi, jak 0 to wagi sie nie zmieniaja

    def train(self, trainingInputs, outputs):
        for _ in range(
                self.threshold):  # loop - ponizsza petla wykona sie dla jednego zestawu danych treningowych tyle ile thresholdow
            for inputs, output in zip(trainingInputs,
                                      outputs):  # zipujemy zeby stworzyc obiekt po ktorym mozemy iterowac i loopujemy przez niego
                prediction = self.predict(inputs)
                self.weights[1:] += self.learningRate * (
                            output - prediction) * inputs  # znajdujemy blad (output - prediction), mnozymy bo przez learningRate i zapisujemy do self.weights[1:]
                self.weights[0] += self.learningRate * (
                            output - prediction)  # updateujemy bias, ale nie mnozymy go juz przez imputy



filename = 'xySet.csv'
filename2 = 'categorySet.csv'

dataset = loadCsv(filename)
dataset2 = loadCsv(filename2)

trainingInputs = []
for i in range (len(dataset)):
    trainingInputs.append((np.array(dataset[i])))

trainingOutputs = []
for i in range (len(dataset2)):
    trainingOutputs.append((np.array(dataset2[i])))

perceptron = Perceptron(2)

perceptron.train(trainingInputs, trainingOutputs)

filename3 = 'xySet2.csv'
dataset3 = loadCsv(filename3)

Inputs = []
for i in range (len(dataset3)):
    Inputs.append((np.array(dataset3[i])))

results = []
for i in range (len(dataset3)):
    results.append(perceptron.predict(Inputs[i]))
    print(perceptron.predict(Inputs[i]))
    print(Inputs[i], sep=" ---")

cols = []
for i in range (len(Inputs)):
    if results[i] == 1:
        plt.plot([Inputs[i][0]], [Inputs[i][1]], 'ro')
    if results[i] == 0:
        plt.plot([Inputs[i][0]], [Inputs[i][1]], 'bo')
plt.plot([-10,10, 15],[10,-10, -15])
plt.show()


