import numpy as np
import random
import pickle

# Función de activación sigmoide


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Derivada de la función sigmoide (para retropropagación)


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class OCRNeuralNetwork:
    # Inicializa la red neuronal
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Inicializar pesos y sesgos aleatoriamente
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # Ejecuta una pasada hacia adelante para obtener la predicción
    def predict(self, a):
        """Propagación hacia adelante para predicción"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return np.argmax(a)

    # Entrena la red usando descenso de gradiente estocástico.
    def train(self, training_data, epochs, learning_rate):
        for _ in range(epochs):
            random.shuffle(training_data)
            for x, y in training_data:
                self.update(x, y, learning_rate)

    # Realiza una actualización de pesos y sesgos para un solo ejemplo usando retropropagación.
    def update(self, x, y, learning_rate):
        # Inicialización de gradientes
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Propagación hacia adelante
        activation = x
        activations = [x]  # Lista de activaciones
        zs = []  # Lista de vectores z
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Retropropagación
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        # Actualizar pesos y sesgos
        self.weights = [w - learning_rate * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - learning_rate * nb
                       for b, nb in zip(self.biases, nabla_b)]

    # Funciones para guardar y cargar pesos
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump((self.weights, self.biases), f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.weights, self.biases = pickle.load(f)
