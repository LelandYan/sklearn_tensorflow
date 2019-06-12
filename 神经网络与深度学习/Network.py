# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/6/10 10:55'
import numpy as np


class Network:
    def __init__(self, sizes):
        """
        :param sizes:列表sizes包含各层神经元的数量
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 第一次神经网络（输入神经网络）不设置偏置值，并使用均值为0，标准差为1的高斯分布
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        n_test = None
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in np.arange(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in np.arange(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def cost_derivative(self,output_activations,y):
        return (output_activations - y)

    def sigmoid_prime(self,z):
        return self.sigmoid(z) * (1-self.sigmoid(z))
    def backprob(self,x,y):
        nabla_b = [np.zeros(b.shape)for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1],y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())
        for l in np.arange(2,self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
        return (nabla_b,nabla_w)