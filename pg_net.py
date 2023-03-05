import numpy as np

class FullyConnectedNet():

    def __init__(self, layers, learning_rate = 1, reg=0):
        std = 0.1
        self.params = {}
        self.cache = {}
        self.num_layers = len(layers) - 1
        for layer in range(self.num_layers):
            self.params['W' + str(layer + 1)] = std * np.random.randn(layers[layer],layers[layer + 1])
            self.params['b' + str(layer + 1)] = np.ones((1,layers[layer + 1]))*0.1

        self.learning_rate = learning_rate
        self.reg = reg

    def affine_forward(self,x,w,b):
        N = np.matrix(x).shape[0]
        D = np.prod(np.matrix(x).shape[1:])
        flat_x = np.reshape(np.matrix(x),(N,D))
        out = flat_x.dot(w) + b
        cache = (x, w, b)
        return out, cache

    def relu_forward(self,x):
        out = np.maximum(0,x)
        cache = x
        return out, cache

    def affine_backward(self, dout, cache):
        x, w, b = cache
        N = x.shape[0]
        D = np.prod(x.shape[1:])
        flat_x = np.reshape(x,(N,D))
        dx, dw, db = dout.dot(w.T).reshape(x.shape), flat_x.T.dot(dout).reshape(w.shape), np.sum(dout, axis=0)
        return dx, dw, db

    def relu_backward(self, dout, cache):
        dx, x = dout, cache
        dx[x < 0] = 0
        return dx

    def forward_pass(self, x):
        out = x
        for layer in range(self.num_layers-1):
            out, cache1 = self.affine_forward(out, self.params['W' + str(layer + 1)], self.params['b' + str(layer + 1)])
            out, cache2 = self.relu_forward(out)
            self.cache[layer] = (cache1, cache2)
        out, cache1 = self.affine_forward(out, self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
        self.cache[self.num_layers-1] = cache1
        return out

    def backward_pass(self, dout):
        dx, dw, db = self.affine_backward(dout, self.cache[self.num_layers-1])
        print(dw)
        print(dx)
        lol
        self.params['W' + str(self.num_layers)] += self.learning_rate * (dw - self.reg * self.params['W' + str(self.num_layers)])
        self.params['b' + str(self.num_layers)] += self.learning_rate * db
        for layer in range(self.num_layers-1, 0, -1):
            dx = self.relu_backward(dx, self.cache[layer-1][1])
            dx, dw, db = self.affine_backward(dx, self.cache[layer-1][0])
            self.params['W' + str(layer)] += self.learning_rate * (dw - self.reg * self.params['W' + str(layer)])
            self.params['b' + str(layer)] += self.learning_rate * db
        return dw

if __name__ == "__main__":
    net = FullyConnectedNet([5, 10, 10, 2])
    print("hi")