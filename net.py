import numpy as np

class FullyConnectedNet():

    def __init__(self, num_in, num_hidden1, num_hidden2, num_out):
        std = 0.1
        self.weights1 = std * np.random.randn(num_in,num_hidden1)
        self.bias1 = np.ones((1,num_hidden1))*0.1
        self.weights2 = std * np.random.randn(num_hidden1, num_hidden2)
        self.bias2 = np.ones((1,num_hidden2))*0.1
        self.weights3 = std * np.random.randn(num_hidden2, num_out)
        self.bias3 = np.ones((1,num_out))*0.1
        self.learning_rate = 0.01

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
        l1, self.cache1 = self.affine_forward(x, self.weights1, self.bias1)
        l2, self.cache2 = self.relu_forward(l1)
        l3, self.cache3 = self.affine_forward(l2, self.weights2, self.bias2)
        l4, self.cache4 = self.relu_forward(l3)
        l5, self.cache5 = self.affine_forward(l4, self.weights3, self.bias3)
        return l5

    def backward_pass(self, dout):
        dx5, dw5, db5 = self.affine_backward(dout, self.cache5)
        dx4 = self.relu_backward(dx5, self.cache4)
        dx3, dw3, db3 = self.affine_backward(dx4, self.cache3)
        dx2 = self.relu_backward(dx3, self.cache2)
        dx1, dw1, db1 = self.affine_backward(dx2, self.cache1)

        self.weights3 += self.learning_rate * dw5
        self.weights2 += self.learning_rate * dw3
        self.weights1 += self.learning_rate * dw1
        self.bias3 += self.learning_rate * db5
        self.bias2 += self.learning_rate * db3
        self.bias1 += self.learning_rate * db1

    def train(self, x, act, y=None):
        if len(x.shape) == 1:
            N = 1
            D = x.shape
        else:
            N, D = x.shape
        scores = self.forward_pass(x)
        if y is None:
            return scores
        scores[np.arange(len(act)),~act] = 0
        filtered_scores = np.zeros_like(scores)
        filtered_scores[np.arange(len(act)),act] = scores[np.arange(len(act)),act]
        loss = np.sum(np.square(y-filtered_scores)/N)
        dscores = (2*(y-filtered_scores))/N
        self.backward_pass(dscores)
        return loss

if __name__ == "__main__":
    net = FullyConnectedNet()
    for i in range(1000):
        x = np.array([[1,0,0,0],[0,1,0,0]])
        y = np.array([[1,0],[0,1]])
        net.train(x,y)
    print('w')
    print(net.weights1)