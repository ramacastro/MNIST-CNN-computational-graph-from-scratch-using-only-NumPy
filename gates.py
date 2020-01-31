import numpy as np


class MatMultGate():
    def forward(self, A, B):
        self.A = A
        self.B = B
        return np.dot(self.A, self.B)

    def backward(self, dZ):
        return np.dot(dZ, self.B.T), np.dot(self.A.T, dZ)


class MatAddGate():
    def forward(self, A, B):
        self.A = A
        self.B = B
        return self.A + self.B

    def backward(self, dZ):
        return dZ, np.sum(dZ, axis=1, keepdims=True)


class MatSigmoidGate():
    def sigmoid(self, A):
        return 1.0/(1.0 + np.exp(-A))

    def forward(self, A):
        self.A = A
        F = self.sigmoid(self.A)
        self.dA = F*(1-F)
        return F

    def backward(self, dZ):
        return dZ*self.dA


class MatLeakyReLUGate():
    def relu(self, A, alpha=0.1):
        A = np.where(A > 0, A, A * alpha)
        return A

    def relu_deriv(self, A, alpha=0.1):
        dA = np.ones_like(A)
        dA[A<0] = alpha
        return dA

    def forward(self, A):
        self.A = A
        F = self.relu(self.A)
        self.dA = self.relu_deriv(self.A)
        return F

    def backward(self, dZ):
        return dZ*self.dA


class MatReLUGate():
    def relu(self, A):
        #print(np.average(A))
        A[A<0] = 0
        return A

    def forward(self, A):
        self.A = A
        F = self.relu(self.A)
        self.dA = (F > 0) * 1
        return F

    def backward(self, dZ):
        return dZ*self.dA


class MSEGate():
    def forward(self, Y, T):
        self.Y = Y
        self.T = T
        _, self.n_examples = self.Y.shape
        loss = (self.T-self.Y)**2
        loss = np.sum(loss, axis=0, keepdims=True)/10
        loss = np.sum(loss)/self.n_examples
        self.dY = -(self.T-self.Y)/self.n_examples
        return loss

    def backward(self):
        return self.dY  


class SoftmaxGate():
    def softmax(self, A):
        A -= np.max(A)
        A_exp = np.exp(A)# + 1e-12
        return  A_exp/np.sum(A_exp, axis=0, keepdims=True) 

    def forward(self, A):
        self.A = A
        _, self.n_examples = self.A.shape
        F = self.softmax(self.A)
        #self.dA = F*(1-F)
        return F

    def backward(self, dZ):
        return dZ#*self.dA


class CrossEntropyLossGate():
    def cross_entropy(self, Y, T):
        log_likelihood = -np.log(Y[T.argmax(axis=0), range(self.n_examples)])
        #print("LOG_LIKELIHOOD_SHAPE:", log_likelihood.shape)
        #log_likelihood = -T*np.log(Y + 1e-12)
        loss = np.average(log_likelihood)
        return loss
        #_, self.n_examples = Y.shape
        #correct_probs = -T*np.log(Y + 1e-12)
        #return np.sum(correct_probs)/self.n_examples    

    def forward(self, Y, T):
        _, self.n_examples = Y.shape
    
        f = self.cross_entropy(Y, T)

        self.dF = Y#np.ones_like(T)#self.Y - self.T
        #self.dF /= self.n_examples
        self.dF[T.argmax(axis=0), range(self.n_examples)] -= 1
        self.dF /= self.n_examples
        
        #f = self.cross_entropy(self.Y, self.T)
        #self.dF = (self.Y - self.T)/self.n_examples
        return f

    def backward(self):
        return self.dF
