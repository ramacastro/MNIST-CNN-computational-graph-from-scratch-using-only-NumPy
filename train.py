from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import gates
import csv
import shelve
from tqdm import tqdm
import time

#COMPUTES FORWARD AND BACKWARD PASS
def compute(X, W1, B1, W2, B2, T):
    #FORWARD PASS
    S1 = gates.MatMultGate()
    Z1 = gates.MatAddGate()
    A1 = gates.MatReLUGate()
    #A1 = gates.MatSigmoidGate()

    S2 = gates.MatMultGate()
    Z2 = gates.MatAddGate()
    A2 = gates.SoftmaxGate()
    #A2 = gates.MatSigmoidGate()
    
    S1_out = S1.forward(W1, X)
    Z1_out = Z1.forward(S1_out, B1)
    A1_out = A1.forward(Z1_out)

    avg_activations = np.average(A1_out)
    dead_activations = np.sum(A1_out == np.zeros_like(A1_out))
    
    S2_out = S2.forward(W2, A1_out)
    Z2_out = Z2.forward(S2_out, B2)
   
    Y = A2.forward(Z2_out)
    #print("Yshape:", Y.shape)

    #LOSS
    LOSS = gates.CrossEntropyLossGate()
    #LOSS = gates.MSEGate()
    loss_value = LOSS.forward(Y, T)

    #print("loss_value:", loss_value)

    #BACKWARD PASS
    dLOSS = LOSS.backward()
    
    dA2 = A2.backward(dLOSS)
    dZ2_S2, dZ2_B2 = Z2.backward(dA2)

    dS2_W2, dS2_A1 = S2.backward(dZ2_S2)
    dA1 = A1.backward(dS2_A1)

    dZ1_S1, dZ1_B1 = Z1.backward(dA1)
    dS1_W1, dS1_X = S1.backward(dZ1_S1)    

    return Y, loss_value, dS1_W1, dZ1_B1, dS2_W2, dZ2_B2, dead_activations, avg_activations
    

def read_csv(filename):
    print("\n[+] Loading " + filename + " ...")

    inputs = []
    targets = []

    with open(filename) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            T = np.zeros(10) + 0.01
            T[int(row[0])] = 0.99
            targets.append(T)
            X = np.asfarray(row[1:])
            X = (X/255)*0.99 + 0.01            
            inputs.append(X)
    return np.asarray(inputs), np.asarray(targets)

np.random.seed(3)
#PARAMETERS
x_size = 784
h_nodes = 100
y_size = 10

batch_size = 128

#LEARNING RATE
lr = 0.01

#REGULARIZATION STRENGTH
reg = 0.0001

#EPOCHS     
epochs = 1000


W1 = np.random.randn(h_nodes,x_size)*np.sqrt(2/x_size)

B1 = np.zeros((h_nodes,1))#0.01*np.random.rand(h_nodes,1)#

W2 = np.random.randn(y_size,h_nodes)*np.sqrt(2/h_nodes)

B2 = np.zeros((y_size,1))#0.01*np.random.rand(y_size,1)#

print("W1[0][0]:", W1[0][1])

X, T = read_csv("mnist_train.csv")

#X = X[:10, :]
#T = T[:10, :]

X -= np.mean(X, axis=0, keepdims=True)


print("\n[+] X_shape:", X.shape)
print("[+] T_shape:", T.shape)

#print(X_mini.shape)
#print(T_mini.shape)

n_examples, _ = X.shape

np.seterr(divide='ignore', invalid='ignore')

loss_values = []

try:
    for e in range(epochs):
        print("\n------------------[EPOCH " + str(e+1) + "]------------------\n")
        X_shuffled, T_shuffled = shuffle(X, T)
        total_cost = 0
        n_batches = 0
        predicted_ok = 0
        total_dead_activations = 0
        total_avg_activations = 0
        dW1i_ratio = 0
        dB1i_ratio = 0
        dW2i_ratio = 0
        dB2i_ratio = 0

        for i in range(0, n_examples, batch_size):
            Xi = X_shuffled[i:i+batch_size, :].T
            Ti = T_shuffled[i:i+batch_size, :].T

            Yi, loss_value, dW1i, dB1i, dW2i, dB2i, dead_activations, avg_activations = compute(Xi, W1, B1, W2, B2, Ti)
            
            predicted_ok += np.sum(np.argmax(Yi, axis=0) == np.argmax(Ti, axis=0))

            #print("[+] loss_value:", loss_value)
            total_dead_activations += dead_activations
            total_avg_activations += avg_activations
            #loss_values.append(loss_value)

            reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
            loss_value += reg_loss

            total_cost += loss_value
            n_batches += 1

            dW1i += reg*W1
            dW2i += reg*W2

            dW1i *= lr
            dB1i *= lr
            dW2i *= lr
            dB2i *= lr
        
            W1_scale = np.linalg.norm(W1)
            B1_scale = np.linalg.norm(B1)
            W2_scale = np.linalg.norm(W2)
            B2_scale = np.linalg.norm(B2)
            
            dW1i_scale = np.linalg.norm(dW1i)
            dB1i_scale = np.linalg.norm(dB1i)
            dW2i_scale = np.linalg.norm(dW2i)
            dB2i_scale = np.linalg.norm(dB2i)

            dW1i_ratio += dW1i_scale/W1_scale

            if B1_scale == 0:
                dB1i_ratio = 0
            else:
                dB1i_ratio += dB1i_scale/B1_scale
            
            dW2i_ratio += dW2i_scale/W2_scale
            
            if B2_scale == 0:
                dB2i_ratio = 0
            else:
                dB2i_ratio += dB2i_scale/B2_scale
            
            W1 -= dW1i
            B1 -= dB1i
            W2 -= dW2i
            B2 -= dB2i

        accuracy = np.round((predicted_ok/(n_batches*batch_size))*100, 2)
        cost = total_cost/n_batches
        loss_values.append(cost)

        print("[+] COST:", cost)
        print("[+] N_BATCHES:", n_batches)
        print("[+] DEAD_ACTIVATIONS:", np.round((total_dead_activations/(h_nodes*n_batches*batch_size))*100, 2), "%")
        print("[+] AVG_ACTIVATIONS:", avg_activations/n_batches)
        print("[+] dW1i_ratio:", dW1i_ratio/n_batches)
        print("[+] dB1i_ratio:", dB1i_ratio/n_batches)
        print("[+] dW2i_ratio:", dW2i_ratio/n_batches)
        print("[+] dB2i_ratio:", dB2i_ratio/n_batches)
        #time.sleep(1)
        #print("[+] PREDICTED_OK:", predicted_ok)
        #print("[+] ACCURACY:", accuracy, "%")
          
except KeyboardInterrupt:
    pass

print("\n---------------------------------------------\n")

trained_parameters_shelve = shelve.open("trained_parameters.db", writeback=True)

trained_parameters_shelve["W1"] = W1
trained_parameters_shelve["B1"] = B1
trained_parameters_shelve["W2"] = W2
trained_parameters_shelve["B2"] = B2

trained_parameters_shelve.sync()

print("[+] Parameters saved successfully to trained_parameters.db")

plt.plot(loss_values)
plt.show()