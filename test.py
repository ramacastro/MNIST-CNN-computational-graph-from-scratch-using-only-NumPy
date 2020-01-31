import numpy as np
import gates
import csv
import shelve

#COMPUTES FORWARD PASS
def predict(iteration, X, W1, B1, W2, B2, T):
    #FORWARD PASS
    S1 = gates.MatMultGate()
    Z1 = gates.MatAddGate()
    A1 = gates.MatReLUGate()
    #A1 = gates.MatSigmoidGate()
    
    S2 = gates.MatMultGate()
    Z2 = gates.MatAddGate()
    A2 = gates.SoftmaxGate()
    
    S1_out = S1.forward(W1, X)
    Z1_out = Z1.forward(S1_out, B1)
    A1_out = A1.forward(Z1_out)
    
    S2_out = S2.forward(W2, A1_out)
    Z2_out = Z2.forward(S2_out, B2)
   
    Y = A2.forward(Z2_out)

    return Y


def read_csv(filename):
	print("\n[+] Loading " + filename + " ...")

	inputs = []
	targets = []

	with open(filename) as csvfile:
		reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			y = np.zeros(10) + 0.01
			y[int(row[0])] = 0.99
			targets.append(y)
			inputs.append((np.asfarray(row[1:])/255)*0.99 + 0.01)

	return np.asarray(inputs), np.asarray(targets)


trained_parameters_shelve = shelve.open("trained_parameters.db", writeback=True)

 
#PARAMETERS
W1 = trained_parameters_shelve["W1"]

B1 = trained_parameters_shelve["B1"]

W2 = trained_parameters_shelve["W2"]

B2 = trained_parameters_shelve["B2"]

trained_parameters_shelve.close()

inputs, targets = read_csv("mnist_test.csv")

n_examples = len(inputs)

predicted_ok = 0

print("\n[+] Calculating accuracy on mnist_test.csv...")

for i in range(n_examples):
    Xi = np.array(inputs[i], ndmin=2).T
    Ti = np.array(targets[i], ndmin=2).T

    Yi = predict(i, Xi, W1, B1, W2, B2, Ti)

    if np.argmax(Yi) == np.argmax(Ti):
        predicted_ok += 1

     
accuracy = np.round((predicted_ok/n_examples)*100, 2)

print("\n[+] Accuracy: ", accuracy, "%")