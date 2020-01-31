import numpy as np
import gates
import cv2
import shelve

#COMPUTES FORWARD PASS
def predict(X, W1, B1, W2, B2):
    #FORWARD PASS
    S1 = gates.MatMultGate()
    Z1 = gates.MatAddGate()
    A1 = gates.MatReLUGate()
    
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


trained_parameters_shelve = shelve.open("trained_parameters.db", writeback=True)
 
#PARAMETERS
W1 = trained_parameters_shelve["W1"]
B1 = trained_parameters_shelve["B1"]

W2 = trained_parameters_shelve["W2"]
B2 = trained_parameters_shelve["B2"]

trained_parameters_shelve.close()

print("\n[+] loading digit.png...")
img = cv2.imread("digit.png",cv2.IMREAD_GRAYSCALE)
X = 255-np.array(img)
#print(X)
X = (X.reshape(784,1)/255)*0.98 + 0.01
Y = predict(X, W1, B1, W2, B2)
print("\n[+] prediction:", np.argmax(Y))


