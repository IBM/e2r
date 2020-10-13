import numpy as np
def optimize_W(Finv,X):
    # F :  Feature dimX # of entities
    # X : # of entities X Quantum_dim
    #We are solving X^T = WF
    W = np.matmul(X.T , Finv)
    #W : quantum_dimX feature_dim
    return W