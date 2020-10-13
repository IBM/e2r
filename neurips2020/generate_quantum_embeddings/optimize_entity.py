from generate_feature_vectors_and_class_labels.options import Options
my_options = Options()
import numpy as np
import time
import torch
from scipy.linalg import eigh

#---------------------------------------------------
# The function below is used to train without inductive knowledge
#---------------------------------------------------
def optimize_entity(y,concepts,nu,choice='random'):
    print("Optimizing entity")
    #concepts[j][i] : 1 if jth concept has ith entity
    #y[j][k] : 1 if jth concept has kth axis
    x= []
    y = np.array(y)
    print("Computing eigs")
    eigs = concepts.T.dot(1-y*(1+nu))+nu*np.sum(y,axis=0,keepdims=True)    
    for i in range(np.shape(eigs)[0]):

        min_indices = np.where(eigs[i,:]==eigs[i,:].min())
        entity = np.zeros(y.shape[1])
        if(choice =='random'):
            mu, sigma = 0, 1
            s = np.random.normal(mu, sigma, np.shape(min_indices))
            s = s/np.linalg.norm(s)
            entity[min_indices] = s
        elif (choice == 'min'):
            entity[min_indices[0]] = 1
        elif(choice == 'uniform'):
            s = np.full(len(min_indices), 1.0/np.sqrt(len(min_indices)))
            entity[min_indices] = s
        else:
            raise Exception
        x.append(entity)

    #x : # of entities X Dimension
    return np.array(x)

if my_options.cpu_gpu == "cpu" or my_options.cpu_gpu=="local_machine":
    #-----------------------------
    def dual(mu,R,c):
        #print("called")
        return np.sum(c**2/((np.tile(mu,[1,np.shape(R)[1]])-R)**2),axis=1,keepdims=True)-4;
    # -----------------------------
    def dual_prime(mu,R,c):
        #print("derivative called ")
        return -2*np.sum(c ** 2 / ((np.tile(mu,[1,np.shape(R)[1]])-R) ** 3),axis=1,keepdims=True) ;
    # -----------------------------
    def my_bisect(x0,x1,R,c,max_steps=100,tol=1e-6):
        for step in range(max_steps):
            x_mid = (x0 + x1) / 2.0
            F0 = dual(x0, R, c)
            F1 = dual(x1, R, c)
            # if(step==0):
            #     assert(np.all(np.sign(F0)!=np.sign(F1)))
            F_mid = dual(x_mid, R,c)
            x0 = np.where(np.sign(F_mid) == np.sign(F0), x_mid, x0)
            x1 = np.where(np.sign(F_mid) == np.sign(F1), x_mid, x1)
            error_max = np.amax(np.abs(x1 - x0))
            print("step= error max=" , (step, error_max))
            if error_max < tol:
                break
        return x_mid
    # -----------------------------
    def optimize_entity_with_W(y,concepts,nu,W, F,delta):
        #Optimize entity using inductive knowledge
        # concepts[j][i] : 1 if jth concept has ith entity
        # y[j][k] : 1 if jth concept has kth axis
        # nu : Hyperparameter corresponding to \lambda
        # W :  quantum_dimX feature_dim
        # F :  Feature dimX # of entities
        # delta : Hyperparam controlling contribution of W term
        # Code to Compute the matrix x
        x = []
        y = np.array(y)
        print("Computing eigs")
        d = np.shape(y)[1]
        n = np.shape(concepts)[1]
        eigs = concepts.T.dot(1 - y * (1 + nu)) + nu * np.sum(y, axis=0, keepdims=True)+delta*np.ones([n,d])
        c = 2*delta*np.matmul(W,F).T
        eps = 1e-12
        x0 = np.expand_dims(np.array([-np.abs(c[i,:]).max()*np.sqrt(d) for i in range(np.shape(c)[0]) ]),axis=1)
        x1 = np.expand_dims(np.array([eigs[i,:].min()-eps for i in range(np.shape(c)[0])]),axis=1)
        mu0 = my_bisect(x0,x1,eigs,c ,tol=1e-3)
        x=0.5*c/(eigs-mu0)
        norm = np.linalg.norm(x,axis=1,keepdims=True)
        print("Avg Deviation from unit  Norm:" , np.mean(np.abs(norm-1)))
        x=x/norm
        return x
        # return X : numpy array of size # of entities X Dimension
    # -----------------------------
else:
    # -----------------------------
    def dual(mu,R,c):
        return torch.sum(c**2/((mu-R)**2),dim=1,keepdim=True)-4;
    # -----------------------------
    def dual_prime(mu,R,c):
        return -2*torch.sum(c ** 2 / ((mu-R) ** 3),dim=1,keepdim=True) ;
    # -----------------------------
    def my_bisect(x0,x1,R,c,max_steps=100,tol=1e-6):
        for step in range(max_steps):
            x_mid = (x0 + x1) / 2.0
            F0 = dual(x0, R, c)
            F1 = dual(x1, R, c)
            F_mid = dual(x_mid, R,c)
            x0 = torch.where(torch.sign(F_mid) == torch.sign(F0), x_mid, x0)
            x1 = torch.where(torch.sign(F_mid) == torch.sign(F1), x_mid, x1)
            error_max = torch.max(torch.abs(x1 - x0))
            print("step= error max=" , (step, error_max))
            if error_max < tol:
                break
        return x_mid
    # -----------------------------
    def optimize_entity_with_W(y,concepts,nu,W, F,delta):
        # Optimize entity using inductive knowledge
        # concepts[j][i] : 1 if jth concept has ith entity
        # y[j][k] : 1 if jth concept has kth axis
        # nu : Hyperparameter corresponding to \lambda
        # W :  quantum_dimX feature_dim
        # F :  Feature dimX # of entities
        # delta : Hyperparam controlling contribution of W term
        # Code to Compute the matrix x

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device ', device)
        y = np.array(y)
        d = np.shape(y)[1]


        n = np.shape(concepts)[1]
        eigs = concepts.T.dot(1 - y * (1 + nu)) + nu * np.sum(y, axis=0, keepdims=True)+delta*np.ones([n,d])
        eigs = torch.from_numpy(eigs).to(device)
        c = 2*delta*np.matmul(W,F).T
        c = torch.from_numpy(c).to(device)
        eps = 1e-12
        x0  = torch.tensor([-torch.abs(c[i,:]).max()*np.sqrt(d) for i in range(c.size()[0]) ]).unsqueeze(dim=1).to(device)
        x1 =  torch.tensor([eigs[i,:].min()-eps for i in range(c.size()[0])]).unsqueeze(dim=1).to(device)
        mu0 = my_bisect(x0,x1,eigs,c ,tol=1e-3)
        x = 0.5*c/(eigs-mu0)
        norm = torch.norm(x,dim=1,keepdim=True)
        print("Avg Norm:" , torch.mean(norm))
        x=x/norm
        return x.cpu().numpy()
        # return X : numpy array of size # of entities X Dimension
    # -----------------------------
