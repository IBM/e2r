from optimize_ss import combine_heuristic

from optimize_entity import optimize_entity
from optimize_entity import optimize_entity_with_W
from optimize_W import optimize_W

from generate_feature_vectors_and_class_labels.options import Options
my_options = Options()

import numpy as np
import time
import pickle as pkl
import datetime
import sys
  
def convert(n): 
    return str(datetime.timedelta(seconds = n)) 

def compute_full_cost(y,x,concepts,nu,gamma):
    y=np.array(y)
    col_counts = np.sum(y,axis=0)
    return np.sum((concepts.dot(np.square(x)))*(1-(np.array(y)*(1+nu))))+\
           nu*np.sum(np.dot(np.square(x),np.array(y).T))\
           +gamma*np.sum(col_counts*(col_counts-1))/2;

def compute_full_cost_binary(P,x,concepts,nu):
    num_concepts,num_ent_pairs= np.shape(concepts)
    d=int(np.shape(P)[1] / 2)
    Q = np.tile(np.expand_dims(np.eye(2*d),axis=0),reps=(num_concepts,1,1))-P
    diff =Q-nu*P
    Psum = np.sum(P,axis=0)

    cost=0.0
    for i in range(num_ent_pairs):
        conceptsi=concepts[:,i].todense()
        filter = np.expand_dims(conceptsi,axis=2)
        Peff = np.sum(diff*filter,axis=0)+nu*Psum
        xi=x[i,:]
        cost+=np.matmul(np.matmul(xi,Peff),xi.T)
    return cost

def compute_full_cost_gamma(y,x,concepts,nu,gamma):
    # Complete src with appropriate format for gamma matrix
    #Return scalar cost
    #To be updated for matrix gamma
    return compute_full_cost(y, x, concepts, nu, gamma)


def compute_full_cost_features(y,x,concepts,nu,gamma, features,W,delta):
    gamma_cost =  compute_full_cost_gamma(y, x, concepts, nu, gamma)
    print("Shape of W:",np.shape(W))
    print("Shape of faetures:", np.shape(features))
    sys.stdout.flush()
    mapped_matrix = np.matmul(W,features)
    feature_cost = delta*np.linalg.norm(x.T-mapped_matrix)**2
    return gamma_cost+feature_cost

def train(concepts,d,nu,r,gamma,iter, pretrained='',choice = 'random',data='figer'):
    #Train without inductive embedding
    m = np.shape(concepts)[0]
    n = np.shape(concepts)[1]
    num_ones = np.sum(concepts)
    num_zeros = m*n-num_ones
    nu_old = nu
    gamma_old = gamma

    nu = nu*num_ones/num_zeros
    print('Original gamma:',gamma)

    gamma = gamma*num_ones/(d*m*(m-1)/2)
    print('New gamma:', gamma)
    if(pretrained==''):

        n = np.shape(concepts)[1]
        x = np.random.normal(0, 1, [n,d])
        x /= np.linalg.norm(x,axis=1,keepdims=True)
    else:
        x = pkl.load(open(pretrained,'rb'))
    start = time.time()
    for i in range(iter):
        print("Iteration %d ",i)
        print(convert(time.time()-start))
        y = combine_heuristic(x,concepts,nu,r,gamma) # Solves Problem 2
        full_cost = compute_full_cost(y, x, concepts, nu, gamma)
        print("Subspace Optimization full cost = %f", full_cost / num_ones)
        x = optimize_entity(y,concepts,nu, choice = choice) #Solves Prob 1
        full_cost = compute_full_cost(y, x, concepts, nu, gamma)
        # if(i>=1 and (((old_cost-full_cost)/old_cost)<0.00001)):
        #     break
        old_cost=full_cost
        print("Entity Optimization full cost = %f", full_cost/num_ones)
        if(i%5==0):
            pkl.dump(x,open('../output1/entities_new_data_'+data+'_d='+str(d)+ "_gamma=" + str(gamma_old) + "_nu=" + str(nu_old)+"_r=" + str(r)+"_"+choice+'.pkl','wb'),protocol=4)
            pkl.dump(y, open('../output1/concepts_new_data_'+data+'_d='+str(d)+ "_gamma=" + str(gamma_old) + "_nu=" + str(nu_old)+"_r=" + str(r)+"_"+choice+'.pkl', 'wb'), protocol=4)



def train_neurips(concepts, d, nu, r, gamma, iter, F, delta, pretrained='',choice = 'random'):
    #Train with inductive embedding
    data = 'figer'
    # Features : feature_dimX # of entities
    m = np.shape(concepts)[0]
    n = np.shape(concepts)[1]
    feature_dim = np.shape(F)[0]
    num_ones = np.sum(concepts)
    num_zeros = m*n-num_ones
    nu_old = nu
    gamma_old = gamma

    nu = nu*num_ones/num_zeros
    print('Original gamma:',gamma)

    gamma = gamma*num_ones/(d*m*(m-1)/2)
    print('New gamma:', gamma)


    if(pretrained==''):

        n = np.shape(concepts)[1]
        x = np.random.normal(0, 1, [n,d])
        x /= np.linalg.norm(x,axis=1,keepdims=True)
        W = np.random.normal(0, 1, [d,feature_dim])
    else:
        x = pkl.load(open(pretrained,'rb'))
        W = pkl.load(open(pretrained+'_W','rb'))
    start = time.time()
    Finv = np.matmul(F.T,np.linalg.pinv(np.matmul(F,F.T)))
    print("Shape of Finv", np.shape(Finv))

    for i in range(iter):
        print("Iteration %d ",i)
        print(convert(time.time()-start))

        y = combine_heuristic(x,concepts,nu,r,gamma) # Solves Problem 2
        full_cost = compute_full_cost_features(y,x,concepts,nu,gamma, F,W,delta)
        print("Subspace Optimization full cost = %f", full_cost / num_ones)

        x = optimize_entity_with_W(y,concepts,nu, W,F,delta) #Solves Prob 1
        full_cost = compute_full_cost_features(y,x,concepts,nu,gamma, F,W,delta)
        print("Entity Optimization full cost = %f", full_cost / num_ones)

        W = optimize_W(Finv,x)
        full_cost = compute_full_cost_features(y, x, concepts, nu, gamma, F, W, delta)
        print("W Optimization full cost = %f", full_cost / num_ones)


        old_cost=full_cost
        if(i%5==0):
                pkl.dump(x, open(my_options.qe_output_dir+'/with_non_leaf_entities_new_data_' + data + '_d=' + str(d) + "_gamma=" + str(gamma_old) + "_nu=" + str(nu_old) + "_r=" + str(r) + "_delta=" + str(delta) + "_" + choice + '.pkl','wb'), protocol=4)
                pkl.dump(y, open(my_options.qe_output_dir+'/with_non_leaf_concepts_new_data_' + data + '_d=' + str(d) + "_gamma=" + str(gamma_old) + "_nu=" + str(nu_old) + "_r=" + str(r) + "_delta=" + str(delta) + "_" + choice + '.pkl','wb'), protocol=4)
                pkl.dump(W, open(my_options.qe_output_dir+'/with_non_leaf_W_new_data_' + data + '_d=' + str(d) + "_gamma=" + str(gamma_old) + "_nu=" + str(nu_old) + "_r=" + str(r) + "_delta=" + str(delta) + "_" + choice + '.pkl', 'wb'), protocol=4)

