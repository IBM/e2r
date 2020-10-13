import numpy as np
import sys
def optimize_rank_constraint(phi,r):
    # phi: No. of conceptsX Dimension
    m = len(phi)
    d = len(phi[0])
    y = [[0]*d for _ in range(m)]
    for j in range(m):
        sorted_list = sorted(zip(phi[j],range(d)), key=lambda x: x[0])
        for k in range(d):
            if((k<r) or (sorted_list[k][0]<0)):
                y[j][sorted_list[k][1]] = 1
            else:
                break
    return y


def optimize_ortho_terms(phi, gamma):
    m = len(phi)
    d = len(phi[0])
    y = [[0] * d for _ in range(m)]
    for k in range(d):
        sorted_list = sorted([(phi[j][k],j) for j in range(m)], key = lambda x: x[0])
        for j in range(m):
            if(sorted_list[j][0]+gamma*j<=0):
                y[sorted_list[j][1]][k] =1
            else:
                break
    return y


def optimize_ortho_terms_2gamma(phi, gamma0,gamma1,):
    m = len(phi)
    d = len(phi[0])
    y = [[0] * d for _ in range(m)]
    for k in range(d):
        sorted_list = sorted([(phi[j][k],j) for j in range(m)], key = lambda x: x[0])
        for j in range(m):
            if(sorted_list[j][0]+gamma*j<=0):
                y[sorted_list[j][1]][k] =1
            else:
                break
    return y



def compute_phi(x, concepts, nu):
    #x: # of entities X d, numpy array
    #concepts: # of subspaces X # of entities, numpy array (sparse)
#     m = len(concepts)
#     d = len(x[0])
    print("computing phi_minus")
    sys.stdout.flush()
    phi_minus = (1+nu)*(concepts.dot(np.square(x)))
    print("computing phi_plus")
    sys.stdout.flush()
    phi_plus = nu*np.sum(np.square(x),axis=0,keepdims=True)
    phi = phi_plus-phi_minus
    return phi


def heuristic1(phi, r, gamma):

    print("Running h1")
    print("shape of phi: %d %d", len(phi),len(phi[0]))
    y = optimize_rank_constraint(phi,r)
    m = len(y)
    d = len(y[0])
    row_counts = [sum(y[j]) for j in range(m)]
    col_counts = [sum([y[j][k] for j in range(m)]) for k in range(d)]
    valid = set([(j, k) for j in range(m) for k in range(d) if((y[j][k] == 1) and (row_counts[j] > r)
                                                           and (phi[j][k]+gamma*(col_counts[k]-1) > 0))])
    while valid:
        (j1 , k1) = max(valid, key=lambda pair:phi[pair[0]][pair[1]]+gamma*(col_counts[pair[1]]-1))
        y[j1][k1] = 0
        row_counts[j1]-=1
        col_counts[k1]-=1
        valid.remove((j1,k1))
        if(row_counts[j1]<=r):
            valid = valid-set([(j1,k) for k in range(d)])
        valid = valid-set([(j,k1) for j in range(m) if (phi[j][k1]+gamma*(col_counts[k1]-1) <= 0) ])
    return y;

def heuristic2(phi, r,gamma):
    print("Running h2")
    y = optimize_ortho_terms(phi,gamma)
    m = len(y)
    d = len(y[0])
    row_counts = [sum(y[j]) for j in range(m)]
    col_counts = [sum([y[j][k] for j in range(m)]) for k in range(d)]
    valid = set([(j, k) for j in range(m) for k in range(d) if ((y[j][k] == 0) and (row_counts[j] < r))])
    while valid:
        (j1,k1) = min(valid, key=lambda pair:phi[pair[0]][pair[1]]+gamma*col_counts[pair[1]])
        y[j1][k1]=1
        row_counts[j1]+=1
        col_counts[k1]+=1
        valid.remove((j1,k1))
        
        if(row_counts[j1]>=r):
            valid=valid-set([(j1,k) for k in range(d)])
    return y



def compute_cost(y,phi,gamma):
    cost = 0.0;
    for j in range(len(y)):
        for k in range(len(y[0])):
            cost+=y[j][k]*phi[j][k]
    print("Contribution of phi: %f", cost)
    col_counts = np.sum(y, axis=0)
    print("Average col. count: %f", np.mean(col_counts))
    print("Contribution of col_counts: %f", gamma*np.sum(col_counts*(col_counts-1))/2)
    return cost+gamma*np.sum(col_counts*(col_counts-1))/2;

#This function optimizes the subspace embedding

def combine_heuristic(x,concepts,nu,r,gamma):
    print("Optimizing subspace")
    phi = compute_phi(x,concepts,nu)
    print("Average of phi= %f", np.mean(phi))
    y1 = heuristic1(phi,r,gamma)
    y2 = heuristic2(phi,r,gamma)
    cost1 = compute_cost(y1,phi,gamma)
    cost2 = compute_cost(y2,phi,gamma)
    cost3 = np.inf
    print("Cost of heuristic 1: %f", cost1)
    print("Cost of heuristic 2: %f", cost2)
    #print("Cost of heuristic 3: %f", cost3)
    if(cost1<=cost2 and cost1 <=cost3):
        return y1
    elif(cost2<=cost1 and cost2 <=cost3):
        return y2

