import sys
sys.path.append('../')
import os
from train import train_neurips
from generate_feature_vectors_and_class_labels.options import Options
my_options = Options()

import scipy as sp
from scipy.sparse import csr_matrix
import numpy as np
import pickle as pkl

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d",type=int, default=300)
parser.add_argument("-nu",type=float, default=10)
parser.add_argument("-gamma",type=float, default=1.0)
parser.add_argument("-r",type=float, default=10)
parser.add_argument("-choice", default="random")
parser.add_argument("-delta",type=float, default=0.0001)

args = parser.parse_args()
data_fname = os.path.join(my_options.qe_input_dir,'with_non_leaf_sparse_entity_type_matrix_train_split.npz')

entity_type_matrix = sp.sparse.load_npz(data_fname)

print(np.shape(entity_type_matrix))
print(type(entity_type_matrix))
print(entity_type_matrix[0,:])
np.shape(entity_type_matrix)


if my_options.context=="left-right":
    features = np.load(open(os.path.join(my_options.qe_input_dir,'with_non_leaf_left_right_context_feature_vector_matrix_train_split_300d.npy'), 'rb')).T

else:
    features = np.load(open(os.path.join(my_options.qe_input_dir ,'with_non_leaf_left_right_context_feature_vector_matrix_train_split_300d.npy'),'rb')).T
    features[:int(features.shape[0] / 2), :] = features[:int(features.shape[0] / 2), :] + features[int(features.shape[0] / 2):, :]
    features = features[:int(features.shape[0] / 2), :]


train_neurips(entity_type_matrix.T, d=args.d, nu=args.nu, r=args.r, gamma=args.gamma, iter=my_options.num_iterations, F=features, delta=args.delta)