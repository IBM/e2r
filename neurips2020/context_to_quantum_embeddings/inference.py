import numpy as np
import os
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import optim
import random
import json
from collections import Counter
from torch.utils.data import Dataset, DataLoader, TensorDataset
import model
import pickle
import argparse
import evaluate
test_batch_size=1
gpu_no=0
predict_parent_from_child=False
eval_on_leaf_only=False


parser = argparse.ArgumentParser(description="Code to Inference")

parser.add_argument("-checkpoint_path",help="Path of checkpoint to test", type=str,required=True)
parser.add_argument("-test_data_path",help="Path of test data",type=str,required=True)
parser.add_argument("-concept_embd_path",help="Path of learned concept embeddings",type=str,required=True)
parser.add_argument("-features_dir_path",help="directory path to features path",type=str,required=True)
parser.add_argument("-quantum_emdb_dimension",help="quantum embedding dimension",type=int,default=300)
parser.add_argument("-threshold_level_wise",help="Threshold Parameter (referred as tau in the paper)",type=float, default=0.15)
parser.add_argument("-constant_to_divide",help="Threshold Parameter (referred as delta in the paper)",type=int, default=5)


args = parser.parse_args()
checkpoint_path = args.checkpoint_path
threshold_level_wise=args.threshold_level_wise
constant_to_divide=args.constant_to_divide
test_data_path=args.test_data_path
file_quantam_concept_embd=args.concept_embd_path
features_dir_path=args.features_dir_path
quantum_emdb_dimension=args.quantum_emdb_dimension

with open(test_data_path, 'r') as filehandle:
     test_data = json.load(filehandle)

for i in range(len(test_data)):
    test_data[i]['entity_id']=i
    test_data[i]['mentionid'] = i



infile = open(file_quantam_concept_embd,'rb')
quantam_concept_embds = pickle.load(infile)
infile.close()



words_train=Counter()

test_left_context=[]
test_right_context=[]
test_labels=[]
test_entity=[]


with open(os.path.join(features_dir_path,'token_list.json'), 'r') as filehandle:
    words_train = json.load(filehandle)


with open(os.path.join(features_dir_path,'with_non_leaf_type_name_train_split.json'), 'r') as filehandle:
    total_node_list = json.load(filehandle)

with open(os.path.join(features_dir_path,'with_non_leaf_leaf_node_list_train_split.json'), 'r') as filehandle:
    leaf_node_list = json.load(filehandle)

non_leaf_node_list=total_node_list[len(leaf_node_list):]

no_total_nodes=len(total_node_list)
no_leaf_nodes = len(leaf_node_list)

with open(os.path.join(features_dir_path,'with_non_leaf_parent_list_train_split.json'), 'r') as filehandle:
    parent_of = json.load(filehandle)

childern_of_root = [i for i, x in enumerate(parent_of) if x == -1]



# create word to index dictionary and reverse
word2idx_train = {o:i for i,o in enumerate(words_train)}
UNK_index=word2idx_train['<UNK>']


for instance in test_data:

    if instance['left_context']==[]:
        instance['left_context']=['<PAD>']

    if instance['right_context'] == []:
        instance['right_context'] = ['<PAD>']

    instance['left_context']= [word2idx_train.get(token, UNK_index) for token in  instance['left_context']]
    instance['right_context'] = [word2idx_train.get(token, UNK_index) for token in instance['right_context']]

class IRDataset(Dataset):
    def __init__(self, test_data):
        self.test_data = test_data

    def __len__(self):
        return len(self.test_data)

    def vectorize(self,tokens, word2idx):
        """Convert tweet to vector
        """
        vec = [self.word2idx.get(token, word2idx_train['<UNK>']) for token in tokens]
        return vec


    def __getitem__(self, index):

        tokens=[]
        tokens.append(self.test_data[index]['left_context'])
        tokens.append(self.test_data[index]['right_context'])

        entity=[]

        entity.append(self.test_data[index]['entity_id'])
        entity.append(self.test_data[index]['entity_id'])

        predicted_labels=self.test_data[index]['label']


        mentionid=[]
        mentionid.append(self.test_data[index]['mentionid'])
        mentionid.append(self.test_data[index]['mentionid'])
        return tokens,entity,mentionid,predicted_labels


def my_collate_fn(old_data):
    """This function will be used to pad the questions to max length
       in the batch and transpose the batch from
       batch_size x max_seq_len to max_seq_len x batch_size.
       It will return padded vectors, labels and lengths of each tweets (before padding)
       It will be used in the Dataloader
    """
    data=[]
    for i in range(len(old_data)):
        data.append((old_data[i][0][0],old_data[i][1][1],old_data[i][2][1],0,old_data[i][3]))
        data.append((old_data[i][0][1],old_data[i][1][1],old_data[i][2][1],1,old_data[i][3]))

    data.sort(key=lambda x: len(x[0]), reverse=True)

    lens = [len(sent) for sent, entity_id , mentionid, left_right,labels in data]
    entity_ids = []
    mentionids=[]
    left_rights=[]
    labels=[]
    padded_sents = torch.zeros(len(data), max(lens)).long()
    for i, (sent, entity_id,mentionid,left_right,label) in enumerate(data):
        padded_sents[i, :lens[i]] = torch.LongTensor(sent)
        entity_ids.append(entity_id)
        mentionids.append(mentionid)
        left_rights.append(left_right)
        labels.append(label)

    padded_sents = padded_sents.transpose(0, 1)
    return padded_sents, torch.FloatTensor(entity_ids),mentionids, torch.Tensor(left_rights), torch.tensor(lens),labels


test_dataset = IRDataset(test_data)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=1,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    collate_fn=my_collate_fn,
    worker_init_fn=None,
)


model_vocab_size=len(words_train)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')   # 'cpu' in this case
print("device {}".format(device))


model = model.BiLSTM(lstm_layer=1, vocab_size=model_vocab_size, hidden_dim=100,quant_embedding_dim=quantum_emdb_dimension,device=device)
checkpoint = torch.load(os.path.join(checkpoint_path), map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model=model.to(device)
model.eval()

quantam_concept_embds=np.array(quantam_concept_embds)

if predict_parent_from_child == False:
    for internal_node in non_leaf_node_list:
        internal_node_index=total_node_list.index(internal_node)
        indices = [i for i, x in enumerate(parent_of) if x == internal_node_index]
        for i in indices:
            quantam_concept_embds[internal_node_index]=np.logical_or(quantam_concept_embds[internal_node_index],quantam_concept_embds[i])


def is_only_leaf(label_list):
    is_leaf = False
    is_non_leaf = False
    for label in gt_labels[0]:
        if label in leaf_node_list:
            is_leaf=True
        else:
            label_index = total_node_list.index(label)
            if label_index not in parent_list:
                is_non_leaf = True

    if is_leaf == True and is_non_leaf == False:
        return 0
    if is_leaf == True and is_non_leaf == True:
        return 1

    if is_leaf == False and is_non_leaf == True:
        return 2


def predict_labels(score,threshold,constant_to_divide):

    score_level_1=score[childern_of_root]
    max_level_1_index=childern_of_root[np.argmax(score_level_1)]
    score_max_level_1=score[max_level_1_index]

    best_level_1_node = total_node_list[max_level_1_index]

    predicted_labels_id=[]

    predicted_labels_id.append(max_level_1_index)

    person_index=total_node_list.index('person')
    organization_index=total_node_list.index('organization')


    childern_of_max_level_1 = [i for i, x in enumerate(parent_of) if x == max_level_1_index]

    if childern_of_max_level_1 == []:
        return predicted_labels_id, "root",0.0, best_level_1_node,score_max_level_1


    score_level_2=score[childern_of_max_level_1]
    max_level_2_index=childern_of_max_level_1[np.argmax(score_level_2)]
    best_level_2_node = total_node_list[max_level_2_index]

    score_max_level_2=score[max_level_2_index]

    if len(childern_of_max_level_1) > 10:
        constant = constant_to_divide
    else:
        constant = 1

    if (score_max_level_1-score_max_level_2)/score_max_level_1 < threshold/constant:
        predicted_labels_id.append(max_level_2_index)

    return predicted_labels_id, best_level_1_node, score_max_level_1,best_level_2_node,score_max_level_2


def predict_labels_simple(score,threshold):
    predicted_labels_id = np.argwhere(score > threshold)
    # print(predicted_labels_id.transpose())

    predicted_labels_id = list(predicted_labels_id[:, 0])

    new_predicted_labels_id = set()
    for id in predicted_labels_id:
        new_predicted_labels_id.add(id)
        if parent_of[id] != -1 and predict_parent_from_child == True:
            new_predicted_labels_id.add(parent_of[id])

    predicted_labels_id = list(new_predicted_labels_id)

    return predicted_labels_id,0,0,0,0


complete_gt_labels=[]
complete_predicted_labels=[]
results=[]

print("Starting Inference at threshold level wise {} constant to divide {}".format(threshold_level_wise,constant_to_divide))
true_and_prediction = []

ground_truth_dict={}
prediction_dict={}
total_gt_label_len=[]
total_predicted_label_len=[]
gt_labels_ids=[]
gt_labels_ids_length=[]
for i in range(len(test_data)):
    gt_labels_ids.append([total_node_list.index(label) for label in test_data[i]['label']])
    gt_labels_ids_length.append(len(test_data[i]['label']))

for epoch in range(1):
    for batch_idx, (context,entity,ids,left_right,lens,gt_labels) in enumerate(test_data_loader):

        gt_labels_id = [total_node_list.index(i) for i in gt_labels[0]]
        parent_list=[parent_of[total_node_list.index(label)] for label in gt_labels[0]]

        if eval_on_leaf_only == True:
            is_leaf=is_only_leaf(gt_labels[0])
            if is_leaf != 0:
                continue
        predict_quant_embd,id_list = model.forward(context.to(device), lens.to(device),ids,left_right)
        predict_quant_embd=predict_quant_embd.data.cpu().numpy()
        mask_matrix=np.multiply(quantam_concept_embds, predict_quant_embd)
        normalizing_constant=np.sum(np.abs(predict_quant_embd)**2,axis=-1)
        score=(np.sum(np.abs(mask_matrix) ** 2, axis=-1))/normalizing_constant

        predicted_labels_id,best_level_1_node,best_level_1_score,best_level_2_node,best_level_2_score=predict_labels(score,threshold_level_wise,constant_to_divide)
        predicted_labels=[total_node_list[i] for i in predicted_labels_id]
        temp_dict = {}
        temp_dict['gt_label'] = gt_labels[0]
        temp_dict['predicted_label'] = predicted_labels
        temp_dict['best_level_1_node']=best_level_1_node
        temp_dict['best_level_2_node'] = best_level_2_node
        temp_dict['best_level_1_score']=best_level_1_score
        temp_dict['best_level_2_score']=best_level_2_score


        total_gt_label_len.append(len(gt_labels[0]))
        total_predicted_label_len.append(len(predicted_labels))


        p,r,f=evaluate.loose_macro([(gt_labels_id, predicted_labels_id)])
        p1,r1,f1=evaluate.strict([(gt_labels_id, predicted_labels_id)])
        true_and_prediction.append((gt_labels_id, predicted_labels_id))

print("strict (p,r,f1):",evaluate.strict(true_and_prediction))
print("loose micro (p,r,f1):",evaluate.loose_micro(true_and_prediction))
print("loose macro (p,r,f1):",evaluate.loose_macro(true_and_prediction))