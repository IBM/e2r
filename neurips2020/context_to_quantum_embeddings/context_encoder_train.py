from nltk import word_tokenize
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
from datetime import date
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import model
import pickle
import time
import argparse
import timeit

gpu_no=0

parser = argparse.ArgumentParser(description="Code to train model to which transforms context Vectors into quantum embedding")

parser.add_argument("-features_dir_path",type=str, help="directory path to features path",default='../feature_vectors_and_class_labels',required=True)
parser.add_argument("-entity_embd_path",help="Path of learned entity embeddings",type=str,required=True,default='../quantum_embds/entities_embd.pkl')
parser.add_argument("-concept_embd_path",type=str,help="Path of learned concept embeddings",required=True,default='../quantum_embds/concepts_embd.pkl')
parser.add_argument("-checkpoint_save_dir",type=str,help="Path to save checkpoints", default='../checkpoints/')
parser.add_argument("-d",type=int,help="quantum embedding dimension", default=300)
parser.add_argument("-context_embd_dim",type=int,help="context embedding dimension", default=200)
parser.add_argument("-lr",type=float, help="Learning rate",default=0.0001)
parser.add_argument("-batch_size",type=int, help="Batch size",default=1000)



args = parser.parse_args()
features_path=args.features_dir_path
entity_embd_path=args.entity_embd_path
concept_embd_path=args.concept_embd_path
checkpoint_save_dir=args.checkpoint_save_dir
learning_rate=args.lr
quant_embedding_dim=args.d
context_embd_dim=args.context_embd_dim
train_batch_size=args.batch_size

def checkIfDuplicates(listOfElems):
    ''' Check if given list contains any duplicates '''
    for index,elem in enumerate(listOfElems):
        if index%100000 == 0:
            print(index)
        if listOfElems.count(elem) > 1:
            return elem
    return False

print("using entities embedding file ",entity_embd_path)
print("using concepts embedding file ",concept_embd_path)


print("learning rate ",learning_rate)
print("quant embedding dimension ",quant_embedding_dim)
print("context embedding dimension ",context_embd_dim)
print("training batch size ", train_batch_size)

infile = open(entity_embd_path,'rb')
quat_embds = pickle.load(infile)
infile.close()

avg_len_quant_embd=np.count_nonzero(quat_embds)/(quat_embds.shape)[0]
print("Avg. Quantum embd non zero entries ",avg_len_quant_embd)


infile = open(concept_embd_path,'rb')
quantam_concept_embds = pickle.load(infile)
infile.close()

quantam_concept_embds=np.array(quantam_concept_embds)


label_conflict_dict={}

label_conflict_dict['/computer']='/computer_science'
label_conflict_dict['/computer/algorithm']='/computer_science/algorithm'
label_conflict_dict['/computer/programming_language']='/computer_science/programming_language'

label_conflict_dict['/government/government']='/government/administration'

words_train=Counter()

train_left_context=[]
train_right_context=[]
train_labels=[]
train_entity=[]

total_entities=Counter()

print("Loading training data")

with open(os.path.join(features_path,'with_non_leaf_lstm_training_data.json'), 'r') as filehandle:
    training_data = json.load(filehandle)

with open(os.path.join(features_path, 'token_list.json'), 'r') as filehandle:
    words_train = json.load(filehandle)

with open(os.path.join(features_path,'with_non_leaf_type_name_train_split.json'), 'r') as filehandle:
    total_node_list = json.load(filehandle)

with open(os.path.join(features_path,'with_non_leaf_leaf_node_list_train_split.json'), 'r') as filehandle:
    leaf_node_list = json.load(filehandle)

print("Loading training data done")

# create word to index dictionary
word2idx_train = {o:i for i,o in enumerate(words_train)}

print("Loading Glove Embedding")

embeddings_matrix=np.load(os.path.join(features_path,'token_embedding_300d.npy'))

print("Loading Glove Embedding Done")

ground_truth_labels=[]

UNK_index=word2idx_train['<UNK>']

for index,instance in enumerate(training_data):

    if instance['left_context']==[]:
        instance['left_context']=['<PAD>']

    if instance['right_context'] == []:
        instance['right_context'] = ['<PAD>']

    instance['left_context']= [word2idx_train.get(token, UNK_index) for token in  instance['left_context']]
    instance['right_context'] = [word2idx_train.get(token, UNK_index) for token in instance['right_context']]
    instance['label'] = [total_node_list.index(token) for token in instance['label']]

    ground_truth_labels.append(instance['label'])


    instance['entity_id'] = index
    instance['mentionid'] = index

class IRDataset(Dataset):
    def __init__(self, training_data):
        self.training_data = training_data



    def __len__(self):
        return len(self.training_data)

    def vectorize(self,tokens, word2idx):
        """Convert tweet to vector
        """
        vec = [self.word2idx.get(token, word2idx_train['<UNK>']) for token in tokens]
        return vec


    def __getitem__(self, index):

        tokens=[]
        tokens.append(self.training_data[index]['left_context'])
        tokens.append(self.training_data[index]['right_context'])

        entity=[]


        entity.append(self.training_data[index]['entity_id'])
        entity.append(self.training_data[index]['entity_id'])

        ground_truth_labels=self.training_data[index]['label']




        mentionid=[]
        mentionid.append(self.training_data[index]['mentionid'])
        mentionid.append(self.training_data[index]['mentionid'])
        return tokens,entity,mentionid,ground_truth_labels


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
    labels = []
    entity_ids = []
    mentionids=[]
    left_rights=[]
    padded_sents = torch.zeros(len(data), max(lens)).long()
    for i, (sent, entityid,mentionid,left_right,label) in enumerate(data):
        padded_sents[i, :lens[i]] = torch.LongTensor(sent)
        entity_ids.append(entityid)
        labels.append(label)
        mentionids.append(mentionid)
        left_rights.append(left_right)

    padded_sents = padded_sents.transpose(0, 1)
    return padded_sents, entity_ids,mentionids, torch.Tensor(left_rights), torch.tensor(lens),labels


train_dataset = IRDataset(training_data)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    sampler=None,
    batch_sampler=None,
    num_workers=1,
    pin_memory=False,
    drop_last=True,
    timeout=0,
    collate_fn=my_collate_fn,
    worker_init_fn=None,
)


model_vocab_size=len(words_train)
#device = torch.device("cuda:{}".format(gpu_no))
device = torch.device('cuda')   # 'cpu' in this case
print(device,flush=True)

model = model.BiLSTM(lstm_layer=1, vocab_size=model_vocab_size, hidden_dim=100,quant_embedding_dim=quant_embedding_dim, device=device)

model.emb.weight.data.copy_(torch.from_numpy(embeddings_matrix))
model.emb.weight.requires_grad=False
model=model.to(device)

for name, param in model.named_parameters():
  if 'bias' in name:
     nn.init.constant_(param, 0.0)
  elif 'weight' in name:
      if name=='emb.weight':
          continue
      nn.init.xavier_normal_(param)

optimizer = optim.Adam(model.parameters(), learning_rate)
loss_fuction=nn.MSELoss()

print("Starting Training")
for epoch in range(7):
    total_loss_train=0.0
    model.train()
    start_time_epoch=time.time()
    for batch_idx, (context,entities,ids,left_right,lens,gt_labels) in enumerate(train_data_loader):
        start_time_batch = time.time()

        if batch_idx%100==0 and batch_idx !=0:
            print("Epoch {}: training batch no. {} training Loss {}".format(epoch,batch_idx,total_loss_train/batch_idx),flush=True)

        optimizer.zero_grad()

        predict_quant_embd,id_list = model.forward(context.to(device), lens.to(device),entities,left_right)
        gt_quant_embd=quat_embds[id_list]

        loss = loss_fuction(predict_quant_embd, torch.from_numpy(gt_quant_embd).float().to(device))

        loss.backward()
        optimizer.step()
        total_loss_train += loss.item()
        end_time_batch = time.time()

    loss_train = total_loss_train / len(train_data_loader)
    print("Epoch {}: Training Loss {}".format(epoch,loss_train),flush=True)

    if epoch > -1:
        today_date = date.today()
        checkpoint_name = "snapshot_epoch_{}_lr_{}_d_{}.pt".format(epoch,learning_rate,quant_embedding_dim)

        snapshot_prefix = os.path.join(
            checkpoint_save_dir, checkpoint_name
        )
        snapshot_path = snapshot_prefix
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "list_of_loss": loss_train,
            "learning_rate": learning_rate
        }
        print("Saving checkpoint {}".format(snapshot_path),flush=True)
        torch.save(state, snapshot_path)


    end_time_epoch=time.time()
    print("Epoch Time {}".format((end_time_epoch-start_time_epoch)),flush=True)