#----------------------------------------------------
    #It essentially first finds all the tokens across all the sentences in train/dev/test sets
    # and then picks Glove embedding vector for each one of them.
#----------------------------------------------------
from generate_feature_vectors_and_class_labels.options import Options
import json
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import preprocess_string
import numpy as np
import scipy as sp
import os
CUSTOM_FILTERS = [lambda x: x.lower(), strip_punctuation]
my_options = Options()

#-----------------------------------------------------------------------
# Generate a set of all tokens across all sentences in train/dev/test
#----------------------------------------------------------------------
token_set= set()

def extract_tokens(sentences):
    my_set=set()
    for sentence in sentences:
        sentence = json.loads(sentence)
        for token in sentence['tokens']:
            temp_word = preprocess_string(token, CUSTOM_FILTERS)
            if temp_word != []:
                tokens_in_sentence = set(temp_word)
                my_set.update(tokens_in_sentence)
    return my_set

with open(os.path.join(my_options.raw_input_dir,my_options.train_data_file)) as json_file:
    sentences_train = json.load(json_file)
    print('# Training Sentences =', len(sentences_train))
    token_set.update(extract_tokens(sentences_train))
    print('# Tokens in Train Set =', len(token_set))

with open(os.path.join(my_options.raw_input_dir,my_options.val_data_file)) as json_file:
    sentences_val = json.load(json_file)
    print('# Val Sentences =', len(sentences_val))
    token_set.update(extract_tokens(sentences_val))
    print('# Tokens in Train+Val Set =', len(token_set))

with open(os.path.join(my_options.raw_input_dir,my_options.test_data_file)) as json_file:
    sentences_test=[]     #test file formatting is slightly different and hence this line.
    for line in json_file:
        sentences_test.append(json.dumps(json.loads(line)))
    print('# Test Sentences =', len(sentences_test))
    token_set.update(extract_tokens(sentences_test))
    print('# Tokens in Train+Val+test Set =', len(token_set))
#-----------------------------------------------------------------------
# Generate the Glove vector based feature embedding of each token
#-----------------------------------------------------------------------
token_embedding =  np.zeros((len(token_set)+2, my_options.feature_dim))
average = np.zeros(my_options.feature_dim)
token_list =['']*(len(token_set)+2)
token_counter=0
print("Reading Glove Embeddings")
f = open(os.path.join(my_options.glove_embedding_dir,'glove.840B.300d.txt'),'r')
for index, line in enumerate(f):
    if index % 100000 == 0:
        print("No. of Words Processed", index)
    splitLines = line.split(' ')
    word = ''
    for i in range(len(splitLines)-my_options.feature_dim):
        word = word+splitLines[i]
    temp_word = preprocess_string(word, CUSTOM_FILTERS)
    wordEmbedding = np.array([float(value) for value in splitLines[len(splitLines) - my_options.feature_dim:]])
    average = average + wordEmbedding
    if temp_word != []:
        for word in temp_word:
            if word != '' and word in token_set:
                token_embedding[token_counter] = wordEmbedding
                token_list[token_counter] = word
                token_counter += 1
                token_set.remove(word)
print(token_counter," words processed!")
average = average/token_counter
print("scanned the whole glove vocabulory")

token_list[-1] = '<UNK>'
token_list[-2] = '<PAD>'
token_embedding[-1] = average              #embedding for '<UNK>'

print("final token counter =", token_counter)
np.save(os.path.join(my_options.feature_output_dir,'token_embedding_300d.npy'), token_embedding)
with open(os.path.join(my_options.feature_output_dir,'token_list.json'), 'w') as filehandle:
    json.dump(token_list, filehandle)
print("successfully saved the vocabulory of size ", token_counter)

unmatched_token_list = list(token_set)
with open(os.path.join(my_options.feature_output_dir,'unmatched_token_list.json'), 'w') as filehandle:
    json.dump(unmatched_token_list, filehandle)