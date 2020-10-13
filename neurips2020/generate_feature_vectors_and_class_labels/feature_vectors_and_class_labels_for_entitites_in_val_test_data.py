#----------------------------------------------------------------------------------------------------
# This essentially generated context [left_ctx, right_ctx] for each
# mention of entity in each sentence of the validation and test set and then converts that context into
# a glove context vector.
#----------------------------------------------------------------------------------------------------
from generate_feature_vectors_and_class_labels.options import Options
import json
import os
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import preprocess_string
import numpy as np
import scipy as sp
CUSTOM_FILTERS = [lambda x: x.lower(), strip_punctuation]
my_options = Options()
dataset='val'
#-------------------------------------------------------
# Supporting Functions
#-------------------------------------------------------
def clean_type_hierarchy(complete_type):

    complete_type.discard('/religion/religion')
    complete_type.discard('/computer')
    complete_type.add('/computer_science')
    complete_type.discard('/computer/algorithm')
    complete_type.add('/computer_science/algorithm')
    complete_type.discard('/computer/programming_language')
    complete_type.add('/computer_science/programming_language')
    complete_type.discard('/government/government')
    complete_type.add('/government/administration')
    return complete_type
#-------------------------------------------------------
def generate_conflicting_labels_dict():
    conflicting_labels_dict = {}
    conflicting_labels_dict['/computer'] = '/computer_science'
    conflicting_labels_dict['/computer/algorithm'] = '/computer_science/algorithm'
    conflicting_labels_dict['/computer/programming_language'] = '/computer_science/programming_language'
    conflicting_labels_dict['/government/government'] = '/government/administration'
    return conflicting_labels_dict
#-------------------------------------------------------
def extract_leaf_and_internal_nodes(complete_type):
    for type in complete_type:
        type_path = type.split('/')[1:]
        leaf_nodes.add(type_path[-1])

        for i in range(len(type_path) - 1):
            internal_nodes.add(type_path[i])

    leaf_nodes_list = list(leaf_nodes)

    for node in leaf_nodes_list:
        if node in internal_nodes:
            leaf_nodes.remove(node)

    print(leaf_nodes)
    print(len(leaf_nodes))

    leaf_nodes_list = list(leaf_nodes)
    leaf_nodes_list.sort()

    internal_nodes_list = list(internal_nodes)
    internal_nodes_list.sort()

    total_nodes_list = leaf_nodes_list + internal_nodes_list
    parent_of = [0] * (len(leaf_nodes_list) + len(internal_nodes_list))
    is_leaf_node_list = [1] * len(leaf_nodes_list) + [0] * len(internal_nodes_list)

    for type in complete_type:
        type_path = type.split('/')[1:]
        if len(type_path) == 1:
            index = total_nodes_list.index(type_path[0])
            parent_of[index] = -1
        else:
            for i in range(1, len(type_path)):
                if type_path[i] in leaf_nodes_list:
                    index = total_nodes_list.index(type_path[i])
                    parent_of[index] = total_nodes_list.index(type_path[i - 1])

    return leaf_nodes_list, internal_nodes_list, total_nodes_list, parent_of, is_leaf_node_list
#-------------------------------------------------------
def generate_context_embedding(context_words_list, token_feature_vectors, token_list, method="average"):
    if context_words_list == []:
        return token_feature_vectors[-2]
    else:
        context_embd_list = []
        no_context_words = 0
        for word in context_words_list:
            no_context_words += 1
            if word in token_list:
                index = token_list.index(word)
                context_embd_list.append(token_feature_vectors[index])
            else:
                context_embd_list.append(token_feature_vectors[-1])

        if method == 'average':
            context_embd = sum(context_embd_list) / no_context_words

        return context_embd
#--------------------------------------------------------
# Generating Type Labels Lists
#--------------------------------------------------------
if dataset=='val':
    with open(os.path.join(my_options.raw_input_dir,my_options.val_data_file)) as json_file:
        sentences_val = json.load(json_file)
else:
    with open(os.path.join(my_options.raw_input_dir,my_options.test_data_file)) as json_file:
        sentences_val = json_file.readlines()


complete_type=set()
conflicting_labels_dict={}
internal_nodes=set()
leaf_nodes=set()
#-----------------------------------------------------------------
# ----collecting all type paths across all mentions of entities
#-----------------------------------------------------------------
if my_options.with_non_leaf:
    with open(os.path.join(my_options.feature_output_dir, 'with_non_leaf_type_name_train_split.json'), 'r') as filehandle:
        total_nodes_list = json.load(filehandle)
else:
    with open(os.path.join(my_options.feature_output_dir, 'without_non_leaf_type_name_train_split.json'), 'r') as filehandle:
        total_nodes_list = json.load(filehandle)

'''
for sentence in sentences_val:
    for mentions in json.loads(sentence)['mentions']:
        for label in mentions['labels']:
            complete_type.add(label)

complete_type = clean_type_hierarchy(complete_type)
conflicting_labels_dict = generate_conflicting_labels_dict()

leaf_nodes_list, internal_nodes_list, total_nodes_list, parent_of, is_leaf_node_list = extract_leaf_and_internal_nodes(complete_type)
'''
#--------------------------------------------------------
# Generating Type Labels and Context Feature Vectors for Entities
#--------------------------------------------------------
entities=[]

#entity_type_matrix = []
entities_labels=[]

entities_total_context_feature_vectors = []
entities_left_context_feature_vectors = []
entities_right_context_feature_vectors = []
entities_left_right_context_feature_vectors = []

val_data=[]


#count_only_non_leaf = 0
#count_both_leaf_and_non_leaf = 0

token_feature_vectors=np.load(os.path.join(my_options.feature_output_dir,'token_embedding_300d.npy'))

with open(os.path.join(my_options.feature_output_dir,'token_list.json'), 'r') as filehandle:
    token_list = json.load(filehandle)

token_dict={}

for idx, token in enumerate(token_list):
    token_dict[token] = idx

for index, sentence in enumerate(sentences_val):
    sentence=json.loads(sentence)
    if index%100000==0:
        print("#sentences processed so far ", index)
    # if index != 50607:
    #     continue
    for mention in sentence['mentions']:
        example_dict={}
        # --------------------------------------------------------------
        # -----Generating context vectors for entities mentions ------------
        # --------------------------------------------------------------
        start_index=mention['start']
        end_index = mention['end']
        if start_index==end_index:
            continue
        entity='_'.join(sentence['tokens'][start_index:end_index])

        total_context_words_no_punctuations = []
        left_context_words_no_punctuations = []
        right_context_words_no_punctuations = []
        #---------------------------------------------------------------------------
        # If we have to include entity also in the left and right context then uncomment
        # following two lines and comment above two lines
        # ---------------------------------------------------------------------------
        #left_context_words.extend(sentence['tokens'][:start_index])
        #right_context_words.extend(sentence['tokens'][end_index:])

        # left_context_words.extend(sentence['tokens'][:end_index])
        # right_context_words.extend(sentence['tokens'][start_index:])
        # total_context_words.extend(sentence['tokens'][:start_index])
        # total_context_words.extend(sentence['tokens'][end_index:])

        total_ctx_embed = np.zeros((my_options.feature_dim))
        left_ctx_embed = np.zeros((my_options.feature_dim))
        right_ctx_embed = np.zeros((my_options.feature_dim))
        for idx, word in enumerate(sentence['tokens']):
            new_word = preprocess_string(word, CUSTOM_FILTERS)
            if new_word == []:
                continue
            else:
                for temp_word in new_word:
                    # if temp_word in token_list:
                    #     index = token_list.index(temp_word)
                    #     temp_word_embed = token_feature_vectors[index]
                    # else:
                    #     temp_word_embed = token_feature_vectors[-1]

                    temp_word_embed = token_feature_vectors[token_dict.get(temp_word, -1)]

                    if idx < start_index:
                        left_context_words_no_punctuations.append(temp_word)
                        total_context_words_no_punctuations.append(temp_word)
                        left_ctx_embed = left_ctx_embed + temp_word_embed
                        total_ctx_embed = total_ctx_embed + temp_word_embed

                    elif idx >= start_index and idx < end_index:
                        left_context_words_no_punctuations.append(temp_word)
                        right_context_words_no_punctuations.append(temp_word)
                        left_ctx_embed = left_ctx_embed + temp_word_embed
                        right_ctx_embed = right_ctx_embed + temp_word_embed

                    else:
                        right_context_words_no_punctuations.append(temp_word)
                        total_context_words_no_punctuations.append(temp_word)
                        right_ctx_embed = right_ctx_embed + temp_word_embed
                        total_ctx_embed = total_ctx_embed + temp_word_embed

        if len(left_context_words_no_punctuations) > 0:
            left_ctx_embed = left_ctx_embed/ len(left_context_words_no_punctuations)
        if len(right_context_words_no_punctuations)>0:
            right_ctx_embed = left_ctx_embed / len(right_context_words_no_punctuations)
        if len(total_context_words_no_punctuations) >0:
            total_ctx_embed = left_ctx_embed / len(total_context_words_no_punctuations)
        # total_ctx_embed = generate_context_embedding(total_context_words_no_punctuations, token_feature_vectors, token_list, method="average")
        # left_ctx_embed = generate_context_embedding(left_context_words_no_punctuations, token_feature_vectors, token_list, method="average")
        # right_ctx_embed = generate_context_embedding(right_context_words_no_punctuations, token_feature_vectors, token_list, method="average")
        left_right_ctx_embed = np.concatenate((left_ctx_embed, right_ctx_embed))
        example_dict['left_context'] = left_context_words_no_punctuations
        example_dict['right_context'] = right_context_words_no_punctuations

        # --------------------------------------------------------------
        # -----Generating type labels for entities mentions ------------
        # --------------------------------------------------------------
        labels = mention['labels']

        sentence_label = [0] * len(total_nodes_list)
        #sentence_label = [0] * len(leaf_nodes_list)
        #----------------------------------------
        if '/religion/religion' in labels:
            labels.remove('/religion/religion')
        labels=[conflicting_labels_dict.get(label, label) for label in labels]
        label_list = []
        for ind, label in enumerate(labels):
            index = label.rindex('/')
            label = label[index + 1:]
            label_list.append(label)
        # ----------------------------------------
        #to_add = True
        #is_leaf_label = False
        #is_non_leaf_label = False
        #only_leaf_label_list = []
        # for ind, label in enumerate(labels):
        #     index=label.rindex('/')
        #     label=label[index+1:]
        #     position=total_nodes_list.index(label)
        #     if is_leaf_node_list[position]==1:
        #         sentence_label[position]=1
        #         is_leaf_label = True
        #         only_leaf_label_list.append(label)
        #     else:
        #         tmp_labels=labels.copy()
        #         tmp_labels.pop(ind)
        #         tmp_labels=[ll[1:ll.rindex('/')] for ll in tmp_labels]
        #         condition=any(label == string for string in tmp_labels)
        #         if condition==False:
        #             #sentence_label[position]=1
        #            is_non_leaf_label = True
        # example_dict['label']=only_leaf_label_list
        # if is_leaf_label == True and is_non_leaf_label == False:
        #     entities.append(entity)
        #
        #     entities_labels.append(label_list)
        #     entity_type_matrix.append(sentence_label)
        #
        #     entities_total_context_feature_vectors.append(total_ctx_embed)
        #     # entities_left_context_feature_vectors.append(left_ctx_embed)
        #     # entities_right_context_feature_vectors.append(right_ctx_embed)
        #     entities_left_right_context_feature_vectors.append(left_right_ctx_embed)
        #
        #     training_data.append(example_dict)

        # if is_leaf_label == True and is_non_leaf_label == True:
        #     count_both_leaf_and_non_leaf += 1
        #
        # if is_leaf_label == False and is_non_leaf_label == True:
        #     count_only_non_leaf += 1
        #
        # assert (is_leaf_label==True or is_non_leaf_label==True)
        entities.append(entity)
        example_dict['label'] = label_list
        val_data.append(example_dict)
        entities_total_context_feature_vectors.append(total_ctx_embed)
        # entities_left_context_feature_vectors.append(left_ctx_embed)
        # entities_right_context_feature_vectors.append(right_ctx_embed)
        entities_left_right_context_feature_vectors.append(left_right_ctx_embed)

#--------------------------------------------------------------------------
# Label and embedding generation part is over. Saving the data now.
#--------------------------------------------------------------------------
print("# entities with only leaf labels {}".format(len(entities)))
print("Size of Entity-Total-Ctx-Feature-Vector Matrix {}".format(len(entities_total_context_feature_vectors)))
print("Size of Entity-Left-Right-Ctx-Feature-Vector Matrix {}".format(len(entities_left_right_context_feature_vectors)))

print("Saving of the Data started")

if dataset=="val":

    np.save(os.path.join(my_options.feature_output_dir, 'total_context_feature_vector_matrix_val_split_300d.npy'),
            np.array(entities_total_context_feature_vectors))
    np.save(os.path.join(my_options.feature_output_dir , 'left_right_context_feature_vector_matrix_val_split_300d.npy'),
            np.array(entities_left_right_context_feature_vectors))

    with open(os.path.join(my_options.feature_output_dir,'entities_name_val_split.json'), 'w') as filehandle:
        json.dump(entities, filehandle)

    with open(os.path.join(my_options.feature_output_dir,'val_data_processed.json'), 'w') as filehandle:
         json.dump(val_data, filehandle)
else:

    np.save(os.path.join(my_options.feature_output_dir, 'total_context_feature_vector_matrix_test_300d.npy'),
            np.array(entities_total_context_feature_vectors))
    np.save(os.path.join(my_options.feature_output_dir, 'left_right_context_feature_vector_matrix_test_300d.npy'),
            np.array(entities_left_right_context_feature_vectors))

    with open(os.path.join(my_options.feature_output_dir,'entities_name_test.json'), 'w') as filehandle:
        json.dump(entities, filehandle)

    with open(os.path.join(my_options.feature_output_dir,'test_data_processed.json'), 'w') as filehandle:
         json.dump(val_data, filehandle)

print("Saving completed successfully!")