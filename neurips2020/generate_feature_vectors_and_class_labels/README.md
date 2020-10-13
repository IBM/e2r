
# Code for Generating feature vectors and class labels for training of quantum embedding #


1. **First finds all the tokens across all the sentences in train/dev/test sets and then pick Glove embedding vector for each one of them. Run the following command:**
    
    `python word_vectors_for_tokens_in_data`


2. **To produce features vectors used to generate quantum embeddings, run the following command:**

    `python feature_vectors_and_class_labels_for_entities_in_traing_data.py`
