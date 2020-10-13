class Options:
    def __init__(self):

        # ------------------- Global Parameters  -----------------------------
        self.ver = "neurips"

        self.context = "total"  # Choices are "total" or "left-right"

        self.cpu_gpu = "cpu"  # choice are "cpu", "gpu"
        self.debug = "False"
        self.choice = "random"

        # ------------------- Parameters for Generating Features on FIGER dataset  -----------------------------
        self.raw_input_dir = "../data_preprocessed"
        self.train_data_file = "train_split_90.json"
        self.val_data_file = "val_split_10.json"
        self.test_data_file = "test.json"
        self.feature_output_dir = "../feature_vectors_and_class_labels"
        self.feature_dim = 300
        self.context_aggregation_scheme = "average"
        self.glove_embedding_dir = "../glove_embedding"
        self.with_non_leaf = True

        # --------- Parameters for Quantum Embedding -----------
        if self.context == "left-right":
            self.qe_input_dir = "../feature_vectors_and_class_labels"
            self.qe_output_dir = "../quantum_embds_left_right"

        if  self.context == "total":
            self.qe_input_dir = "../feature_vectors_and_class_labels"
            self.qe_output_dir = "../quantum_embeddings"

        self.qe_dim = self.d = [300]
        self.gamma =  [1]
        self._lambda = self.nu = [10]
        self.subspace_dim_lower_bound = self.r = [10]
        self.alpha = self.delta = [0.0001]
        self.num_iterations = 11
