import pandas as pd
import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from collections import defaultdict
import argparse
from tqdm import tqdm
import pickle

from utils import *
from models import *
from factories import *


class ActiveLearningPipeline:
    def __init__(self, seed,
                 feature_vectors,
                 labels,
                 selection_criterion,
                 dataset_name,
                 model_name,
                 train_config,
                 iterations=5,
                 budget_per_iter=0.01,
                 test_ratio=0.2,
                 val_ratio=0.05,
                 model_config=None,
                 load_from_pkl=False,
                 fine_tune=False
                 ):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.feature_vectors = feature_vectors
        self.labels = labels
        self.iterations = iterations
        self.total_size = len(self.labels)
        self.budget_per_iter = budget_per_iter
        self.train_config = train_config
        self.selection_criterion = selection_criterion
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.n_classes = len(self.labels.unique())
        print(f"Num. Classes: {self.n_classes}")
        self.model_config = model_config
        self.train_indices, self.val_indices, self.test_indices, self.available_pool_indices = self._split_points()
        self.load_from_pkl = load_from_pkl
        self.fine_tune = fine_tune
        self.current_model = None
        self.current_iter_ratio = 0

    def run_pipeline(self):
        """
        Run the active learning pipeline
        """
        accuracy_scores = []
        bar = tqdm(range(self.iterations), desc='AL Iters')
        for iteration in bar:
            if len(self.train_indices) / self.total_size > 0.5:
                # raise error if the train set is larger than half the samples
                raise ValueError('The train set is larger than half the samples')
            print(f'Iteration {iteration + 1}/{self.iterations}, Train Size: {len(self.train_indices)}')
            _, trained_model = self._train_model()
            accuracy = self._evaluate_model(trained_model)

            bar.set_postfix({'Accuracy': accuracy*100})
            accuracy_scores.append(accuracy)
            print(f'Accuracy: {accuracy}')
            print('----------------------------------------')
            self._update_step(trained_model)
            self.current_iter_ratio += (iteration + 1) / self.iterations

            if not self.fine_tune:
                del trained_model
                torch.cuda.empty_cache()
        
        return accuracy_scores
    
    def _split_points(self):
        all_idx = np.arange(self.total_size)

        train_n = int(round(self.budget_per_iter * self.total_size))
        val_n = int(round(self.val_ratio * self.total_size))
        test_n = int(round(self.test_ratio * self.total_size))

        # splitting point for train / test / val / pool
        test_idx = self.rng.choice(all_idx, size=test_n, replace=False)
        rem = np.setdiff1d(all_idx, test_idx, assume_unique=False)

        val_idx = self.rng.choice(rem, size=val_n, replace=False)
        rem = np.setdiff1d(rem, val_idx, assume_unique=False)

        train_idx = self.rng.choice(rem, size=train_n, replace=False)
        pool_idx  = np.setdiff1d(rem, train_idx, assume_unique=False)

        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), pool_idx.tolist()

    def _train_model(self):
        """
        Train the model
        """
        input_dim = self.feature_vectors.shape[1]
        first_round = False

        if self.fine_tune and (self.current_model is not None):
            model = self.current_model
        else:
            model = load_model_wrapper(self.model_name,
                                    n_classes=self.n_classes,
                                    input_dim=input_dim,
                                    model_config=self.model_config)
            first_round = True

        loss_steps, trained_model = train_deep_model(model,
                                self.feature_vectors[self.train_indices],
                                self.labels[self.train_indices],
                                self.feature_vectors[self.val_indices],
                                self.labels[self.val_indices],
                                self.train_config,
                                self.fine_tune,
                                first_round
                            )
    
        
        if self.fine_tune:
            self.current_model = trained_model
        else:
            self.current_model = None

        return loss_steps, trained_model

    def _gnn_sampling(self, trained_model):
        # 1. Get Embeddings
        embeddings = self._compute_embeddings(trained_model)
        embeddings_cpu = embeddings.detach().cpu().numpy()

        trained_model = trained_model.to('cpu')
        print("Model back on CPU")

        # 2. Get GNN Probabilities
        edge_index = self._build_graph_from_embeddings(embeddings)
        graph_data = self._as_pyg_data(embeddings, edge_index)

        print("Finished building graph, Training GNN....")
        gnn_model = self._train_label_propagation_gnn(graph_data)

        all_nodes_mask = graph_data.train_mask | graph_data.valid_mask | graph_data.test_mask | graph_data.pool_mask
        _, all_probs = validate_gnn(gnn_model, graph_data, all_nodes_mask)
        pool_probs = all_probs[graph_data.pool_mask.cpu()]
        del gnn_model, graph_data
        torch.cuda.empty_cache()
        print("Model back on CPU")

        # 3. Calculate UNCERTAINTY
        sorted_probs, _ = torch.sort(pool_probs, dim=1, descending=True)
        margins = (sorted_probs[:, 0] - sorted_probs[:, 1]).detach().cpu().numpy()
        # Normalize uncertainty score (0 to 1)
        uncertainty_scores = 1.0 - margins

        # 4. Calculate EXPLORATION
        pool_embeddings = embeddings_cpu[self.available_pool_indices]
        train_embeddings = embeddings_cpu[self.train_indices]
        dists = cdist(pool_embeddings, train_embeddings, metric='euclidean')

        # For each pool sample, find its distance to the nearest training point
        min_dists = np.min(dists, axis=1)
        exploration_scores = min_dists  # The farther = more unexplored region

        # Normalize exploration scores (0 to 1)
        if exploration_scores.max() > 0:
            exploration_scores = exploration_scores / exploration_scores.max()
        print("Exploration scores calculated.")

        # 5. Combine with ALPHA
        alpha = self.current_iter_ratio

        # Ensure scores are numpy for combining
        hybrid_scores = uncertainty_scores

        # 6. Select top-k based on the new hybrid score
        budget_n = int(self.budget_per_iter * self.total_size)
        # Use argpartition to find top-k highest scores (we want high uncertainty and high exploration)
        selected_pos = np.argpartition(-hybrid_scores, budget_n)[:budget_n]

        return selected_pos

    def _random_sampling(self):
        """
        Random samplings
        :return:
        new_selected_samples: numpy array, new selected samples
        """
        # Calculate the integer number of samples to select
        budget_n = int(self.budget_per_iter * self.total_size)
        # Use the integer budget_n for the size argument
        return self.rng.choice(len(self.available_pool_indices), budget_n, replace=False)


    def _uncertainty_sampling(self, trained_model):
        x_pool = self.feature_vectors[self.available_pool_indices]
        y_pool = self.labels[self.available_pool_indices]
        
        _, probs, _ = validate(trained_model, x_pool, y_pool, 
                        self.train_config.batch_size, 
                        self.train_config.device)
        
        if self.selection_criterion == 'least_confidence':
            uncertainties = (1 - torch.max(probs, dim=1)[0]).numpy()
        
        elif self.selection_criterion == 'entropy':
            uncertainties = (-torch.sum(probs * torch.log(probs + 1e-10), dim=1)).numpy()
        
        elif self.selection_criterion == 'margin':
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            margins = sorted_probs[:, 0] - sorted_probs[:, 1]
            uncertainties = -margins.numpy()  # Negative so high uncertainty = high value
    
        # Select top-k most uncertain
        budget_n = int(self.budget_per_iter * self.total_size)
        selected_pos = np.argpartition(-uncertainties, budget_n)[:budget_n]
        return selected_pos

    def _sampling(self, trained_model):
        """
        Sampling wrapper
        :return:
        new_selected: list, newly selected samples
        """
        if self.selection_criterion in ['least_confidence', 'entropy', 'margin']:
            pos = self._uncertainty_sampling(trained_model)
        elif self.selection_criterion == 'random':
            pos = self._random_sampling()
        elif self.selection_criterion == 'gnn':
            pos = self._gnn_sampling(trained_model)
        else:
            raise ValueError("Unknown selection criterion")
        new_selected = np.array(self.available_pool_indices)[pos]
        return new_selected

    def _update_train_indices(self, new_selected_samples):
        """
        Update the train indices
        """
        self.train_indices = np.concatenate([self.train_indices, new_selected_samples])

    def _update_available_pool_indices(self, new_selected_samples):
        """
        Update the available pool indices
        """
        self.available_pool_indices = np.setdiff1d(self.available_pool_indices, new_selected_samples)

    def _update_step(self, trained_model):
        """
        Update the pool and train indices
        """
        new_selected_samples = self._sampling(trained_model)
        self._update_available_pool_indices(new_selected_samples)
        self._update_train_indices(new_selected_samples)

    def _evaluate_model(self, trained_model):
        """
        Evaluate the model
        :param trained_model: trained model
        :return: accuracy: float, accuracy of the model on the test set
        """
        if any(idx in self.train_indices for idx in self.test_indices):
            raise ValueError('Data leakage detected: test indices are in the train set.')
        
        test_acc, _, _ = validate(trained_model, self.feature_vectors[self.test_indices], self.labels[self.test_indices],
                         self.train_config.batch_size, self.train_config.device)
        return test_acc
    
    def _compute_embeddings(self, trained_model: torch.nn.Module) -> torch.Tensor:
        """
        Compute embeddings for the entire dataset using the trained model.
        """
        device = self.train_config.device
        model = trained_model.to(device)
        _, _, embeddings = validate(model, self.feature_vectors, self.labels, self.train_config.batch_size, self.train_config.device)

        return embeddings
    
    def _build_graph_from_embeddings(self, embeddings: torch.Tensor, k: int = 10, symmetrize: bool = True):
        """
        Build a k-NN graph (edge_index) from intermediate embeddings and
        optionally return a fully prepared PyG Data object with masks.

        Args:
            embeddings: torch.Tensor [N, D] on CPU (or GPU, but CPU is safer for memory)
            k: neighbors per node
            symmetrize: make graph undirected by mirroring edges

        Returns:
            edge_index: LongTensor [2, E]
        """
        edge_index = build_knn_graph(embeddings, k=k, symmetrize=symmetrize)
        return edge_index

    def _as_pyg_data(self, embeddings: torch.Tensor, edge_index: torch.Tensor):
        """
        Package embeddings + edge_index + labels/masks into a PyG Data object.
        """
        device = self.train_config.device
        N = embeddings.size(0)

        x = embeddings.to(device)
        y = torch.as_tensor(self.labels, dtype=torch.long, device=device)

        data = Data(x=x, edge_index=edge_index.to(device), y=y)

        train_mask = torch.zeros(N, dtype=torch.bool, device=device)
        train_mask[self.train_indices] = True

        val_mask = torch.zeros(N, dtype=torch.bool, device=device)
        val_mask[self.val_indices] = True

        test_mask = torch.zeros(N, dtype=torch.bool, device=device)
        test_mask[self.test_indices] = True

        pool_mask = torch.zeros(N, dtype=torch.bool, device=device)
        pool_mask[self.available_pool_indices] = True

        data.train_mask, data.valid_mask, data.test_mask, data.pool_mask = train_mask, val_mask, test_mask, pool_mask
        return data
    
    def _train_label_propagation_gnn(self, pyg_data):
        in_channels = pyg_data.x.shape[1]
        hidden_channels = 1024
        gnn = GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels, output_dim=self.n_classes)

        gnn = gnn.to(self.train_config.device)
        print("GNN on GPU")
        gnn_loader = NeighborLoader(data=pyg_data, input_nodes=pyg_data.train_mask, batch_size=1024, num_neighbors=[15, 10], shuffle=True)
        _, trained_gnn = train_gnn_model(gnn, pyg_data, gnn_loader, self.train_config)
        return trained_gnn



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # pipeline configs
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--budget_per_iter_ratio', type=float, default=0.01)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    # training model configs
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--gnn_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts")
    parser.add_argument("--log_every", type=int, default=25)
    # model and dataset name and path
    parser.add_argument('--model_name', type=str, default="lstm")
    parser.add_argument('--dataset_name', type=str, default="IMDB")

    parser.add_argument("--embedding_dim", type=int, default=128, help="Dimension for word embeddings")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Dimension for LSTM hidden state")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of LSTM layers")

    # flag to fine tune and not retrain from scratch
    parser.add_argument("--fine_tune", action='store_true')

    # flag to load the results from previous runs
    parser.add_argument("--load_from_pkl", action='store_true')

    # device
    parser.add_argument("--device", type=str, default='cuda')  

    hp = parser.parse_args()

    # add training config into a configurator
    train_config = TrainConfig(hp.epochs, hp.gnn_epochs, hp.lr, hp.weight_decay, hp.momentum, hp.batch_size, hp.ckpt_dir, hp.log_every, hp.device, hp.model_name)

    # X, y (entire set)
    x, y, data_meta = load_dataset_wrapper(hp.dataset_name)


    # TODO: add more criterias here
    #selection_criteria = ['margin', 'random']
    selection_criteria = ['gnn', 'least_confidence', 'entropy', 'margin', 'random']
    accuracy_scores_dict = defaultdict(list)

    model_config = None
    if hp.model_name == 'lstm':
        model_config = {
            'vocab_size': data_meta['vocab_size'],
            'output_size': 2,  # Single output for sigmoid
            'embedding_dim': hp.embedding_dim,
            'hidden_dim': hp.hidden_dim,
            'n_layers': hp.n_layers
        }



    print(f"\n----------- STARTING ACTIVE LEARNING PIPELINE FOR DATASET {hp.dataset_name} WITH MODEL {hp.model_name}, FINETUNING (?) -> {hp.fine_tune} -------------------\n")


    for i, seed in enumerate(range(1, 4)):
        print(f"---- SEED {seed} ----")

        if not hp.load_from_pkl:
            for criterion in selection_criteria:
                print(f"----  Criterion: {criterion} ----")

                seed_all(seed)
                
                AL_class = ActiveLearningPipeline(seed=seed,
                                                feature_vectors=x,
                                                labels=y,
                                                selection_criterion=criterion,
                                                dataset_name=hp.dataset_name,
                                                model_name=hp.model_name,
                                                train_config=train_config,
                                                iterations=hp.iterations,
                                                budget_per_iter=hp.budget_per_iter_ratio,
                                                test_ratio=hp.test_ratio,
                                                val_ratio=hp.val_ratio,
                                                model_config=model_config,
                                                load_from_pkl=hp.load_from_pkl,
                                                fine_tune=hp.fine_tune
                                                )
                
                accuracy_scores_dict[criterion] = AL_class.run_pipeline()

            with open(f'saved_accs/accuracies_seed={seed}_dataset={hp.dataset_name}_finetune={hp.fine_tune}.pkl', 'wb') as f:
                    pickle.dump(accuracy_scores_dict, f)  

        else:
            with open(f'saved_accs/accuracies_seed={seed}_dataset={hp.dataset_name}_finetune={hp.fine_tune}.pkl', 'rb') as f:
                accuracy_scores_dict = pickle.load(f)

        generate_plot(accuracy_scores_dict, seed, hp.dataset_name)
        print(f"======= Finished iteration for seed {seed} =======")