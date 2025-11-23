"""
Active Learning Pipeline for GNN-based sample selection.
Implements various selection strategies including GNN-based, uncertainty-based, and random sampling.
"""

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
    """
    Main pipeline for active learning experiments.
    Iteratively selects samples from pool, trains models, and evaluates performance.
    """

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
        """
        Initialize AL pipeline.
        
        Args:
            seed: Random seed for reproducibility
            feature_vectors: Input features for all samples
            labels: Ground truth labels
            selection_criterion: Strategy for sample selection (random/margin/entropy/gnn)
            dataset_name: Name of dataset (CIFAR10/IMDB/drybean)
            model_name: Base model architecture
            train_config: Training configuration
            iterations: Number of AL rounds
            budget_per_iter: Fraction of data to add each iteration
            test_ratio: Test set proportion
            val_ratio: Validation set proportion
            model_config: Model-specific configuration
            load_from_pkl: Load previous results
            fine_tune: Whether to fine-tune vs retrain from scratch
        """
        self.seed = seed
        self.iter = 0
        self.rng = np.random.default_rng(self.seed)
        self.feature_vectors = feature_vectors
        self.labels = labels
        self.iterations = iterations
        self.train_config = train_config
        self.selection_criterion = selection_criterion
        self.dataset_name = dataset_name
        self.model_name = model_name

        # Dataset splits configuration
        self.total_size = len(self.labels)
        self.budget_per_iter = budget_per_iter
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

        # Metrics tracking for analysis
        self.nf = list()              # Node homophily
        self.ef = list()              # Edge homophily
        self.uncert_margin = list()   # Margin uncertainties
        self.uncert_gnn = list()      # GNN uncertainties

    def run_pipeline(self):
        """
        Execute the active learning pipeline.
        
        Returns:
            accuracy_scores: List of test accuracies per iteration
        """

        accuracy_scores = []
        bar = tqdm(range(self.iterations), desc='AL Iters')
        for iteration in bar:
            self.iter = iteration
            if len(self.train_indices) / self.total_size > 0.5:
                # raise error if the train set is larger than half the samples
                raise ValueError('The train set is larger than half the samples')
            print(f'Iteration {iteration + 1}/{self.iterations}, Train Size: {len(self.train_indices)}')

            # Train and evaluate
            _, trained_model = self._train_model()
            accuracy = self._evaluate_model(trained_model)

            bar.set_postfix({'Accuracy': accuracy*100})
            accuracy_scores.append(accuracy)
            print(f'Accuracy: {accuracy}')
            print('----------------------------------------')

            # Select new samples for next iteration
            self._update_step(trained_model)
            self.current_iter_ratio += (iteration + 1) / self.iterations

            # Clean up memory if not fine-tuning
            if not self.fine_tune:
                del trained_model
                torch.cuda.empty_cache()
        
        return accuracy_scores
    
    def _split_points(self):
        """
        Initial random split of data into train/val/test/pool sets.
        
        Returns:
            Tuple of indices for (train, val, test, pool)
        """
        all_idx = np.arange(self.total_size)

        # Calculate split sizes
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
        Train or fine-tune the base model.
        
        Returns:
            loss_steps: Training loss history
            trained_model: Trained model instance
        """
        input_dim = self.feature_vectors.shape[1]
        first_round = False

        # Use existing model for fine-tuning or create new one
        if self.fine_tune and (self.current_model is not None):
            model = self.current_model
        else:
            model = load_model_wrapper(self.model_name,
                                    n_classes=self.n_classes,
                                    input_dim=input_dim,
                                    model_config=self.model_config)
            first_round = True

        # Train model
        loss_steps, trained_model = train_deep_model(model,
                                self.feature_vectors[self.train_indices],
                                self.labels[self.train_indices],
                                self.feature_vectors[self.val_indices],
                                self.labels[self.val_indices],
                                self.train_config,
                                self.fine_tune,
                                first_round
                            )
    
        # Store model for next iteration if fine-tuning
        if self.fine_tune:
            self.current_model = trained_model
        else:
            self.current_model = None

        return loss_steps, trained_model

    def _gnn_sampling(self, trained_model, graph_data):
        """
        Select samples using GNN-based uncertainty propagation.
        
        Args:
            trained_model: Current base model
            graph_data: PyG Data object with graph structure
            
        Returns:
            selected_pos: Positions of selected samples in pool
        """

        trained_model = trained_model.to('cpu')
        print("Model back on CPU")

        # Train GNN for label propagation
        gnn_model = self._train_label_propagation_gnn(graph_data)

        # Get GNN predictions for all nodes
        all_nodes_mask = graph_data.train_mask | graph_data.valid_mask | graph_data.test_mask | graph_data.pool_mask
        _, all_probs = validate_gnn(gnn_model, graph_data, all_nodes_mask)
        pool_probs = all_probs[graph_data.pool_mask.cpu()]

        # cleaning up memory
        del gnn_model
        torch.cuda.empty_cache()
        print("Model back on CPU")

        # calculating and normalizing uncertainty scores
        sorted_probs, _ = torch.sort(pool_probs, dim=1, descending=True)
        margins = (sorted_probs[:, 0] - sorted_probs[:, 1]).detach().cpu().numpy()
        uncertainty_scores = -margins
        self.uncert_gnn.append(uncertainty_scores.copy())

        # select top k based on uncertainty GNN score
        budget_n = int(self.budget_per_iter * self.total_size)
        selected_pos = np.argpartition(-uncertainty_scores, budget_n)[:budget_n]

        # convert local pool positions -> global node indices
        pool_global_indices = np.where(graph_data.pool_mask.cpu().numpy())[0]
        selected_global = pool_global_indices[selected_pos]

        # Optional: Visualize selected node neighborhoods
        # Uncomment to enable neighborhood plotting
        # if (self.iter == 0) or ((self.iter + 1) % 5 == 0):
        #     # pick a 10th (practically random node those we picked for the training)
        #     g_idx = selected_global[10] 
        #     all_masks = self.build_set_inclusion_mask(graph_data)
        #     get_gnn_neighborhood(
        #         graph_data=graph_data,  
        #         all_masks=all_masks,
        #         node_idx=g_idx,  
        #         iteration=self.iter,
        #         seed=self.seed,
        #         dataset_name=self.dataset_name
        #     )

        

        return selected_pos
    

    def _random_sampling(self):
        """
        Random baseline: Select samples uniformly at random.
        
        Returns:
            Positions of randomly selected samples
        """

        budget_n = int(self.budget_per_iter * self.total_size)
        return self.rng.choice(len(self.available_pool_indices), budget_n, replace=False)


    def _uncertainty_sampling(self, trained_model):
        """
        Select samples based on model uncertainty (margin/entropy/least_confidence).
        
        Args:
            trained_model: Current base model
            
        Returns:
            selected_pos: Positions of selected samples
        """

        x_pool = self.feature_vectors[self.available_pool_indices]
        y_pool = self.labels[self.available_pool_indices]
        
        # Get model predictions on pool
        _, probs, _ = validate(trained_model, x_pool, y_pool, 
                        self.train_config.batch_size, 
                        self.train_config.device)
        
        # Calculate uncertainty based on criterion
        if self.selection_criterion == 'least_confidence':
            uncertainties = (1 - torch.max(probs, dim=1)[0]).numpy()
        
        elif self.selection_criterion == 'entropy':
            uncertainties = (-torch.sum(probs * torch.log(probs + 1e-10), dim=1)).numpy()
        
        elif self.selection_criterion == 'margin':
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            margins = sorted_probs[:, 0] - sorted_probs[:, 1]
            uncertainties = -margins.numpy()

            # store
            self.uncert_margin.append(uncertainties.copy())  
    
        # select top-k most uncertain
        budget_n = int(self.budget_per_iter * self.total_size)
        selected_pos = np.argpartition(-uncertainties, budget_n)[:budget_n]
        return selected_pos

    def _sampling(self, trained_model):
        """
        Main sampling orchestrator - builds graph and calls appropriate strategy.
        
        Args:
            trained_model: Current base model
            
        Returns:
            new_selected: Indices of newly selected samples
        """

        # Build kNN graph from embeddings
        embeddings = self._compute_embeddings(trained_model)
        edge_index = self._build_graph_from_embeddings(embeddings)
        graph_data = self._as_pyg_data(embeddings, edge_index)

        # Track homophily metrics for analysis
        if self.selection_criterion in ['gnn', 'margin']:
            self.nf.append(node_homophily(edge_index, self.labels))
            self.ef.append(edge_homophily(edge_index, self.labels))

            # Periodic visualization
            if self.iter == 0 or ((self.iter + 1) % 5 == 0):
                plot_tsne_embeddings(
                                    embeddings,
                                    self.labels,
                                    self.train_indices,
                                    self.iter+1,
                                    self.dataset_name,
                                    self.selection_criterion,
                                    self.seed
                                    )
                
        # Execute selection strategy
        if self.selection_criterion in ['least_confidence', 'entropy', 'margin']:
            pos = self._uncertainty_sampling(trained_model)
        elif self.selection_criterion == 'random':
            pos = self._random_sampling()
        elif self.selection_criterion == 'gnn':
            pos = self._gnn_sampling(trained_model, graph_data)
        else:
            raise ValueError("Unknown selection criterion")
        new_selected = np.array(self.available_pool_indices)[pos]
        return new_selected

    def _update_train_indices(self, new_selected_samples):
        """Add newly selected samples to training set."""
        self.train_indices = np.concatenate([self.train_indices, new_selected_samples])

    def _update_available_pool_indices(self, new_selected_samples):
        """Remove selected samples from available pool."""
        self.available_pool_indices = np.setdiff1d(self.available_pool_indices, new_selected_samples)

    def _update_step(self, trained_model):
        """Execute one AL iteration: select samples and update sets."""
        new_selected_samples = self._sampling(trained_model)
        self._update_available_pool_indices(new_selected_samples)
        self._update_train_indices(new_selected_samples)

    def _evaluate_model(self, trained_model):
        """
        Evaluate model on test set.
        
        Returns:
            Test accuracy
        """

        # Data leakage check
        if any(idx in self.train_indices for idx in self.test_indices):
            raise ValueError('Data leakage detected: test indices are in the train set.')
        
        test_acc, _, _ = validate(trained_model, self.feature_vectors[self.test_indices], self.labels[self.test_indices],
                         self.train_config.batch_size, self.train_config.device)
        return test_acc
    
    def _compute_embeddings(self, trained_model: torch.nn.Module) -> torch.Tensor:
        """Extract feature embeddings from base model for entire dataset."""

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
        Create PyG Data object with graph structure and masks.
        
        Returns:
            Data object with node features, edges, labels, and split masks
        """

        device = self.train_config.device
        N = embeddings.size(0)

        # Node features and labels
        x = embeddings.to(device)
        y = torch.as_tensor(self.labels, dtype=torch.long, device=device)

        data = Data(x=x, edge_index=edge_index.to(device), y=y)

        # Create masks for different splits
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
        """
        Train GraphSAGE model for label propagation.
        
        Returns:
            Trained GNN model
        """
                
        in_channels = pyg_data.x.shape[1]
        hidden_channels = 1024
        gnn = GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels, output_dim=self.n_classes)

        gnn = gnn.to(self.train_config.device)
        print("GNN on GPU")
        gnn_loader = NeighborLoader(data=pyg_data, input_nodes=pyg_data.train_mask, batch_size=256, num_neighbors=[15, 10], shuffle=True)
        _, trained_gnn = train_gnn_model(gnn, pyg_data, gnn_loader, self.train_config)
        return trained_gnn



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Pipeline configs
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--budget_per_iter_ratio', type=float, default=0.01)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.05)

    # Training model configs
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--gnn_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts")
    parser.add_argument("--log_every", type=int, default=25)

    # Model and dataset name and path
    parser.add_argument('--model_name', type=str, default="lstm")
    parser.add_argument('--dataset_name', type=str, default="IMDB")

    # LSTM-specific parameters
    parser.add_argument("--embedding_dim", type=int, default=128, help="Dimension for word embeddings")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Dimension for LSTM hidden state")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of LSTM layers")

    # Flag to fine tune and not retrain from scratch
    parser.add_argument("--fine_tune", action='store_true')

    # Flag to load the results from previous runs
    parser.add_argument("--load_from_pkl", action='store_true')

    # Device
    parser.add_argument("--device", type=str, default='cuda')  

    hp = parser.parse_args()

    # Initialize training configuration
    train_config = TrainConfig(hp.epochs, hp.gnn_epochs, hp.lr, hp.weight_decay, hp.momentum, hp.batch_size, hp.ckpt_dir, hp.log_every, hp.device, hp.model_name)

    # Load dataset
    x, y, data_meta = load_dataset_wrapper(hp.dataset_name)

    # Selection strategies to compare
    selection_criteria = ['gnn', 'least_confidence', 'entropy', 'margin', 'random']
    accuracy_scores_dict = defaultdict(list)

    # Model-specific configuration
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

    # Run experiments with different seeds
    for i, seed in enumerate(range(1, 4)):
        print(f"---- SEED {seed} ----")

        if not hp.load_from_pkl:

            margin_node_hom = None
            margin_edge_hom = None
            gnn_node_hom = None
            gnn_edge_hom = None
            margin_uncert = None
            gnn_uncert = None

            # Run each selection strategy
            for criterion in selection_criteria:
                print(f"----  Criterion: {criterion} ----")

                seed_all(seed)
                # Initialize and run pipeline
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

                # store homophily for margin / gnn
                if criterion == 'margin' and len(AL_class.ef) > 0:
                    margin_edge_hom = AL_class.ef
                    margin_node_hom = AL_class.nf
                    margin_uncert = AL_class.uncert_margin

                if criterion == 'gnn' and len(AL_class.ef) > 0:
                    gnn_edge_hom = AL_class.ef
                    gnn_node_hom = AL_class.nf
                    gnn_uncert = AL_class.uncert_gnn

            # Save results
            with open(f'saved_accs/accuracies_seed={seed}_dataset={hp.dataset_name}_finetune={hp.fine_tune}.pkl', 'wb') as f:
                    pickle.dump(accuracy_scores_dict, f)


            # Generate comparison plots
            if (
                margin_edge_hom is not None and
                gnn_edge_hom is not None and
                margin_node_hom is not None and
                gnn_node_hom is not None
            ):
                plot_homophily(margin_edge_hom, gnn_edge_hom, margin_node_hom, gnn_node_hom, 
                               hp.dataset_name, seed=seed)  
                
            if margin_uncert is not None and gnn_uncert is not None:
                plot_uncertainty_side_by_side(margin_uncert, gnn_uncert, 
                                              hp.dataset_name, seed)
        else:
            # Load previous results
            with open(f'saved_accs/accuracies_seed={seed}_dataset={hp.dataset_name}_finetune={hp.fine_tune}.pkl', 'rb') as f:
                accuracy_scores_dict = pickle.load(f)
                
        # Generate accuracy plot
        generate_plot(accuracy_scores_dict, seed, hp.dataset_name)
        print(f"======= Finished iteration for seed {seed} =======")