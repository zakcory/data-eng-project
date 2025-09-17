import pandas as pd
import numpy as np
from collections import defaultdict
import argparse

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
                 iterations=20,
                 budget_per_iter=0.01,
                 test_ratio=0.2,
                 val_ratio=0.05,
                 n_classes=10
                 ):
        self.seed = seed
        self.feature_vectors = feature_vectors
        self.labels = labels
        self.iterations = iterations
        self.budget_per_iter = budget_per_iter
        self.train_config = train_config
        self.selection_criterion = selection_criterion
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.n_classes = n_classes
        self.train_indices, self.val_indices, self.test_indices, self.available_pool_indices = self._split_points()


    def _update_params(self, trained_model):
        pass

    def run_pipeline(self):
        """
        Run the active learning pipeline
        """
        accuracy_scores = []
        for iteration in range(self.iterations):
            if len(self.train_indices) > 600:
                # raise error if the train set is larger than 600 samples
                raise ValueError('The train set is larger than 600 samples')
            print(f'Iteration {iteration + 1}/{self.iterations}')
            loss_steps, trained_model = self._train_model()
            accuracy = self._evaluate_model(trained_model)
            accuracy_scores.append(accuracy)
            print(f'Accuracy: {accuracy}')
            print('----------------------------------------')
            self._update_params(trained_model)
        return accuracy_scores
    
    def _split_points(self):
        total_size = len(self.labels)
        all_idx = np.arange(total_size)

        rng = np.random.default_rng(self.seed)

        train_n = int(round(self.budget_per_iter * total_size))
        val_n = int(round(self.val_ratio * total_size))
        test_n = int(round(self.test_ratio * total_size))

        # splitting point for train / test / val / pool
        test_idx = rng.choice(all_idx, size=test_n, replace=False)
        rem = np.setdiff1d(all_idx, test_idx, assume_unique=False)

        val_idx = rng.choice(rem, size=val_n, replace=False)
        rem = np.setdiff1d(rem, val_idx, assume_unique=False)

        train_idx = rng.choice(rem, size=train_n, replace=False)
        pool_idx  = np.setdiff1d(rem, train_idx, assume_unique=False)

        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), pool_idx.tolist()

    def _train_model(self):
        """
        Train the model
        """
        model = load_model_wrapper(self.model_name, n_classes=self.n_classes)
        # Corrected argument order:
        return train_deep_model(model,
                                    self.feature_vectors[self.train_indices],
                                    self.labels[self.train_indices],
                                    self.feature_vectors[self.val_indices],
                                    self.labels[self.val_indices],
                                    self.train_config)

    def _random_sampling(self):
        """
        Random samplings
        :return:
        new_selected_samples: numpy array, new selected samples
        """
        np.random.seed(self.seed)
        return np.random.choice(list(range(len(self.available_pool_indices))), self.budget_per_iter, replace=False)

    def _custom_sampling(self, trained_model):
        """
        Custom sampling method to be implemented
        :return:
        new_selected_samples: numpy array, new selected samples
        """
        pool_nids = np.array(self.available_pool_indices)
        pool_idx = [self.node_id_mapping[nid] for nid in pool_nids]
        X_pool = self.feature_vectors[pool_idx]
        # predicted probabilities for each class
        probs = trained_model.predict_proba(X_pool)
        # uncertainty = 1 - max predicted class probability
        uncertainties = 1 - np.max(probs, axis=1)
        # pick top-k most uncertain
        selected_pos = np.argpartition(-uncertainties, self.budget_per_iter)[:self.budget_per_iter]
        return selected_pos

    def _sampling(self, trained_model):
        """
        Sampling wrapper
        :return:
        new_selected: list, newly selected samples
        """
        if self.selection_criterion == 'custom':
            pos = self._custom_sampling(trained_model)
        elif self.selection_criterion == 'random':
            pos = self._random_sampling()
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
        
        test_acc = validate(trained_model, self.feature_vectors[self.test_indices], self.labels[self.test_indices],
                         self.train_config.batch_size, self.train_config.device)
        return test_acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # pipeline configs
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--budget_per_iter_ratio', type=float, default=0.01)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    # training model configs
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts")
    parser.add_argument("--log_every", type=int, default=10)
    # model and dataset name and path
    parser.add_argument('--model_name', type=str, default="fraudnet")
    parser.add_argument('--dataset_name', type=str, default="credit-cards")
    # device
    parser.add_argument("--device", type=str, default='cuda')  

    hp = parser.parse_args()

    # add training config into a configurator
    train_config = TrainConfig(hp.epochs, hp.lr, hp.weight_decay, hp.momentum, hp.batch_size, hp.ckpt_dir, hp.log_every, hp.device) 

    # X, y (entire set)
    x, y = load_dataset_wrapper(hp.dataset_name)

    # TODO: add more criterias here
    selection_criteria = ['custom', 'random']
    accuracy_scores_dict = defaultdict(list)

    print(f"\n-----------STARTING ACTIVE LEARNING PIPELINE FOR DATASET {hp.dataset_name} WITH MODEL {hp.model_name}-------------------\n")

    for i, seed in enumerate(range(1, 4)):
        print(f"seed {seed}")
        for criterion in selection_criteria:
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
                                              val_ratio=hp.val_ratio
                                              )
            
            accuracy_scores_dict[criterion] = AL_class.run_pipeline()
        generate_plot(accuracy_scores_dict)
        print(f"======= Finished iteration for seed {seed} =======")