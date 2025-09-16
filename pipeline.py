import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import argparse
from utils import *
from models import *


class ActiveLearningPipeline:
    def __init__(self, seed,
                 test_indices,
                 available_pool_indices,
                 train_indices,
                 selection_criterion,
                 dataset_name,
                 model_name,
                 iterations=10,
                 budget_per_iter=30
                 ):
        self.seed = seed
        self.iterations = iterations
        self.budget_per_iter = budget_per_iter
        self.available_pool_indices = available_pool_indices
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.selection_criterion = selection_criterion
        self.dataset_name = dataset_name
        self.model_name = model_name

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
            trained_model = self._train_model()
            accuracy = self._evaluate_model(trained_model)
            accuracy_scores.append(accuracy)
            print(f'Accuracy: {accuracy}')
            print('----------------------------------------')
            self._update_params(trained_model)
        return accuracy_scores

    def _train_model(self):
        """
        Train the model
        """
        model = load_model_wrapper('CNN')
        train_indices = [self.node_id_mapping[node_id] for node_id in self.train_indices]
        return train_model_wrapper(self.feature_vectors[train_indices], self.labels[train_indices], model)

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
        test_indices = [self.node_id_mapping[node_id] for node_id in self.test_indices]
        train_indices = [self.node_id_mapping[node_id] for node_id in self.train_indices]
        if any(idx in train_indices for idx in test_indices):
            raise ValueError('Data leakage detected: test indices are in the train set.')
        preds = trained_model.predict(self.feature_vectors[test_indices])
        return round(np.mean(preds == self.labels[test_indices]), 3)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices_dict_path', type=str, default='indices_dict_part1.pkl')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--budget_per_iter', type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts")

    # model and dataset name
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)

    # decide the initial test/train split
    hp = parser.parse_args()
    with open(hp.indices_dict_path, 'rb') as f:
        indices_dict = pickle.load(f)
    available_pool_indices = indices_dict['available_pool_indices']
    train_indices = indices_dict['train_indices']
    test_indices = indices_dict['test_indices']

    # TODO: add more criterias here
    selection_criteria = ['custom', 'random']
    accuracy_scores_dict = defaultdict(list)

    print(f"\n-----------STARTING ACTIVE LEARNING PIPELINE FOR DATASET {hp.dataset_name} WITH MODEL {hp.model_name}-------------------\n")

    for i, seed in enumerate(range(1, 4)):
        print(f"seed {seed}")
        for criterion in selection_criteria:
            AL_class = ActiveLearningPipeline(seed=seed,
                                              test_indices=test_indices,
                                              available_pool_indices=available_pool_indices,
                                              train_indices=train_indices,
                                              selection_criterion=criterion,
                                              dataset_name=hp.dataset_name,
                                              model_name=hp.model_name,
                                              iterations=hp.iterations,
                                              budget_per_iter=hp.budget_per_iter
                                              )
            
            accuracy_scores_dict[criterion] = AL_class.run_pipeline()
        generate_plot(accuracy_scores_dict)
        print(f"======= Finished iteration for seed {seed} =======")