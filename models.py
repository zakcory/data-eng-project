from torch_geometric.nn import SAGEConv
import torch
import torch_geometric
import torch.nn.functional as F



# TODO: ADD MODELS HERE

# GNN model (for active learning)
class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim, seed):
        super().__init__()
        self.conv1 = SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels)
        self.conv2 = SAGEConv(in_channels=hidden_channels, out_channels=output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x





# loading model function
def load_model_wrapper(model_name):
    # example usage
    if model_name == 'ResNet140':
        pass

# training model function
def train_model_wrapper(X_train, y_train, model):
    pass