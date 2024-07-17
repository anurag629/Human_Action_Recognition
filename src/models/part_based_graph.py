import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric

class PartBasedGraphCNN(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(PartBasedGraphCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gcn1 = GCNConv(32, 64)
        
        conv_output_size = 64 * (input_shape[0] // 2) * (input_shape[1] // 2)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        edge_index = torch_geometric.utils.grid_graph(x.shape[2:]).to(x.device)
        x = x.view(x.size(0), -1, x.size(1))
        x = F.relu(self.gcn1(x, edge_index))
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
