import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalGraphCNN(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(SpatioTemporalGraphCNN, self).__init__()
        self.conv1 = nn.Conv3d(input_shape[2], 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        conv_output_size = 64 * (input_shape[0] // 4) * (input_shape[1] // 4) * (input_shape[2] // 4)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
