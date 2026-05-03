import torch
import torch.nn as nn
import torch.nn.functional as F


class FashionCNN(nn.Module):
    def __init__(self, num_classes: int = 10, embedding_dim: int = 128):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64 * 7 * 7, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))  # 14 -> 7

        x = torch.flatten(x, 1)
        x = self.dropout(x)

        embedding = F.relu(self.fc1(x))
        logits = self.fc2(embedding)

        return logits

    def extract_embedding(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        embedding = F.relu(self.fc1(x))

        return embedding