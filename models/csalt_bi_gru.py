import torch.nn as nn
import torch.nn.functional as F


class CSALT_BiGRU(nn.Module):
    def __init__(self, num_features, num_classes):
        self.layer_1 = nn.GRU(input_size=num_features, hidden_size=650, bidirectional=True)
        self.layer_2_4 = nn.GRU(input_size=650, hidden_size=650, num_layers=3, bidirectional=True)
        self.classifier = nn.GRU(input_size=650, hidden_size=num_classes)

    def forward(self, batch):
        print(batch.shape)
        batch = self.layer_1(batch)
        print("After 1 layer: ", batch)
        batch = self.layer_2_4(batch)
        print("After layer 2_2", batch.shape)
        pred = self.classifier(batch)
        print("classifier: ", batch.shape)
        log_probs = F.log_softmax(pred, dim=1)
        return log_probs
