# import torch
import torch.nn as nn
import torch.nn.functional as F


class CSALT_BiGRU(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CSALT_BiGRU, self).__init__()
        self.layer_1 = nn.GRU(input_size=num_features, hidden_size=650, bidirectional=True, num_layers=4, batch_first=True)
        self.classifier = nn.GRU(input_size=1300, hidden_size=num_classes, batch_first=True)

    def forward(self, batch):
        print(batch.shape)
        output, _ = self.layer_1(batch)
        print("After 1 layer: ", output.shape)
        output, _ = self.classifier(output)
        print("classifier: ", output.shape)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs


# if __name__ == '__main__':
#     input = torch.randn(32, 3, 128)
#     model = CSALT_BiGRU(128, 60)
#     print(model)
#     print(model(input))