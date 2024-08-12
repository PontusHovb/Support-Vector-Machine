import torch
import torch.nn as nn

class LinearSVM(nn.Module):
    def __init__(self, input_size, num_classes=1, C=1.0):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.C = C

    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

    def hinge_loss(self, output, target):
        zero_vector = torch.zeros_like(output*target.t())
        one_vector = torch.ones_like(output*target.t())
        max_vector = torch.max(zero_vector, one_vector-output*target.t())
        loss = torch.sum(max_vector)
        return loss

    def l2_regularization(self):
        l2_reg = torch.sum(self.fc.weight**2)
        return l2_reg

    def svm_loss(self, output, target):
        hinge_loss = self.hinge_loss(output, target)
        l2_reg = self.l2_regularization()
        loss = hinge_loss + self.C * l2_reg
        return loss