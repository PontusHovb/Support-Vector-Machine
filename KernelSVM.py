import torch
import torch.nn as nn

class KernelSVM(nn.Module):
    def __init__(self, input_size, num_classes=1, C=1.0, kernel='gaussian'):
        super(KernelSVM, self).__init__()
        self.num_classes = num_classes
        self.C = C
        self.kernel = kernel

        self.support_vectors = nn.Parameter(torch.randn(1, input_size), requires_grad=True)
        self.dummy_param = nn.Parameter(torch.randn(1), requires_grad=True)  # Dummy parameter for L2 regularization

    def forward(self, x):
        batch_size, _ = x.shape

        if self.kernel == 'linear':
            kernel_matrix = torch.matmul(x, self.support_vectors.t())
        elif self.kernel == 'poly':
            degree = 2
            bias = 1.0
            kernel_matrix = (torch.matmul(x, self.support_vectors.t()) + bias)**degree
        elif self.kernel == 'gaussian':
            gamma = 0.5
            kernel_matrix = torch.exp(-gamma * torch.norm(x[:, None, :] - self.support_vectors, dim=2)**2)
        else:
            raise ValueError("Unsupported kernel type")

        output = torch.sum(kernel_matrix, dim=1) - self.dummy_param
        return output

    def hinge_loss(self, output, target):
        zero_vector = torch.zeros_like(output*target.t())
        ones_vector = torch.ones_like(output*target.t())
        max_vector = torch.max(zero_vector, ones_vector-output*target.t())
        loss = torch.sum(max_vector)
        return loss

    def l2_regularization(self):
        l2_reg = torch.norm(self.support_vectors, p=2)**2
        return l2_reg

    def svm_loss(self, output, target):
        hinge_loss = self.hinge_loss(output, target)
        l2_reg = self.l2_regularization()
        loss = hinge_loss + self.C * l2_reg
        return loss