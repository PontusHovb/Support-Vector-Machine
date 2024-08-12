import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from KernelSVM import KernelSVM
from SoftMarginSVM import LinearSVM

BATCH_SIZE = 64
INPUT_SIZE = 784
NUM_EPOCHS = 25
LEARNING_RATE = 0.0001
MOMENTUM = 0.0

# Define the dataset and data loaders
train_data = datasets.MNIST('./data/anupam-data/pytorch/data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_data = datasets.MNIST('./data/anupam-data/pytorch/data/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

subset_indices = ((train_data.targets == 0) + (train_data.targets == 1)).nonzero().view(-1)

train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                           batch_size=BATCH_SIZE, 
                                           shuffle=False, 
                                           sampler=SubsetRandomSampler(subset_indices),
                                           drop_last=True)

subset_indices = ((test_data.targets == 0) + (test_data.targets == 1)).nonzero().view(-1)

test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                          batch_size=BATCH_SIZE,
                                          shuffle=False,
                                          sampler=SubsetRandomSampler(subset_indices),
                                          drop_last=True)

# Create models
svm_model_soft_margin = LinearSVM(INPUT_SIZE)    
svm_model_kernel = KernelSVM(INPUT_SIZE)

# Create two optimizers for the models
svm_optimizer_soft_margin = torch.optim.SGD(svm_model_soft_margin.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
svm_optimizer_kernel = torch.optim.SGD(svm_model_kernel.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# Test the SVM models
def test_model(model, loader):
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.reshape(-1, 28*28)
    
        outputs = model(images)    
        predicted = outputs.data >= 0
        total += labels.size(0) 
        correct += (predicted.view(-1).long() == labels).sum()    
    
    return (100 * (correct.float() / total))

def train_model(model, optimizer):
    x = list()
    loss_y = list()
    acc_y = list()

    # Training loop for SVM with kernel
    for epoch in range(NUM_EPOCHS):
        avg_loss_epoch = 0
        batch_loss = 0
        total_batches = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28*28)                      
            labels = Variable(2*(labels.float()-0.5))
                
            
            outputs = model(images).view(-1)        
            loss_svm = model.svm_loss(outputs, labels)
            
            optimizer.zero_grad()
            loss_svm.backward()
            optimizer.step()

            total_batches += 1     
            batch_loss += loss_svm.item()

        avg_loss_epoch = batch_loss/total_batches
        print ('Epoch [{}/{}], SVM - Average Loss: {:.4f}' 
                    .format(epoch+1, NUM_EPOCHS, avg_loss_epoch ))

        x.append(epoch+1)
        loss_y.append(avg_loss_epoch)
        acc_y.append(test_model(model, test_loader))

    return x, acc_y

# Train models
x_softmargin, y_acc_softmargin = train_model(svm_model_soft_margin, svm_optimizer_kernel)
x_kernel, y_acc_kernel = train_model(svm_model_kernel, svm_optimizer_kernel)

# Print results 
accuracy_soft_margin = test_model(svm_model_soft_margin, test_loader)
print('Accuracy of SVM with Soft Margin on the test images: {:.2f}%'.format(accuracy_soft_margin))

plt.plot(x_softmargin, y_acc_softmargin)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy for SVM with Soft Margin')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0, decimals=2))
plt.show()


accuracy_kernel = test_model(svm_model_kernel, test_loader)
print('Accuracy of SVM with Kernel on the test images: {:.2f}%'.format(accuracy_kernel))

plt.plot(x_kernel, y_acc_kernel)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy for SVM with Gaussian Kernel')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0, decimals=2))
plt.show()

