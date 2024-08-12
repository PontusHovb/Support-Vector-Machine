# Support Vector Machine (SVM)
In this project Support Vector Machine is implemented for image classification of handwritten numbers in the MINST database. Both a soft-margin SVM as well as Kernel SVM with three different kernels (linear, poly and gaussian) are implemented.

## Soft margins
Soft-margin Support Vector Machine (SVM) is an extension of the standard SVM that allows for some misclassifications. It introduces a penalty term for misclassified points, controlled by a regularization parameter, to balance the trade-off between maximizing the margin and minimizing classification errors. This approach enhances the SVM's ability to handle non-linearly separable data and improves generalization to unseen data.

TODO: Illustration
TODO: Accuracy

## SVM with kernel
Support Vector Machine (SVM) with a kernel enables SVM to handle non-linear classifications. Instead of working in the original feature space, SVM with a kernel transforms the data into a higher-dimensional space where a linear separation is possible. This makes SVM with kernels particularly effective for classification tasks where the data is not linearly separable in its original form.

TODO: Illustration
TODO: Accuracy

## Source
- [MNIST database](https://www.tensorflow.org/datasets/catalog/mnist)