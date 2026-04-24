Self-Pruning Neural Network

Overview

This project implements a self-pruning neural network that learns to remove unnecessary connections during training. Instead of pruning after training, the model uses a gating mechanism to control the importance of each weight.

Each weight has a learnable gate value, and these gates are optimized along with the model parameters to encourage sparsity.

Key Idea

Every weight is associated with a gate value
Gate values are passed through a sigmoid function (between 0 and 1)
Effective weight = weight × gate
If a gate becomes very small, that connection is effectively removed

Loss Function

The training objective combines classification and sparsity:

Total Loss = CrossEntropy Loss + λ × Sparsity Loss

CrossEntropy Loss ensures prediction accuracy
Sparsity Loss applies L1 regularization on gate values

Dataset

CIFAR-10 dataset
Loaded using torchvision.datasets
A subset is used for faster training

Model Architecture

Input: 3 × 32 × 32 images

Fully connected network with:
Two hidden layers
ReLU activation
Custom PrunableLinear layers

Results

Lambda Accuracy Sparsity
0.0001 29.20% 0.00%
0.001 26.90% 0.00%
0.01 29.30% 0.00%

Observations

Increasing lambda increased the sparsity penalty during training
However, no effective pruning occurred as sparsity remained zero
Accuracy showed slight variation but no major drop
Gate values remained distributed instead of moving toward zero

Visualizations

Lambda vs Accuracy
Lambda vs Sparsity
Sparsity vs Accuracy trade-off
Gate value distributions for each lambda

These plots help understand how regularization affects model behavior

Key Insight

The pruning mechanism is correctly implemented, but the current setup does not produce effective sparsity

Proper tuning of lambda and training duration is important to achieve meaningful pruning

How to Run

Install dependencies:

pip install torch torchvision matplotlib numpy

Run the script:

python self_pruning_nn.py

Output

Training logs with epoch-wise loss and sparsity
Final accuracy and sparsity values
Comparison across lambda values
Visualization plots

Limitations

Sparsity remains very low
Training duration is short
Lambda values may not be strong enough

Future Improvements

Increase training epochs
Use stronger or adaptive lambda values
Apply threshold-based pruning
Extend to convolutional networks
Explore structured pruning

Conclusion

This project demonstrates a working implementation of a self-pruning neural network using a gating mechanism. While effective pruning was not achieved, the results provide useful insights into how sparsity behaves during training.