Case Study Report – Self-Pruning Neural Network

Objective:

The aim of this project is to build a neural network that can learn to prune its own connections during training. Instead of removing weights after training, the model uses a gating mechanism to identify less important connections and suppress them dynamically.

Why L1 Regularization Encourages Sparsity?

L1 regularization works by penalizing the absolute values of parameters. In this implementation, it is applied to the sigmoid-transformed gate values, which lie between 0 and 1. This creates a constant pressure for these gate values to decrease. Over time, some gates are expected to move closer to zero, effectively turning off the corresponding weights and making the network sparse.

Methodology:

A custom linear layer was implemented where each weight is associated with a learnable gate parameter. These gate values are passed through a sigmoid function and multiplied with the weights during the forward pass.

The loss function used during training is:

Total Loss = Classification Loss + λ × Sparsity Loss

The sparsity loss is computed as the sum of all gate values, encouraging the network to reduce unnecessary connections. The model was trained on a subset of the CIFAR-10 dataset for faster experimentation, and results were evaluated across three different lambda values.

Results Summary:
Lambda	Test Accuracy	Sparsity (%)
0.0001	29.20%	0.00%
0.001	26.90%	0.00%
0.01	29.30%	0.00%
Observations

Increasing the value of λ led to a noticeable increase in the overall loss during training, indicating that the sparsity penalty was being applied more strongly. However, the sparsity level remained at 0.00% across all experiments, showing that the model did not push any gate values below the pruning threshold.

Accuracy showed minor fluctuations across different λ values. While there was a slight drop at λ = 0.001, the accuracy recovered at λ = 0.01. This suggests that the sparsity constraint affected optimization but did not significantly impact the final performance in this setup.

The gate distributions also remained spread out between 0 and 1, without forming a clear concentration near zero. This indicates that pruning was not effectively happening during training.

Analysis

The results suggest that the current regularization strength is not sufficient to induce meaningful sparsity in the network. Even though higher λ values increased the penalty term in the loss function, they did not push gate values close enough to zero to deactivate connections.

There are a few possible reasons for this behavior. First, the number of training epochs is relatively low, which may not give enough time for the gates to converge toward zero. Second, the scale of the sparsity loss compared to the classification loss might not be strong enough to dominate the optimization process. Third, the sigmoid transformation can make it harder for values to reach exactly zero, especially in early training stages.

Visualization Insights

The lambda versus accuracy graph shows slight variation in performance but no clear degradation pattern. The lambda versus sparsity graph remains flat, confirming that no effective pruning occurred. The sparsity versus accuracy plot does not show a meaningful trade-off due to the lack of sparsity.

The gate distribution plots further confirm this observation, as values remain broadly distributed instead of clustering near zero.

Conclusion:

The model successfully integrates a gating mechanism and sparsity loss into the training process. However, it does not achieve effective pruning under the current settings.

While the framework is correctly implemented, achieving meaningful sparsity requires stronger regularization, longer training, or improved gating strategies. This experiment highlights the importance of balancing regularization strength and training dynamics when designing self-pruning neural networks.

Future Improvements:

Increasing the number of training epochs could allow gate values more time to converge toward zero. Experimenting with larger λ values or dynamically adjusting λ during training may help enforce stronger sparsity. Additionally, alternative approaches such as hard thresholding or structured pruning could improve pruning effectiveness.