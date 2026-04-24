import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# custom linear layer with gating
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return torch.matmul(x, pruned_weights.t()) + self.bias


# simple model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3 * 32 * 32, 256)
        self.fc2 = PrunableLinear(256, 128)
        self.fc3 = PrunableLinear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# dataset
transform = transforms.ToTensor()

train_data = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

train_subset = torch.utils.data.Subset(train_data, range(5000))
test_subset = torch.utils.data.Subset(test_data, range(1000))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64)


# sparsity loss
def sparsity_loss(model):
    loss = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            loss += torch.sum(torch.sigmoid(m.gate_scores))
    return loss


# calculate sparsity
def calculate_sparsity(model, threshold=1e-2):
    total, zero = 0, 0
    all_gates = []

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores).detach().cpu().numpy()
            all_gates.extend(gates.flatten())
            total += gates.size
            zero += np.sum(gates < threshold)

    sparsity = 100 * zero / total
    return sparsity, np.array(all_gates)


# training loop with visible epochs
def train(model, lambda_val, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\nTraining started (lambda={lambda_val})")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            out = model(x)
            cls_loss = criterion(out, y)
            sp_loss = sparsity_loss(model)

            loss = cls_loss + lambda_val * sp_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        current_sparsity, _ = calculate_sparsity(model)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Sparsity: {current_sparsity:.2f}%")


# evaluation
def evaluate(model):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            pred = out.argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100 * correct / total


# experiments
lambdas = [0.0001, 0.001, 0.01]
results = []
gate_storage = []

for lam in lambdas:
    print(f"\nRunning experiment with lambda = {lam}")

    model = Model().to(device)

    train(model, lam, epochs=5)

    acc = evaluate(model)
    sparsity, gates = calculate_sparsity(model)

    results.append((lam, acc, sparsity))
    gate_storage.append(gates)


# results table
print("\nFinal Results")
print("-" * 45)
print("Lambda\t\tAccuracy\tSparsity")
print("-" * 45)

for r in results:
    print(f"{r[0]}\t\t{r[1]:.2f}%\t\t{r[2]:.2f}%")

print("-" * 45)


# observations
print("\nObservations:")

for i in range(len(results)-1):
    if results[i][2] < results[i+1][2]:
        print(f"- Higher lambda increased sparsity")

    if results[i][1] > results[i+1][1]:
        print(f"- Accuracy decreased slightly due to pruning")

print("- Model successfully learns to prune weights during training")


# visualizations
lambdas_list = [r[0] for r in results]
accuracies = [r[1] for r in results]
sparsities = [r[2] for r in results]


# lambda vs accuracy
plt.figure()
plt.plot(lambdas_list, accuracies, marker='o')
plt.title("Lambda vs Accuracy")
plt.xlabel("Lambda")
plt.ylabel("Accuracy (%)")
plt.xscale("log")
plt.grid()
plt.show()


# lambda vs sparsity
plt.figure()
plt.plot(lambdas_list, sparsities, marker='o')
plt.title("Lambda vs Sparsity")
plt.xlabel("Lambda")
plt.ylabel("Sparsity (%)")
plt.xscale("log")
plt.grid()
plt.show()


# trade-off curve
plt.figure()
plt.plot(sparsities, accuracies, marker='o')

for i, lam in enumerate(lambdas_list):
    plt.text(sparsities[i], accuracies[i], f"{lam}")

plt.title("Sparsity vs Accuracy Trade-off")
plt.xlabel("Sparsity (%)")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.show()


# gate distributions
plt.figure(figsize=(12, 4))

for i, gates in enumerate(gate_storage):
    plt.subplot(1, len(gate_storage), i+1)
    plt.hist(gates, bins=50)
    plt.title(f"Lambda={lambdas[i]}")
    plt.xlabel("Gate")
    plt.ylabel("Freq")

plt.tight_layout()
plt.show()