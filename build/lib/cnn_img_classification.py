"""
Ensemble AI
Date: Mar 24, 2025

This script trains a CNN (NdLinear) model on the CIFAR-10 dataset and evaluates its performance.

Usage Example:

python src/cnn_img_classification.py \
--batch_size 64 \
--learning_rate 0.001 \
--epochs 20 \
--data_dir './data' \
--output_file 'training_results.pdf'
"""

import argparse
import logging
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from ndlinear import NdLinear


def get_args():
    parser = argparse.ArgumentParser(description="HyperMLP Training Script")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for the CIFAR-10 dataset')
    parser.add_argument('--output_file', type=str, default='training_results.pdf',
                        help='Output file for saving training results')

    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def get_compute_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        logging.info("CUDA and MPS not available; using CPU instead.")
        return torch.device("cpu")

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def load_data(transform, data_dir, batch_size):
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

class HyperVision(nn.Module):
    def __init__(self, input_shape, hidden_size):
        super(HyperVision, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.ndlinear = NdLinear((64, 8, 8), hidden_size)
        final_dim = math.prod(hidden_size)
        self.fc_out = nn.Linear(final_dim, 100)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.ndlinear(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_out(self.relu(x))
        return x

def initialize_model(device):
    hyper_vision = HyperVision((3, 32, 32), (64, 8, 8)).to(device)
    return hyper_vision

def get_optimizer(hyper_vision, lr):
    optimizer_hyper = optim.Adam(hyper_vision.parameters(), lr=lr)
    return optimizer_hyper

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(hyper_vision, trainloader, criterion, optimizer_hyper, device):
    hyper_vision.train()
    running_loss_hyper = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        # Train HyperVision
        optimizer_hyper.zero_grad()
        outputs_hyper = hyper_vision(images)
        loss_hyper = criterion(outputs_hyper, labels)
        loss_hyper.backward()
        optimizer_hyper.step()
        running_loss_hyper += loss_hyper.item()

    return running_loss_hyper / len(trainloader)

def evaluate(hyper_vision, testloader, device):
    hyper_vision.eval()
    correct_hyper, total = 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs_hyper = hyper_vision(images)
            _, predicted_hyper = torch.max(outputs_hyper, 1)
            correct_hyper += (predicted_hyper == labels).sum().item()

            total += labels.size(0)

    return 100 * correct_hyper / total

def plot_and_save(hyper_losses, hyper_acc, params_hyper, epochs, filename):
    plt.figure(figsize=(12, 5))

    # Plot Loss Curves
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), hyper_losses, label=f"HyperVision (Params: {params_hyper})", linestyle="solid")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Plot Accuracy Curves
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), hyper_acc, label="HyperVision Accuracy", linestyle="solid")
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    # Parse arguments
    args = get_args()

    # Set up logging
    setup_logging()

    # Setup device and data
    device = get_compute_device()
    transform = get_transform()
    trainloader, testloader = load_data(transform, args.data_dir, args.batch_size)

    # Initialize model and optimizer
    hyper_vision = initialize_model(device)
    optimizer_hyper = get_optimizer(hyper_vision, args.learning_rate)

    criterion = nn.CrossEntropyLoss()

    hyper_losses = []
    hyper_acc = []
    params_hyper = count_parameters(hyper_vision)
    logging.info(f"# Parameters - HyperVision: {params_hyper}")

    for epoch in range(args.epochs):
        loss_hyper = train(hyper_vision, trainloader, criterion, optimizer_hyper, device)
        hyper_losses.append(loss_hyper)

        acc_hyper = evaluate(hyper_vision, testloader, device)
        hyper_acc.append(acc_hyper)

        logging.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"HyperMLP Loss: {loss_hyper:.4f}, Acc: {acc_hyper:.2f}%")

    # Plot and save results
    plot_and_save(hyper_losses, hyper_acc, params_hyper, args.epochs, args.output_file)

if __name__ == "__main__":
    main()