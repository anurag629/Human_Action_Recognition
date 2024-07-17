import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from data.loaders.ucf50_loader import UCF50EncodedDataset
from src.models.non_graph_based import NonGraphBasedCNN
from src.models.part_based_graph import PartBasedGraphCNN
from src.models.spatio_temporal_graph import SpatioTemporalGraphCNN
from src.models.part_based_graph_convolutions import PartBasedGraphConvolutionalNetwork
from src.utils.logger import setup_logger
from src.config import Config
import cv2
import numpy as np
from PIL import Image

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_model(model_type, num_classes, input_shape):
    print(f"Initializing {model_type} model with {num_classes} classes and input shape {input_shape}.")
    if model_type == 'non_graph':
        return NonGraphBasedCNN(num_classes, input_shape)
    elif model_type == 'part_based_graph':
        return PartBasedGraphCNN(num_classes, input_shape)
    elif model_type == 'spatio_temporal_graph':
        return SpatioTemporalGraphCNN(num_classes, input_shape)
    elif model_type == 'part_based_graph_convolutions':
        return PartBasedGraphConvolutionalNetwork(num_classes, input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_data(config):
    print("Loading dataset...")
    dataset = UCF50EncodedDataset(config.encoded_data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"Dataset loaded with {len(dataset)} samples: {train_size} for training, {val_size} for validation.")
    return train_loader, val_loader, len(os.listdir(config.data_dir))

def get_class_names(data_dir):
    return sorted(os.listdir(data_dir))

def train_model(config, train_loader, val_loader, num_classes):
    experiment_dir = f'experiments/experiment_{config.experiment_number}'
    log_dir = os.path.join(experiment_dir, 'logs')
    model_dir = os.path.join(experiment_dir, 'models')
    plot_dir = os.path.join(experiment_dir, 'plots')
    ensure_dir_exists(log_dir)
    ensure_dir_exists(model_dir)
    ensure_dir_exists(plot_dir)
    logger = setup_logger('trainer', os.path.join(log_dir, 'trainer.log'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config.model_type, num_classes, config.input_shape).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    print("Starting training...")
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        logger.info(f"Epoch [{epoch+1}/{config.epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Epoch [{epoch+1}/{config.epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    save_model(model, model_dir, f'{config.model_type}.pth')
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, plot_dir, config.model_type, 'training')
    print("Training completed.")

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def save_model(model, model_dir, model_name):
    ensure_dir_exists(model_dir)
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, plot_dir, model_type, phase):
    if not train_losses or not val_losses or not train_accuracies or not val_accuracies:
        print("No data available for plotting.")
        return

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f'{model_type}_{phase}_metrics.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Plots saved to {plot_path}")

def preprocess_frame(frame):
    # Convert the frame (NumPy array) to a PIL image
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    frame = transform(frame)
    frame = frame.unsqueeze(0)  # Add batch dimension
    return frame

def test_model(config):
    experiment_dir = f'experiments/experiment_{config.experiment_number}'
    log_dir = os.path.join(experiment_dir, 'logs')
    ensure_dir_exists(log_dir)
    logger = setup_logger('tester', os.path.join(log_dir, 'tester.log'))

    print("Loading dataset for testing...")
    dataset = UCF50EncodedDataset(config.encoded_data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config.model_type, len(os.listdir(config.data_dir)), config.input_shape).to(device)
    model_path = os.path.join(experiment_dir, 'models', f'{config.model_type}.pth')
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
    val_losses, val_accuracies = [val_loss], [val_acc]  # Wrap in lists for consistency
    logger.info(f"Validation Loss: {val_losses[0]:.4f}, Validation Accuracy: {val_accuracies[0]:.4f}")
    print(f"Validation Loss: {val_losses[0]:.4f}, Validation Accuracy: {val_accuracies[0]:.4f}")

    # Save validation plots
    plot_dir = os.path.join(experiment_dir, 'plots')
    ensure_dir_exists(plot_dir)
    plot_metrics(val_losses, val_losses, val_accuracies, val_accuracies, plot_dir, config.model_type, 'testing')

def realtime_test(config):
    experiment_dir = f'experiments/experiment_{config.experiment_number}'
    model_dir = os.path.join(experiment_dir, 'models')
    model_path = os.path.join(model_dir, f'{config.model_type}.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config.model_type, len(os.listdir(config.data_dir)), config.input_shape).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")

    class_names = get_class_names(config.data_dir)

    cap = cv2.VideoCapture(0)  # Use 0 for laptop camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame = preprocess_frame(frame).to(device)
        with torch.no_grad():
            prediction = model(preprocessed_frame)
        predicted_class = torch.argmax(prediction, axis=1).item()
        predicted_class_name = class_names[predicted_class]

        cv2.putText(frame, f'Action: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Real-Time Action Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def upload_test(config, video_path):
    experiment_dir = f'experiments/experiment_{config.experiment_number}'
    model_dir = os.path.join(experiment_dir, 'models')
    model_path = os.path.join(model_dir, f'{config.model_type}.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config.model_type, len(os.listdir(config.data_dir)), config.input_shape).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")

    class_names = get_class_names(config.data_dir)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame = preprocess_frame(frame).to(device)
        with torch.no_grad():
            prediction = model(preprocessed_frame)
        predicted_class = torch.argmax(prediction, axis=1).item()
        predicted_class_name = class_names[predicted_class]

        cv2.putText(frame, f'Action: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Video Action Recognition', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Part-based GCN for Action Recognition")
    parser.add_argument('--config', type=str, default='configs/config.json', help='Path to the configuration file')

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.set_defaults(func=lambda config: train_model(config, *load_data(config)))
    train_parser.add_argument('--config', type=str, default='configs/config.json', help='Path to the configuration file')

    test_parser = subparsers.add_parser('test', help='Test a trained model')
    test_parser.set_defaults(func=test_model)
    test_parser.add_argument('--config', type=str, default='configs/config.json', help='Path to the configuration file')

    realtime_parser = subparsers.add_parser('realtime', help='Test model with laptop camera')
    realtime_parser.set_defaults(func=realtime_test)
    realtime_parser.add_argument('--config', type=str, default='configs/config.json', help='Path to the configuration file')

    upload_parser = subparsers.add_parser('upload', help='Test model with uploaded video')
    upload_parser.set_defaults(func=upload_test)
    upload_parser.add_argument('--config', type=str, default='configs/config.json', help='Path to the configuration file')
    upload_parser.add_argument('--video', type=str, required=True, help='Path to the video file')

    args = parser.parse_args()
    config = Config(args.config)
    
    if args.command == 'train':
        args.func(config)
    elif args.command == 'test':
        args.func(config)
    elif args.command == 'realtime':
        args.func(config)
    elif args.command == 'upload':
        args.func(config, args.video)

if __name__ == "__main__":
    main()
