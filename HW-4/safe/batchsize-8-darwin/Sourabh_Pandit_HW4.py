import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import argparse
import numpy as np
import glob
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from time import time
from sklearn.model_selection import train_test_split

import torchvision

class SUN397Dataset(Dataset):
    """
    A custom dataset class for loading the SUN397 dataset.
    """

    # ################################################### #
    # Dataset constructor
    # ################################################### #
    def __init__(self, data_dir, transform=None):
        """
        Initializes the dataset with images and labels.
        Args:
            data_dir (str): Path to the data directory.
            transform (callabel, optional): Optional transform to be applied on an image.
        """

        # Root dir for data
        self.data_dir = data_dir

        # List of pathes to images
        self.image_paths = []

        # Class lablel bedroom, airport etc
        self.labels = []

        class_folders = []          # Find all class folders upto 2 levels deep

        # Iterate over the data dir
        for subdir in os.listdir(data_dir):
            # Like ./data/a etc
            subdir_path = os.path.join(data_dir, subdir)

            if os.path.isdir(subdir_path):
                for cls_name in os.listdir(subdir_path):
                    full_cls_path = os.path.join(subdir_path, cls_name) # like ./data/a/bedroom
                    if os.path.isdir(full_cls_path):
                        class_folders.append(full_cls_path)

        # Extract class names from folder names and sort
        class_names = sorted([os.path.basename(path) for path in class_folders])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        #  Collect all image_paths and lables by iterating through all class folders
        for cls_path in class_folders:
            cls_name = os.path.basename(cls_path)   # Class names are the same as folder names
            label = self.class_to_idx[cls_name]     # 0: bedroom, etc

            for fname in os.listdir(cls_path):
                if fname.endswith(".jpg"):
                    fpath = os.path.join(cls_path, fname)
                    self.image_paths.append(fpath)
                    self.labels.append(label)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Compute per-channel mean and std
        self.mean, self.std = self.compute_mean_std()
        print(f"Mean: {self.mean}, StDev: {self.std}")

        if transform is not None:
            self.transform = transform
        else:
            # Set final transform with normalization
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])



    def get_mean_std(self):
        return self.mean, self.std

    # ################################################### #
    # Return the size of the dataset
    # ################################################### #
    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.image_paths)


    # ################################################### #
    # Return the image and the label at an index
    # ################################################### #
    def __getitem__(self, idx):
        """
        Retrieves an image and its label at the specified index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: (image, label)
        """
        label = self.labels[idx]
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label



    # ################################################### #
    # Compute mean and std deviation for RGG channels for
    # all the images in the dataset
    # ################################################### #
    def compute_mean_std(self, batch_size=1, num_workers=0):
        """
        Computes per-channel mean and standard deviation of the dataset.

        Args:
            batch_size (int): Batch size for loading data.
            num_workers (int): Number of subprocesses for data loading.

        Returns:
            tuple: (mean list, std list)
        """
        # Temporarily use basic transform without normalization
        original_transform = self.transform
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        loader = DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        channel_sum = torch.zeros(3)
        channel_squared_sum = torch.zeros(3)
        total_pixels = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn()
        ) as progress:
            task = progress.add_task("[cyan]Computing mean/std...", total=len(loader))

            #for images, _ in loader:
            for images, _ in loader:
                total_pixels += images.numel() / 3
                channel_sum += images.sum(dim=[0, 2, 3])
                channel_squared_sum += (images ** 2).sum(dim=[0, 2, 3])
                progress.update(task, advance=1)

        mean = channel_sum / total_pixels
        std = (channel_squared_sum / total_pixels - mean ** 2).sqrt()

        # Restore original transform
        self.transform = original_transform

        return mean.tolist(), std.tolist()


class CNN(nn.Module):
    """
    Define your CNN Model here
    """
    # ################################################### #
    # CNN Constructor/CNN Architecture
    # ################################################### #
    def __init__(self, num_classes=4):
        """
        Initializes the layers of the CNN model.

        Args:
            num_classes (int): Number of output classes.
        """

        super(CNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     # Output: 64 x 112 x 112
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     # Output: 128 x 56 x 56
        )

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     # Output: 256 x 28 x 28
        )

        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((8, 8))    # Output: 512 x 8 x 8
        )

        # Fully connected layer (flatten first)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )


    # ################################################### #
    # Forward pass
    # ################################################### #
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output of the model.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = self.classifier(x)
        return x




# ################################################### #
# Return means and stdevs for RGB channels forl all
# the images in the dataset
# ################################################### #
def calculate_mean_std(**kwargs):
    """
    Fill in the per channel mean and standard deviation of the dataset.
    Just fill in the values, no need to compute them.
    """
    # return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]

    means = [0.5127117037773132, 0.45288100838661194, 0.39744383096694946]
    stdevs = [0.24955996870994568, 0.25343775749206543, 0.2614697515964508]
    return means, stdevs

'''
All of the following functions are optional. They are provided to help you get started.
'''

def train(model, train_loader, val_loader=None, num_epochs=5, lr=0.001, bs=8, device="cpu"):
    """
    Trains the CNN model and saves the best version based on highest validation accuracy.

    Args:
        model: CNN model instance
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation
        num_epochs: Number of training epochs
        lr: Learning rate
        device: "cpu" or "cuda"
    """
    print(f"Enter Train()")
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    def evaluate(model, loader):
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(loader)
        return accuracy, avg_val_loss

    for epoch in range(num_epochs):
        running_loss = 0.0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn()
        ) as progress:
            task = progress.add_task(f"[cyan]Epoch {epoch+1}/{num_epochs} Training...", total=len(train_loader))

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress.update(task, advance=1)

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Avg Train Loss: {avg_loss:.4f}")

        # Validation step
        if val_loader:
            val_acc, val_loss = evaluate(model, val_loader)
            print(f"Validation → Accuracy: {val_acc:.2f}%, Loss: {val_loss:.4f}")

            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                host = os.uname().nodename.split(".")[0]
                cwd = os.getcwd().split("/")[-1]         # just the last folder name
                model_pt_name = f"model_{host}__epoch_{epoch+1}_of_{num_epochs}__lr_{lr}__BS_{bs}__{cwd}__acc_{val_acc:.2f}.pt"

                torch.save(model.state_dict(), model_pt_name) # torch.save(model.state_dict(), "model.pt")
                print(f"New best model saved (Val Accuracy: {val_acc:.2f}%)")
                print(f"NEW MODEL NAME: {model_pt_name}", flush=True)

                os.system(f"cp {model_pt_name} model.pt")
                for f in os.listdir("."):
                    if f.startswith("model_") and f.endswith(".pt") and f != model_pt_name and f != "model.pt":
                        os.remove(f)

        print()  # Newline for spacing

    print(f"Exit Train()")



def test(model, test_loader, **kwargs):
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./data', help='Path to training data directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=200, help='Number of Epochs')
    parser.add_argument('--bs', type=int, default=8, help='Batch Size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
    return parser.parse_args()


def test_dataset():

    print(f"Enter Test()")
    ds  = SUN397Dataset("./data")
    calculate_mean_std()
    print(f"Exit Test()")

def test_cnn_constructor():
    model = CNN(num_classes=4)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

def debug_forward_pass():
    model = CNN(num_classes=4)
    x = torch.randn(1, 3, 224, 224)
    print(f"Input: {x.shape}")

    x = model.conv1(x)
    print(f"After conv1: {x.shape}")

    x = model.conv2(x)
    print(f"After conv2: {x.shape}")

    x = x.view(x.size(0), -1)
    print(f"After flatten: {x.shape}")

    x = model.fc(x)
    print(f"After fc: {x.shape}")




# ################################################### #
# ################################################### #
def main():
    args = parse_args()

    seed = args.seed
    train_dir = args.train_dir
    num_epochs = args.epochs
    lr = args.lr
    bs= args.bs

    # Do we have a GPU to use ?
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"seed = {seed}")
    print(f"train_dir = {train_dir}")
    print(f"num_epochs = {num_epochs}")
    print(f"learning_rate = {lr}")
    print(f"batch_size = {bs}")
    print (f"device = {device}")

    torch.manual_seed(args.seed)

    # Load dataset
    dataset = SUN397Dataset("./data")
    print(f"\nHere 1")
    mean, std = dataset.get_mean_std()
    num_classes = len(dataset.class_to_idx)

    generator = torch.Generator().manual_seed(seed)
    indices = list(range(len(dataset)))
    labels = dataset.labels

    train_indices, val_indices =  train_test_split(indices, test_size=0.2, stratify=labels, random_state=seed)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
        ])

    train_dataset = SUN397Dataset("./data", transform=train_transform)
    val_dataset = SUN397Dataset("./data", transform=None)

    train_data = torch.utils.data.Subset(train_dataset, train_indices)
    val_data = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False)

    model = CNN(num_classes=num_classes)
    train(model, train_loader, val_loader=val_loader, num_epochs=num_epochs, lr=lr, bs=bs, device=device)

if __name__ == "__main__":
    main()

