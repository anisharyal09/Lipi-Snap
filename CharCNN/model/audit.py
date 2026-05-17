import torch
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image

class CharCNN(nn.Module):
    def __init__(self, num_classes=62):
        super(CharCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def audit(model_path, val_dir):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint['classes']
    model = CharCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Transformations *
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = datasets.ImageFolder(val_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs("generated_outputs/misclassified", exist_ok=True)

    print("Auditing validation set...")
    with torch.no_grad():
        for i, (image, label) in enumerate(loader):
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            _, pred = torch.max(output, 1)

            if pred != label:
                # Save the image with a filename that tells us the mistake
                true_name = classes[label]
                pred_name = classes[pred]
                
                # Convert back to PIL to save
                img_inv = (image.cpu().squeeze().numpy() * 0.5 + 0.5) * 255
                img_pil = Image.fromarray(img_inv.astype('uint8'))
                img_pil.save(f"generated_outputs/misclassified/idx{i}_True-{true_name}_Pred-{pred_name}.png")

    print("Done! Check the 'generated_outputs/misclassified' folder.")

audit("model/best_char_cnn.pth", "data/merged_dataset/val")