# Stop Sign Image Classifier — Transfer Learning with PyTorch

Final Project for the **Computer Vision and Image Processing** course on Coursera (IBM / Skills Network)

---

## Project Overview

This project builds a binary image classifier that detects whether an image contains a **stop sign** or not. It was developed as part of a simulated real-world scenario at **Jokwu**, a self-driving car startup based in Cape Town, South Africa, where the goal is to build a vision-based road sign detection system for autonomous vehicles.

The classifier is built using **transfer learning** with a pretrained **ResNet-18** model from PyTorch's `torchvision.models`, fine-tuned on a custom labeled dataset of stop sign images.

---

## Key Concepts Covered

- Image preprocessing and data augmentation with `torchvision.transforms`
- Transfer learning using pretrained ResNet-18 (ImageNet weights)
- Freezing feature extractor layers and replacing the output head
- Training loop with loss tracking and best-weight checkpointing
- Learning rate scheduling with `CyclicLR`
- Model evaluation and visualization of predictions

---

## Project Structure

```
├── PeerAssignment-completed.ipynb   # Main Jupyter Notebook (all 25 exercises completed)
├── model.pt                         # Saved model weights (generated after training)
├── dataset/
│   ├── train/
│   │   ├── stop/
│   │   └── not_stop/
│   └── val/
│       ├── stop/
│       └── not_stop/
├── stop-1.jpeg                      # Sample test images
├── stop-2.jpeg
├── not-stop-1.jpeg
├── not-stop-2.jpeg
└── README.md
```

---

## Dataset

| Attribute    | Details                               |
|--------------|---------------------------------------|
| Format       | JPEG images                           |
| Classes      | `stop`, `not_stop`                    |
| Input Size   | Resized to 224x224 pixels             |
| Split        | 90% Training / 10% Validation         |
| Task         | Binary Image Classification           |

The dataset is downloaded automatically in the notebook from IBM Cloud Object Storage and split into training and validation sets using an annotations JSON file.

---

## Model Architecture

- **Base Model:** ResNet-18 (pretrained on ImageNet)
- **Modification:** Final fully connected layer replaced with `Linear(512, 2)` for binary classification
- **Frozen Layers:** All convolutional layers frozen — only the new FC layer is trained
- **Loss Function:** `CrossEntropyLoss`
- **Optimizer:** SGD with momentum
- **LR Scheduler:** CyclicLR (`triangular2` mode)

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/stop-sign-classifier.git
cd stop-sign-classifier
```

### 2. Install dependencies
```bash
pip install torch torchvision numpy pandas matplotlib tqdm pillow
```

### 3. Open the notebook
```bash
jupyter notebook PeerAssignment-completed.ipynb
```

### 4. Run all cells
The notebook will:
- Download and prepare the dataset automatically
- Train the ResNet-18 classifier
- Save the model as `model.pt`
- Test predictions on sample images

---

## Training Configuration

| Hyperparameter   | Value        |
|------------------|--------------|
| Epochs           | 5            |
| Batch Size       | 32           |
| Learning Rate    | 0.001–0.01   |
| Momentum         | 0.9          |
| LR Scheduler     | CyclicLR     |

Training loss decreases steadily across epochs, and validation accuracy improves consistently — demonstrating effective transfer learning even with a small dataset.

---

## Inference Example

```python
from PIL import Image
import torch
from torchvision import transforms, models

class_names = ['not_stop', 'stop']

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load("model.pt", map_location='cpu'))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

image = Image.open("stop-1.jpeg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    predicted = torch.argmax(output, dim=1).item()

print(f"Predicted: {class_names[predicted]}")
```

---

## Sample Predictions

| Image           | Prediction |
|-----------------|------------|
| stop-1.jpeg     | stop       |
| stop-2.jpeg     | stop       |
| not-stop-1.jpeg | not_stop   |
| not-stop-2.jpeg | not_stop   |

---

## Technologies Used

- Python 3.x
- PyTorch and TorchVision
- PIL (Pillow)
- Matplotlib
- NumPy
- Jupyter Notebook

---

## Course Information

- **Course:** Computer Vision and Image Processing
- **Platform:** Coursera (IBM / Skills Network)
- **Project:** Final Graded Assignment — Building an Image Classifier with Transfer Learning

---

## License

This project is for educational purposes as part of a Coursera course assignment.  
Dataset and starter code © IBM Corporation. All rights reserved.
