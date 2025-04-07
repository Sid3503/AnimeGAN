# AnimeGAN: Generating Anime Faces with DCGANs

![Anime Face Examples](https://github.com/user-attachments/assets/a79280ef-2617-4548-a76a-9470a535895a)

## 📋 Overview

AnimeGAN is a deep learning project that uses Generative Adversarial Networks (GANs) to generate anime character faces. Built using PyTorch, this implementation trains a DCGAN (Deep Convolutional GAN) on over 63,000 anime face images to produce new, unique anime character designs.

## 🌟 Features

- DCGAN architecture optimized for anime face generation
- Training visualization tools to monitor progress
- Batch processing for efficient training
- Pre-processing pipeline for the Anime Face Dataset
- Customizable hyperparameters for model tuning
- Video output of training progression

## 🔧 Requirements

- Python 3.6+
- PyTorch 1.7.0+
- torchvision 0.8.1+
- matplotlib
- OpenCV (for video generation)
- numpy
- tqdm (for progress bars)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/anime-gan.git
cd anime-gan

# Install dependencies
pip install numpy matplotlib torch==1.7.1 torchvision==0.8.2 tqdm opencv-python
```

## 💾 Dataset

This project uses the [Anime Face Dataset](https://www.kaggle.com/splcher/animefacedataset) from Kaggle, which contains over 63,000 high-quality anime character face images.

To download the dataset:

```python
# Using the opendatasets library
import opendatasets as od
od.download('https://www.kaggle.com/splcher/animefacedataset')
```

You'll need a Kaggle account and API credentials. See the [Kaggle API documentation](https://github.com/Kaggle/kaggle-api) for details on setting up your API key.

## 🏗️ Model Architecture

### Generator

The generator transforms random noise vectors into anime face images using transposed convolutional layers:

- Input: Random latent vector of size `(batch_size, 128, 1, 1)`
- 5 transposed convolutional layers with batch normalization and ReLU activations
- Output: Generated image of size `(batch_size, 3, 64, 64)` with Tanh activation

### Discriminator

The discriminator evaluates whether an image is real or generated:

- Input: Image of size `(batch_size, 3, 64, 64)`
- 5 convolutional layers with batch normalization and LeakyReLU activations
- Output: Probability score between 0-1 (fake vs. real)

## ⚙️ Training

Training happens through adversarial learning, where the generator and discriminator networks compete with each other:

```python
# Quick start training
python train.py --epochs 40 --learning_rate 0.0001 --batch_size 128
```

### Parameters

- `--epochs`: Number of training epochs (default: 40)
- `--learning_rate`: Learning rate for Adam optimizer (default: 0.0001)
- `--batch_size`: Batch size for training (default: 128)
- `--image_size`: Size of the images (default: 64)
- `--latent_size`: Size of the latent vector (default: 128)
- `--sample_interval`: Interval to save sample images (default: 1)

## 📊 Results

The model progressively learns to generate anime faces over training epochs:

- Early epochs (1-10): Blurry shapes and colors
- Middle epochs (10-25): Recognizable facial features
- Later epochs (25-40): Refined details and style

Training Process:

https://github.com/user-attachments/assets/b254df75-8725-4f1d-a966-81f34339c7c4

## 📁 Project Structure

```
anime-gan/
├── train.py               # Main training script
├── models/                # Model architecture definitions
│   ├── generator.py
│   └── discriminator.py
├── utils/                 # Utility functions
│   ├── data_loader.py
│   └── visualization.py
├── generated/             # Output directory for generated images
├── checkpoints/           # Model checkpoints during training
├── README.md
└── requirements.txt
```

## 💡 Usage Examples

### Generate images with a trained model

```python
import torch
from models.generator import Generator

# Load a trained generator
generator = Generator()
generator.load_state_dict(torch.load('G.pth'))
generator.eval()

# Generate images
latent_vectors = torch.randn(16, 128, 1, 1)
with torch.no_grad():
    fake_images = generator(latent_vectors)

# Save images
from torchvision.utils import save_image
save_image(fake_images, 'generated_anime_faces.png', normalize=True)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgements

- [Anime Face Dataset](https://www.kaggle.com/splcher/animefacedataset) for the training data
- [DCGAN Paper](https://arxiv.org/abs/1511.06434) for the model architecture inspiration
