# 🌌 Variational Autoencoder (VAE)


**A PyTorch implementation of a Variational Autoencoder for latent space learning and generative tasks.**

</div>

## 📖 Overview

This repository presents a clean and concise implementation of a Variational Autoencoder (VAE) using PyTorch. VAEs are powerful generative models capable of learning a compressed, continuous latent representation of input data, allowing for both efficient data reconstruction and the generation of novel, similar data samples.

The project demonstrates how to:
*   Define a VAE architecture with encoder and decoder networks.
*   Train the VAE using the Evidence Lower Bound (ELBO) objective, which combines reconstruction loss and KL divergence.
*   Generate new data samples by sampling from the learned latent distribution.

It's an excellent starting point for understanding the fundamentals of VAEs and their application in unsupervised learning and generative modeling, particularly for image data like MNIST.

## ✨ Features

-   **Variational Autoencoder (VAE) Architecture**: Implements a complete VAE with distinct encoder and decoder networks.
-   **Probabilistic Latent Space**: Learns a continuous and disentangled latent representation by modeling it as a Gaussian distribution.
-   **Image Reconstruction**: Reconstructs input images from their latent representations, demonstrating effective data compression.
-   **Novel Image Generation**: Generates entirely new, realistic images by sampling from the learned latent space.
-   **ELBO Loss Optimization**: Utilizes the Evidence Lower Bound (ELBO) for stable and effective training, balancing reconstruction accuracy and latent space regularization.
-   **PyTorch Implementation**: Built entirely with PyTorch for flexibility, performance, and GPU acceleration.
-   **Model Checkpointing**: Saves trained model states, allowing for training resumption or later inference.



## 🚀 Quick Start

Follow these steps to get the Variational Autoencoder up and running on your local machine.


### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/SrisuryaTeja/Variational-Auto-Encoder.git
    cd Variational-Auto-Encoder
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
### Training the VAE

To start training the Variational Autoencoder model, run the `train.py` script. This script will handle data loading (e.g., MNIST), model instantiation, the training loop, and model checkpointing.

```bash
python train.py
```

Upon successful completion, the script will typically save model checkpoints in the `checkpoints/` directory and may generate output images (like `generated_images.png`) to visualize the VAE's performance.

## 📁 Project Structure

```
Variational-Auto-Encoder/
├── .gitignore                   # Standard Git ignore file
├── checkpoints/                 # Directory to save trained VAE model weights and states
├── generated_images.png         # An example image showing VAE's generative output
├── model.py                     # Defines the PyTorch module for the VAE (Encoder and Decoder)
├── requirements.txt             # Lists Python dependencies (e.g., torch)
└── train.py                     # Main script for training the VAE model and generating samples
```

## ⚙️ Configuration

The primary configuration for the VAE model and training process is managed within the `train.py` script and `model.py`. Key parameters you might want to adjust include:

### Training Parameters (in `train.py`)
-   **`epochs`**: Number of training iterations over the dataset.
-   **`batch_size`**: Number of samples per gradient update.
-   **`learning_rate`**: Step size for the optimizer during training.
-   **`latent_dim`**: The dimensionality of the latent space (number of features in the compressed representation).
-   **`device`**: Specifies whether to use `cpu` or `cuda` (GPU) for training.

### Model Architecture (in `model.py`)
-   **`input_dim`**: The dimension of the input data (e.g., `784` for flattened 28x28 MNIST images).
-   **`hidden_dims`**: Dimensions of intermediate layers in the encoder and decoder.

These parameters are usually hardcoded or passed as command-line arguments within `train.py`.

## 🔧 Development

### Development Workflow
1.  **Modify `model.py`**: Adjust the VAE architecture (encoder, decoder layers, activation functions) to experiment with different model capacities.
2.  **Modify `train.py`**: Tweak training hyperparameters (learning rate, batch size, number of epochs), experiment with different loss components, or integrate new datasets.
3.  **Run training**: Execute `python train.py` to observe the effects of your changes.
4.  **Monitor outputs**: Check the `checkpoints/` directory for saved models and `generated_images.png` for visual feedback.

## 🧪 Testing

This project does not include explicit unit tests or a dedicated testing framework. The primary way to "test" the model's performance is by:
-   Observing the training loss curves.
-   Inspecting the `generated_images.png` to visually assess the quality and diversity of generated samples.
-   Analyzing the reconstructed images to gauge reconstruction accuracy.





