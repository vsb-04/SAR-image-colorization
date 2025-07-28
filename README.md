# SAR‑Image‑Colorization

![Project Banner](assets/banner.png)

**SAR‑Image‑Colorization** is a deep learning-based framework for turning Synthetic Aperture Radar (SAR) images from grayscale into intuitive, colorized visualizations using a conditional GAN (pix2pix) setup. By learning from paired SAR and optical imagery, the model enhances interpretability of SAR data for remote sensing and geospatial analysis.

---
dataset from kaggle 
    link - https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain
## 📌 Features

- **True SAR colorization** using a pix2pix-style Conditional GAN (cGAN).
- **U‑Net generator** with encoder‑decoder and skip connections.
- **PatchGAN discriminator** to ensure realistic local textures.
- Composite loss: adversarial (binary cross‑entropy) + pixel-wise L1.
- Trains on paired datasets of Sentinel‑1 (SAR) and Sentinel‑2 (optical) data :contentReference[oaicite:2]{index=2}.

---

## 🚀 Architecture

### Generator (U‑Net)
- Input: 256×256×1 (L-channel grayscale SAR image)
- Encoder: 8 layers (Conv2D → BatchNorm → LeakyReLU), doubling filters up to 512
- Decoder: 7 upsampling layers (Conv2DTranspose → BatchNorm → ReLU), dropout on first 3 layers
- Skip‑connections between encoder and decoder to preserve spatial structure and detail :contentReference[oaicite:3]{index=3}

### Discriminator (PatchGAN)
- Inputs: concatenated original SAR (1‑ch) + generated or real RGB image (3‑ch)
- Structure: several convolutional downsampling blocks (64 → 1 filter)
- Outputs: 30×30 patch-level discriminator map, helping focus on local realism :contentReference[oaicite:4]{index=4}

---

## 🧪 Installation & Setup

### Prerequisites
- Python 3.7+
- TensorFlow or PyTorch (depending on implementation)
- Common ML libraries: NumPy, SciPy, OpenCV, GDAL, etc.
- CUDA/CuDNN (if using GPU acceleration)


