# 🧠 Synthetic Data Generation using Variational Autoencoders (VAE)

## 📌 Project Overview
This project aims to generate **synthetic tabular data** that closely mimics real-world datasets using a **Variational Autoencoder (VAE)**.  
By learning the underlying data distribution, the model can **generate new, realistic samples** — a valuable technique for **data augmentation**, **privacy preservation**, and **imbalanced dataset correction**.

---

## ⚙️ Key Objectives
- Learn how to explore and preprocess tabular data.
- Implement a **VAE model** from scratch using **PyTorch**.
- Compare **real vs synthetic data distributions**.
- Understand how deep generative models can be used for **data augmentation** in small datasets.

---

## 🧩 Architecture
The VAE consists of two main components:

### 🔹 Encoder
Compresses input features into a lower-dimensional **latent space**.  
It outputs two vectors:
- **μ (mean)** — represents the center of the latent distribution.  
- **σ (standard deviation)** — controls the spread of the distribution.

### 🔹 Decoder
Takes random samples from the latent space and **reconstructs new synthetic data** that follows the same distribution as the original input.

---

## 🧠 Training Process
1. Normalize the data using **StandardScaler** for stable learning.  
2. Pass data through the **encoder** to get latent representations (μ, σ).  
3. Sample a latent vector `z` using the **reparameterization trick**:  
   `z = μ + σ * ε`, where `ε ~ N(0, 1)`.  
4. Decode `z` to reconstruct the input.
5. Optimize the **VAE loss**:  
   ```
   Total Loss = Reconstruction Loss + KL Divergence
   ```
   - *Reconstruction Loss*: Measures how close the output is to the input.  
   - *KL Divergence*: Keeps the latent space distribution close to a normal distribution.

---

## 📊 Results and Analysis
After training, the VAE was able to **generate new synthetic data points** by sampling random latent vectors and decoding them.

Example:
```python
z = torch.randn(500, latent_dim)
synthetic = vae.decode(z).numpy()
```

To validate the results, we compared **feature distributions** between real and synthetic data using **Kernel Density Estimation (KDE) plots**.

The two distributions were visually similar, showing that the VAE successfully captured the data structure.

---

## 🧩 Improvements Implemented
- Added **Batch Normalization** layers to stabilize and accelerate training.  
- Tuned **learning rate** and **latent dimension** for smoother convergence.  
- Used **ReLU** activations and a deeper encoder-decoder architecture for better feature extraction.

---

## 📁 Project Structure
```
📂 synthetic-data-vae
│
├── data/                  # Dataset
├── vae_model.py           # VAE architecture
├── train.py               # Training loop
├── generate.py            # Generate synthetic data
├── analysis.ipynb         # Data exploration and visualization
└── README.md              # Project documentation
```

---

## 🚀 Future Work
- Extend the model to handle **categorical data**.
- Apply **Conditional VAE (CVAE)** for controlled generation.
- Integrate into **data augmentation pipelines** for ML models.

---

## 🧑‍💻 Technologies Used
- **Python**
- **PyTorch**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**

---

## 📈 Example Visualization
![Distributions](example_kde_plot.png)

---

## 💡 Key Takeaways
- Variational Autoencoders are powerful tools for **learning data distributions**.
- Synthetic data can improve model robustness and privacy.
- Deep learning methods can be applied effectively to **tabular data generation**.

---

### ✍️ Author
**Youcef Ouddane**  
🎓 AI & Data Science Student | 🔬 Deep Learning Enthusiast  
📅 2025
