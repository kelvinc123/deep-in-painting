import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda
from diffusers import AutoencoderKL

##### Modify this
DATAPATH = "./data/FirstBatch/combined"
device = "cuda" if torch.cuda.is_available() else "cpu"
#####


# Orignal Dir
train_dir = os.path.join(DATAPATH, "train_dir")
val_dir = os.path.join(DATAPATH, "val_dir")
train_image_dir = os.path.join(train_dir, "images")
train_mask_dir = os.path.join(train_dir, "masks")
train_label_dir = os.path.join(train_dir, "labels")
val_image_dir = os.path.join(val_dir, "images")
val_mask_dir = os.path.join(val_dir, "masks")
val_label_dir = os.path.join(val_dir, "labels")

# VAE Dir
vae_train_dir = os.path.join(DATAPATH, "vae_train_dir")
vae_val_dir = os.path.join(DATAPATH, "vae_val_dir")
vae_train_image_dir = os.path.join(vae_train_dir, "images")
vae_train_mask_dir = os.path.join(vae_train_dir, "masks")
vae_train_label_dir = os.path.join(vae_train_dir, "labels")
vae_val_image_dir = os.path.join(vae_val_dir, "images")
vae_val_mask_dir = os.path.join(vae_val_dir, "masks")
vae_val_label_dir = os.path.join(vae_val_dir, "labels")

# Create Dir
os.makedirs(vae_train_image_dir, exist_ok=True)
os.makedirs(vae_train_mask_dir, exist_ok=True)
os.makedirs(vae_val_image_dir, exist_ok=True)
os.makedirs(vae_val_mask_dir, exist_ok=True)

# Load pre-trained VAE model
vae = AutoencoderKL.from_pretrained("vae")
vae = vae.to(device)

# Define the preprocessing transformations for images
image_transform = Compose([
    Resize((vae.config.sample_size, vae.config.sample_size)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    Lambda(lambda x: x * 2.0),  # Scale to [-1, 1]
])

# Define the preprocessing transformations for masks
mask_transform = Compose([
    Resize((vae.config.sample_size, vae.config.sample_size)),
    ToTensor(),
    Lambda(lambda x: torch.cat([x, x, x], 0)),  # Repeat the mask channel to create a 3-channel image
])


# TRAIN -> VAE TRAIN
image_files = os.listdir(train_image_dir)
print("Converting train data to vae...")
for image_file in tqdm(image_files):
    img_name = os.path.splitext(image_file)[0]
    image_path = os.path.join(train_image_dir, image_file)
    mask_path = os.path.join(train_mask_dir, image_file)
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # Preprocess
    preprocessed_image = image_transform(image).unsqueeze(0)  # Add batch dimension
    preprocessed_image = preprocessed_image.to(device)
    preprocessed_mask = mask_transform(mask).unsqueeze(0)  # Add batch dimension
    preprocessed_mask = preprocessed_mask.to(device)

    # Encode the image to latent space
    with torch.no_grad():
        latent_image = np.array(vae.encode(preprocessed_image).latent_dist.sample().detach().cpu())
        latent_mask = np.array(vae.encode(preprocessed_mask).latent_dist.sample().detach().cpu())

    np.savez(os.path.join(vae_train_image_dir, f"{img_name}.npz"), latent_image)
    np.savez(os.path.join(vae_train_mask_dir, f"{img_name}.npz"), latent_mask)

# VAL -> VAE VAL
image_files = os.listdir(val_image_dir)
print("Converting val data to vae...")
for image_file in tqdm(image_files):
    img_name = os.path.splitext(image_file)[0]
    image_path = os.path.join(val_image_dir, image_file)
    mask_path = os.path.join(val_mask_dir, image_file)
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # Preprocess
    preprocessed_image = image_transform(image).unsqueeze(0)  # Add batch dimension
    preprocessed_image = preprocessed_image.to(device)
    preprocessed_mask = mask_transform(mask).unsqueeze(0)  # Add batch dimension
    preprocessed_mask = preprocessed_mask.to(device)

    # Encode the image to latent space
    with torch.no_grad():
        latent_image = np.array(vae.encode(preprocessed_image).latent_dist.sample().detach().cpu())
        latent_mask = np.array(vae.encode(preprocessed_mask).latent_dist.sample().detach().cpu())

    np.savez(os.path.join(vae_val_image_dir, f"{img_name}.npz"), latent_image)
    np.savez(os.path.join(vae_val_mask_dir, f"{img_name}.npz"), latent_mask)


# Copy label
shutil.copytree(train_label_dir, vae_train_label_dir)
shutil.copytree(val_label_dir, vae_val_label_dir)

# Copy classname
shutil.copy(os.path.join(train_dir, "labels.txt"), os.path.join(vae_train_dir, "labels.txt"))
shutil.copy(os.path.join(train_dir, "labels.txt"), os.path.join(vae_val_dir, "labels.txt"))