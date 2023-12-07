"""Invertible and Deterministic transformation of 4x96x96 <-> 3x512x512"""

import os
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda, ToPILImage
from diffusers import AutoencoderKL

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load pre-trained VAE model
vae_path = os.path.join("MAT", "vae")
vae = AutoencoderKL.from_pretrained(vae_path)
vae = vae.to(device)

# Define the preprocessing transformations for images
image_transform = Compose([
    Resize((vae.config.sample_size, vae.config.sample_size)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define the preprocessing transformations for masks
mask_transform = Compose([
    Resize((vae.config.sample_size, vae.config.sample_size)),
    ToTensor(),
    Lambda(lambda x: torch.cat([x, x, x], 0)),  # Repeat the mask channel to create a 3-channel image
])

def apply_mask_transform(tensor_mask):
    return mask_transform(tensor_mask)

def apply_image_transform(tensor_img):
    return image_transform(tensor_img)

def inverse_mask_transform(transformed_mask, original_size):
    # Extract a single channel from the 3-channel mask
    transformed_mask = transformed_mask[0, :, :].unsqueeze(0)
    
    # Resize to original size and convert to PIL Image
    resize = Resize(original_size)
    inverse_transform = Compose([resize])
    
    return ToPILImage()(inverse_transform(transformed_mask))

def inverse_image_transform(transformed_tensor, original_size):
    # Inverse Normalize
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transformed_tensor = transformed_tensor * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)

    # Resize to original size and convert to PIL Image
    resize = Resize(original_size)
    inverse_transform = Compose([resize])
    inverse_image = inverse_transform(transformed_tensor.unsqueeze(0))

    return ToPILImage()(inverse_image[0])

def transform_to_3_512_512(tensor):
    padded_tensor = torch.nn.functional.pad(tensor[:3], (0, 416, 0, 416), mode='constant', value=0)
    # Distributing the fourth channel across the three channels
    for i in range(3):
        padded_tensor[i, :96, 480:512] = tensor[3, :, 32 * i:32 * (i + 1)]

    return padded_tensor

def revert_back_to_4_96_96(tensor):
    # Reconstructing the fourth channel from the three channels
    reconstructed_channel = torch.zeros((1, 96, 96))
    for i in range(3):
        reconstructed_channel[0, :, 32 * i:32 * (i + 1)] = tensor[i, :96, 480:512]

    # Removing padding to get back to 4 x 96 x 96
    reverted_tensor = torch.cat((tensor[:3, :96, :96], reconstructed_channel), dim=0)

    return reverted_tensor

def vae_encode(vae, tensor):
    # Encode the image to latent space
    tensor = tensor.to(device)
    with torch.no_grad():
        latent = vae.encode(tensor).latent_dist.sample().detach().cpu()
    return latent

def vae_decode(vae, tensor):
    tensor = tensor.to(device)
    with torch.no_grad():
        reconstructed_image = vae.decode(tensor).sample.detach().cpu()
    return reconstructed_image

if __name__ == "__main__":
    from PIL import Image

    DATA_DIR = os.path.join("MAT", "test_sets", "Places")
    IMAGE_DIR = os.path.join(DATA_DIR, "images")
    MASK_DIR = os.path.join(DATA_DIR, "masks")

    LATENT_IMAGE_DIR = os.path.join(DATA_DIR, "latent_images")
    LATENT_MASK_DIR = os.path.join(DATA_DIR, "latent_masks")
    REC_IMAGE_DIR = os.path.join(DATA_DIR, "rec_images")
    REC_MASK_DIR = os.path.join(DATA_DIR, "rec_masks")
    os.makedirs(LATENT_IMAGE_DIR, exist_ok=True)
    os.makedirs(LATENT_MASK_DIR, exist_ok=True)
    os.makedirs(REC_IMAGE_DIR, exist_ok=True)
    os.makedirs(REC_MASK_DIR, exist_ok=True)
    prompt = ""
    ext_save = ".png"
    for img_name in os.listdir(IMAGE_DIR):
        mask_name = img_name.replace(".jpg", ".png")
        if "test" in img_name:
            mask_name = mask_name.replace("test", "mask")
        if "test" not in img_name:
            continue

        img_path = os.path.join(IMAGE_DIR, img_name) 
        mask_path = os.path.join(MASK_DIR, mask_name)

        # Load images and masks size; 3x512x512
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        # mask = Image.fromarray(255*(1 - (np.array(mask) // 255)))  # Invert mask

        # Transform to be able to fed into VAE Encoder
        # by normalization resize: Output 3x768x768
        transformed_img = apply_image_transform(img)
        transformed_mask = apply_mask_transform(mask)

        # VAE Encoder: Output 4x96x96
        latent_image = vae_encode(vae=vae, tensor=transformed_img.unsqueeze(0))[0]
        latent_mask = vae_encode(vae=vae, tensor=transformed_mask.unsqueeze(0))[0]

        # Convert to 3x512x512
        latent_image_512 = transform_to_3_512_512(latent_image)
        latent_mask_512 = transform_to_3_512_512(latent_mask)

        # Save latent image and mask
        latent_img_result_path = os.path.join(LATENT_IMAGE_DIR, f"{os.path.splitext(img_name)[0]}{ext_save}")
        latent_mask_result_path = os.path.join(LATENT_MASK_DIR, f"{os.path.splitext(img_name)[0]}{ext_save}")
        ToPILImage()(latent_image_512).save(latent_img_result_path)
        ToPILImage()(latent_mask_512).save(latent_mask_result_path)

        ########## MAT NETWORK TO INPAINT LATENT HERE

        # Convert to 4x96x96
        reconstructed_image_96 = revert_back_to_4_96_96(latent_image_512)
        reconstructed_mask_96 = revert_back_to_4_96_96(latent_mask_512)

        # VAE Decoder: Output 4x96x96
        reconstructed_image = vae_decode(vae, reconstructed_image_96.unsqueeze(0))[0]
        reconstructed_mask = vae_decode(vae, reconstructed_mask_96.unsqueeze(0))[0]

        # Convert to Image with size 3x512x512
        reconstructed_image = inverse_image_transform(reconstructed_image, 512)
        reconstructed_mask = inverse_mask_transform(reconstructed_mask, 512)

        # Save reconstructed image and masks
        rec_img_result_path = os.path.join(REC_IMAGE_DIR, f"{os.path.splitext(img_name)[0]}{ext_save}")
        rec_mask_result_path = os.path.join(REC_MASK_DIR, f"{os.path.splitext(img_name)[0]}{ext_save}")
        reconstructed_image.save(rec_img_result_path)
        reconstructed_mask.save(rec_mask_result_path)