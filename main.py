import os
# from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from segment_anything.inference import SegmentPredictor
from segment_anything.utils import show_points_on_image, show_masks_on_image , show_mask_creation, ask_for_point, resize_and_crop_image
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# SETTINGS
SEGMENT_MODEL_NAME = "facebook/sam-vit-huge"
SD_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
SPECIAL_SD_MODEL_MADE_FOR_INPAINT = "runwayml/stable-diffusion-inpainting"  # Try compare this with the SD_MODEL_NAME
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PUT THE DATASET OF IMAGE HERE
DATASET_SAMPLE = os.path.join("sample_images", "sample_dog.jpg")
PROMPT = "a brown cat, smile, high resolution"

# IN this project, we use Stable Diffusion Inpainting Pipeline
# to get more flexibility, don't use pipeline. We might be able to find new techniques
# Tutorial for training our own pipeline (near bottom of the page): https://huggingface.co/blog/stable_diffusion

def GET_DIRNAME(save_path):
    counter = 1
    original_save_path = save_path
    while os.path.exists(save_path):
        save_path = original_save_path + "_" + str(counter)
        counter += 1
    return save_path

if __name__ == "__main__":

    # Save path for saving experiment
    filename_without_extension, ext = os.path.splitext(DATASET_SAMPLE)
    file_basename = os.path.basename(filename_without_extension)
    SAVE_PATH = GET_DIRNAME(os.path.join("inpaint_result", file_basename))
    os.makedirs(SAVE_PATH)

    # Load segment anything model
    predictor = SegmentPredictor(model_name=SEGMENT_MODEL_NAME, device=DEVICE)

    # Load stable diffusion model pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(SD_MODEL_NAME, torch_dtype=torch.float16)
    pipe = pipe.to(DEVICE)

    # Open image and transform it to 512 by 512 (input for SD)
    raw_image = resize_and_crop_image(DATASET_SAMPLE).convert("RGB")

    # Save image as an original image
    raw_image.save(os.path.join(SAVE_PATH, f"raw{ext}"))

    # Save prompt
    with open(os.path.join(SAVE_PATH, "prompt.txt"), "w") as f:
        f.write(PROMPT)

    # Ask for point from the user
    input_points = ask_for_point(raw_image)

    # Add point, point will be marked as green star
    show_points_on_image(raw_image, input_points[0], save_path=SAVE_PATH, ext=ext)

    # Predict mask
    predictor.prep_embedding(raw_image)
    masks, scores = predictor.predict_masks_from_points(img=raw_image, input_points=input_points)
    show_masks_on_image(raw_image, masks[0], scores, save_path=SAVE_PATH, ext=ext)

    # Get the mask with the highest scores
    masks = masks[0].squeeze().cpu().detach().numpy()
    scores = scores[0][0].cpu().detach().numpy()
    idx = np.argmax(scores)
    mask = np.vectorize(np.float32)(masks[idx])
    show_mask_creation(raw_image, mask, save_path=SAVE_PATH, ext=ext)

    # Generate inpainting
    image = pipe(prompt=PROMPT, image=raw_image, mask_image=mask).images[0]  # pipeline already gives us PIL
    
    # Save image
    image.show()
    image.save(os.path.join(SAVE_PATH, f"result{ext}"))