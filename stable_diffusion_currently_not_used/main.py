import os
from diffusers import StableDiffusionPipeline
import torch


MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe = pipe.to(DEVICE)

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]  
    
    os.makedirs("results", exist_ok=True)
    image.save(os.path.join("results", "astronaut_rides_horse.png"))