import torch
from PIL import Image
import requests

from utils import *
from inference import SegmentPredictor

# https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb

MODEL_NAME = "facebook/sam-vit-huge"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    predictor = SegmentPredictor(model_name=MODEL_NAME, device=DEVICE)

    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    # Convert to embeddings
    predictor.prep_embedding(raw_image)

    # # Try pointing to a pixel
    # input_points = [[[450, 600]]]
    # show_points_on_image(raw_image, input_points[0])

    # masks, scores = predictor.predict_masks_from_points(raw_image, input_points)
    # show_masks_on_image(raw_image, masks[0], scores)

    # Multiple points
    input_points = [[[550, 600], [2100, 1000]]]
    show_points_on_image(raw_image, input_points)
    masks, scores = predictor.predict_masks_from_points(raw_image, input_points)
    show_masks_on_image(raw_image, masks[0], scores)

