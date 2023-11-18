import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import requests
import matplotlib.pyplot as plt

from .utils import *

# https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb

class SegmentPredictor:

    def __init__(self, model_name, device="cuda"):
        self.model = SamModel.from_pretrained(model_name).to(device)
        self.processor = SamProcessor.from_pretrained(model_name)
        self.device = device

    def prep_embedding(self, img):
        inputs = self.processor(img, return_tensors="pt").to(self.device)
        self.image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])


    def predict_masks_from_points(self, img, input_points):
        """
        Input:
            img
            input_points: [[[x1,y1],[x2,y2],...]]
        """
        inputs = self.processor(img, input_points=input_points, return_tensors="pt").to(self.device)
        # pop the pixel_values as they are not neded
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": self.image_embeddings})

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        scores = outputs.iou_scores

        return masks, scores
    
    def predict_masks_from_boxes(self, img, input_boxes):
        """
        Input:
            img
            input_points: [[[x1,y1],[x2,y2],...]]
        """
        inputs = self.processor(img, input_boxes=[input_boxes], return_tensors="pt").to(self.device)
        # pop the pixel_values as they are not neded
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": self.image_embeddings})

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        scores = outputs.iou_scores


        return masks, scores
    
    def predict_masks_from_points_and_boxes(self, img, input_points, input_boxes):
        """
        Input:
            img
            input_points: [[[x1,y1],[x2,y2],...]]
        """
        inputs = self.processor(img, input_boxes=[input_boxes], input_points=[input_points], return_tensors="pt").to(self.device)
        # pop the pixel_values as they are not neded
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": self.image_embeddings})

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        scores = outputs.iou_scores


        return masks, scores