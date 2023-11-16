from torch.utils.data import DataLoader
import os
import numpy as np
import argparse
from utils import *
from PIL import Image

def launch():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 120
    args.dataset_path = os.path.join("..", "dataset", "pokemon_dataset")
    args.device = "cuda"
    args.lr = 3e-4

    loader = get_data(args)
    imgs, _ = next(iter(loader)) 
    imgs = (imgs + 1) / 2
    imgs = (imgs * 255).type(torch.uint8)


    # creating a object 
    im = Image.open(os.path.join(args.dataset_path, "pokemon", "abra.png")) 
    print(np.array(im).shape)
    
    im.show()
    
    save_images(imgs[:5], "ko.png")
    

if __name__ == "__main__":
    launch()