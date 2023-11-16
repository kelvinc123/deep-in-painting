import os
import argparse
from ddpm import train

def launch():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 4
    args.image_size = 80
    args.dataset_path = os.path.join("..", "dataset", "pokemon_dataset")
    args.device = "cuda"
    args.lr = 1e-4
    train(args)


if __name__ == '__main__':
    launch()