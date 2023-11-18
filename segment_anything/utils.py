import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_on_image(raw_image, input_points, input_labels=None, save_path=None, ext=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    if save_path:
        if not ext:
            ext = ".jpg"
        plt.savefig(os.path.join(save_path, f"point_on_image{ext}"))
    plt.show()

def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_masks_on_image(raw_image, masks, scores, save_path=None, ext=None):
    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if scores.shape[0] == 1:
        scores = scores.squeeze()
    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask = mask.cpu().detach()
        axes[i].imshow(np.array(raw_image))
        show_mask(mask, axes[i])
        axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
        axes[i].axis("off")
    if save_path:
        if not ext:
            ext = ".jpg"
        plt.savefig(os.path.join(save_path, f"predicted_masks{ext}"))
    plt.show()

def show_mask_creation(raw_image, mask, save_path=None, ext=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 15))
    axes[0].imshow(np.array(raw_image))
    axes[1].imshow(mask,cmap="grey")
    axes[2].imshow(np.array(raw_image))
    show_mask(mask, axes[2])
    if save_path:
        if not ext:
            ext = ".jpg"
        plt.savefig(os.path.join(save_path, f"mask_with_highest_score{ext}"))
    plt.show()

def ask_for_point(raw_image):
    input_points = []
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(raw_image)
    def on_click(event):
        if event.inaxes is not None:
            print('x=%d, y=%d' % (int(event.xdata), int(event.ydata)))
            input_points.append([[int(event.xdata), int(event.ydata)]])
            plt.close(fig)

    # Connect the click event to the handler
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    return input_points

# Function to resize and crop image (from chatgpt)
def resize_and_crop_image(image_path):
    img = Image.open(image_path)

    # If both dimensions are less than 512, crop to square first
    if img.width < 512 and img.height < 512:
        # Find the smaller dimension
        min_dimension = min(img.width, img.height)

        # Crop to a square
        left = (img.width - min_dimension) / 2
        top = (img.height - min_dimension) / 2
        right = (img.width + min_dimension) / 2
        bottom = (img.height + min_dimension) / 2

        img = img.crop((left, top, right, bottom))

        # Resize to 512x512
        img = img.resize((512, 512), Image.Resampling.LANCZOS)

    # If one or both dimensions are greater than or equal to 512
    else:
        # Resize if necessary to maintain aspect ratio
        aspect_ratio = img.width / img.height
        if img.width < 512:
            new_width = 512
            new_height = int(new_width / aspect_ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        elif img.height < 512:
            new_height = 512
            new_width = int(aspect_ratio * new_height)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Crop the larger dimension to make it a square
        min_dimension = min(img.width, img.height)
        left = (img.width - min_dimension) / 2
        top = (img.height - min_dimension) / 2
        right = (img.width + min_dimension) / 2
        bottom = (img.height + min_dimension) / 2

        img = img.crop((left, top, right, bottom))

        # Resize to 512x512 if not already
        if img.width != 512 or img.height != 512:
            img = img.resize((512, 512), Image.Resampling.LANCZOS)

    return img