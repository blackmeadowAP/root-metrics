#!/usr/bin/env python
# coding: utf-8

# Import Python Standard Library dependencies
import datetime
from functools import partial
from glob import glob
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
from typing import Any, Dict, Optional

# Import utility functions
from cjm_psl_utils.core import download_file, file_extract, get_source_code
from cjm_pil_utils.core import resize_img, get_img_files, stack_imgs
from cjm_pytorch_utils.core import pil_to_tensor, tensor_to_pil, get_torch_device, set_seed, denorm_img_tensor, move_data_to_device
from cjm_pandas_utils.core import markdown_to_pandas, convert_to_numeric, convert_to_string
from cjm_torchvision_tfms.core import ResizeMax, PadSquare, CustomRandomIoUCrop

# Import the distinctipy module
from distinctipy import distinctipy

# Import matplotlib for creating plots
import matplotlib.pyplot as plt

# Import numpy
import numpy as np

# Import the pandas package
import pandas as pd

# Set options for Pandas DataFrame display
pd.set_option('max_colwidth', None)  # Do not truncate the contents of cells in the DataFrame
pd.set_option('display.max_rows', None)  # Display all rows in the DataFrame
pd.set_option('display.max_columns', None)  # Display all columns in the DataFrame

# Import PIL for image manipulation
from PIL import Image, ImageDraw

# Import PyTorch dependencies
import torch
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtnt.utils import get_module_summary
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms.v2  as transforms
from torchvision.transforms.v2 import functional as TF

# Import Mask R-CNN
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Import tqdm for progress bar
from tqdm.auto import tqdm
import cv2
import gc
#очистка памяти видеокарты
gc.collect()
torch.cuda.empty_cache()

# Set the seed for generating random numbers in PyTorch, NumPy, and Python's random module.
seed = 1234
set_seed(seed)

device = get_torch_device()
dtype = torch.float32
device, dtype

#Setting the dataset
# The name for the project
project_name = f"pytorch-mask-r-cnn-instance-segmentation"

# The path for the project folder
project_dir = Path(f"./{project_name}/")

# Create the project directory if it does not already exist
project_dir.mkdir(parents=True, exist_ok=True)

# Define path to store datasets
dataset_dir = Path("./Datasets/")
# Create the dataset directory if it does not exist
dataset_dir.mkdir(parents=True, exist_ok=True)

# Prepend a `background` class to the list of class names
class_names = ['background', 'line', 'root']

# Display labels using a Pandas DataFrame
pd.DataFrame(class_names)

# Generate a list of colors with a length equal to the number of labels
colors = distinctipy.get_colors(len(class_names))

# Make a copy of the color map in integer format
int_colors = [tuple(int(c*255) for c in color) for color in colors]

# Generate a color swatch to visualize the color map
distinctipy.color_swatch(colors)

print('Device: ', device)


# Set the name of the font file
font_file = 'KFOlCnqEu92Fr1MmEU9vAw.ttf'
# Download the font file
download_file(f"https://fonts.gstatic.com/s/roboto/v30/{font_file}", "./")

draw_bboxes = partial(draw_bounding_boxes, fill=False, width=2, font=font_file, font_size=25)


def create_polygon_mask(image_size, vertices):
    """
    Create a grayscale image with a white polygonal area on a black background.

    Parameters:
    - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
    - vertices (list): A list of tuples, each containing the x, y coordinates of a vertex
                        of the polygon. Vertices should be in clockwise or counter-clockwise order.

    Returns:
    - PIL.Image.Image: A PIL Image object containing the polygonal mask.
    """

    # Create a new black image with the given dimensions
    mask_img = Image.new('L', image_size, 0)
    
    # Draw the polygon on the image. The area inside the polygon will be white (255).
    ImageDraw.Draw(mask_img, 'L').polygon(vertices, fill=(255))

    # Return the image with the drawn polygon
    return mask_img

def run_epoch(model, dataloader, optimizer, lr_scheduler, device, scaler, epoch_id, is_training):
    """
    Function to run a single training or evaluation epoch.
    
    Args:
        model: A PyTorch model to train or evaluate.
        dataloader: A PyTorch DataLoader providing the data.
        optimizer: The optimizer to use for training the model.
        loss_func: The loss function used for training.
        device: The device (CPU or GPU) to run the model on.
        scaler: Gradient scaler for mixed-precision training.
        is_training: Boolean flag indicating whether the model is in training or evaluation mode.
    
    Returns:
        The average loss for the epoch.
    """
    # Set the model to training mode
    model.train()
    
    epoch_loss = 0  # Initialize the total loss for this epoch
    progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")  # Initialize a progress bar
    
    # Loop over the data
    for batch_id, (inputs, targets) in enumerate(dataloader):
        # Move inputs and targets to the specified device
        inputs = torch.stack(inputs).to(device)
        
        # Forward pass with Automatic Mixed Precision (AMP) context manager
        with autocast(torch.device(device).type):
            if is_training:
                losses = model(inputs.to(device), move_data_to_device(targets, device))
            else:
                with torch.no_grad():
                    losses = model(inputs.to(device), move_data_to_device(targets, device))
        
            # Compute the loss
            loss = sum([loss for loss in losses.values()])  # Sum up the losses

        # If in training mode, backpropagate the error and update the weights
        if is_training:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                if new_scaler >= old_scaler:
                    lr_scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
            optimizer.zero_grad()

        # Update the total loss
        loss_item = loss.item()
        epoch_loss += loss_item
        
        # Update the progress bar
        progress_bar_dict = dict(loss=loss_item, avg_loss=epoch_loss/(batch_id+1))
        if is_training:
            progress_bar_dict.update(lr=lr_scheduler.get_last_lr()[0])
        progress_bar.set_postfix(progress_bar_dict)
        progress_bar.update()

        # If loss is NaN or infinity, stop training
        if is_training:
            stop_training_message = f"Loss is NaN or infinite at epoch {epoch_id}, batch {batch_id}. Stopping training."
            assert not math.isnan(loss_item) and math.isfinite(loss_item), stop_training_message

    # Cleanup and close the progress bar 
    progress_bar.close()
    
    # Return the average loss for this epoch
    return epoch_loss / (batch_id + 1)

def train_loop(model, 
               train_dataloader, 
               valid_dataloader, 
               optimizer,  
               lr_scheduler, 
               device, 
               epochs, 
               checkpoint_path, 
               use_scaler=False):
    """
    Main training loop.
    
    Args:
        model: A PyTorch model to train.
        train_dataloader: A PyTorch DataLoader providing the training data.
        valid_dataloader: A PyTorch DataLoader providing the validation data.
        optimizer: The optimizer to use for training the model.
        lr_scheduler: The learning rate scheduler.
        device: The device (CPU or GPU) to run the model on.
        epochs: The number of epochs to train for.
        checkpoint_path: The path where to save the best model checkpoint.
        use_scaler: Whether to scale graidents when using a CUDA device
    
    Returns:
        None
    """
    # Initialize a gradient scaler for mixed-precision training if the device is a CUDA GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and use_scaler else None
    best_loss = float('inf')  # Initialize the best validation loss

    # Loop over the epochs
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Run a training epoch and get the training loss
        train_loss = run_epoch(model, train_dataloader, optimizer, lr_scheduler, device, scaler, epoch, is_training=True)
        # Run an evaluation epoch and get the validation loss
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_dataloader, None, None, device, scaler, epoch, is_training=False)

        # If the validation loss is lower than the best validation loss seen so far, save the model checkpoint
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

            # Save metadata about the training process
            training_metadata = {
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss, 
                'learning_rate': lr_scheduler.get_last_lr()[0],
                'model_architecture': model.name
            }
            with open(Path(checkpoint_path.parent/'training_metadata.json'), 'w') as f:
                json.dump(training_metadata, f)

    # If the device is a GPU, empty the cache
    if device.type != 'cpu':
        getattr(torch, device.type).empty_cache()

dataset_path = Path("/home/user4/Desktop/programs roots/Datasets/unpack 4d")    
        
# Generate timestamp for the training session (Year-Month-Day_Hour_Minute_Second)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create a directory to store the checkpoints if it does not already exist
checkpoint_dir = Path(project_dir/f"{timestamp}")

# Create the checkpoint directory if it does not already exist
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Create a color map and write it to a JSON file
color_map = {'items': [{'label': label, 'color': color} for label, color in zip(class_names, colors)]}
with open(f"{checkpoint_dir}/{dataset_path.name}-colormap.json", "w") as file:
    json.dump(color_map, file)

# Print the name of the file that the color map was written to
print(f"{checkpoint_dir}/{dataset_path.name}-colormap.json")


# Get a list of image files in the dataset
#ЗАГРУЗКА РАНЕЕ ОБУЧЕННОЙ МОДЕЛИ
filepath = '/home/user4/Desktop/programs roots/model.pth'
#torch.save(model, filepath) 
model = torch.load(filepath)

# Set the model to evaluation mode
model.eval();

def predict_img(path_input, output_path):
    print(path_input)
    test_img = Image.open(path_input).convert('RGB')
    input_img = test_img
    resized_image = input_img
    # Convert the resized image to a PyTorch tensor
    resized_image_tensor = transforms.ToTensor()(resized_image)
    min_img_scale = min(test_img.size) / min(input_img.size)
    model.to(device)
    input_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])(input_img)[None].to(device)
    
    # Make a prediction with the model
    with torch.no_grad():
        model_output = model(input_tensor)
    
    # Set the confidence threshold
    threshold = 0.26
    

    # Move model output to the CPU
    model_output = move_data_to_device(model_output, 'cpu')

    # Filter the output based on the confidence threshold
    scores_mask = model_output[0]['scores'] > threshold

    # Scale the predicted bounding boxes
    pred_bboxes = BoundingBoxes(model_output[0]['boxes'][scores_mask]*min_img_scale, format='xyxy', canvas_size=input_img.size[::-1])

    # Get the class names for the predicted label indices
    pred_labels = [class_names[int(label)] for label in model_output[0]['labels'][scores_mask]]

    # Extract the confidence scores
    pred_scores = model_output[0]['scores']

    # Scale and stack the predicted segmentation masks
    pred_masks = F.interpolate(model_output[0]['masks'][scores_mask], size=test_img.size[::-1])
    pred_masks = torch.concat([Mask(torch.where(mask >= threshold, 1, 0), dtype=torch.bool) for mask in pred_masks])

    # Get the annotation colors for the targets and predictions
    #target_colors=[int_colors[i] for i in [class_names.index(label) for label in target_labels]]
    pred_colors=[int_colors[i] for i in [class_names.index(label) for label in pred_labels]]

    # Convert the test images to a tensor
    img_tensor = transforms.PILToTensor()(input_img)

    # Annotate the test image with the target segmentation masks
    annotated_tensor = draw_segmentation_masks(image=img_tensor, masks=pred_masks, alpha=0.9, colors=pred_colors)
    annotated_test_img = tensor_to_pil(annotated_tensor)
    splitted = str(path_input).split('/')
    name = splitted[-1]
    
    annotated_test_img.save(os.path.join(output_path, name)) 
    print(os.path.join(output_path, name), 'saved')
    return os.path.join(output_path, name)

def mask_extractor(img_path, output_path_roots, output_path_leaves):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape
    leaves = img.copy()
    roots = img.copy()
    #roots = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #leaves = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape
    print(height, width, channels)
   
    #r, g, b = leaves.split()
    lowest_y = 1000
    lowest_x = 400
    new_leaves = Image.fromarray(leaves)
    new_roots = Image.fromarray(roots)

    print(new_leaves.size, new_leaves.height, new_leaves.width)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for x in range(width - 1):
        for y in range(height - 1):
            #print(x, y)
            #print(x, y)
            #pixel = img.getpixel((x, y))

            r = img[y, x, 0]
            g = img[y, x, 1]
            b = img[y, x, 2]
            

            if r >= 10 and r <= 20 and g >= 120 and g <= 130 and b >= 240 and b <= 250:
                print('blue')
            #if r >= 230 and r <= 250 and g >= 10 and g <= 30 and b >= 230 and b <= 250:
                #lowest_y = min(lowest_y, y)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                #lowest_x = min(lowest_x, x)
                lowest_y = max(y, lowest_y)
                lowest_x = min(x, lowest_x)

    
    #roots
    print(lowest_x,lowest_y)
    # Create a new image with pixels above the lowest pink pixel
    new_image_roots = Image.new('RGB', (new_roots.width, new_roots.height), (255, 255, 255))
    for x in range(new_roots.width):
        for y in range(lowest_y, new_roots.height):
            #print(x, y, lowest_y)
            pixel = new_roots.getpixel((x, y))
            new_image_roots.putpixel((x, y - lowest_y), pixel)   
    
    kernel_31 = np.ones((3,1),np.uint8)
    kernel_13 = np.ones((1,3),np.uint8)
    kernel = np.ones((9, 5), np.uint8) 
    
    #LB_roots = np.array([5,105,230])
    LB_roots = np.array([4,4,4])
    UB_roots = np.array([60,160,255])
    roots = np.array(new_image_roots)
    mask_roots = cv2.inRange(roots,LB_roots,UB_roots)
    mask_roots = cv2.morphologyEx(mask_roots, cv2.MORPH_OPEN, kernel_31)
    mask_roots = cv2.morphologyEx(mask_roots, cv2.MORPH_CLOSE, kernel_13)
    mask_roots = cv2.morphologyEx(mask_roots, cv2.MORPH_OPEN, kernel_13)
    mask_roots = cv2.dilate(mask_roots, kernel, iterations=1) 
    
    splitted = str(img_path).split('/')
    name = splitted[-1]
    
    np_roots = np.array(new_image_roots)
    
    cv2.imwrite(os.path.join(output_path_roots, name), mask_roots)
    print(os.path.join(output_path_roots, name), 'saved')
    
class_names = ['background', 'line', 'root']

# Generate a list of colors with a length equal to the number of labels
colors = distinctipy.get_colors(len(class_names))

# Make a copy of the color map in integer format
int_colors = [tuple(int(c*255) for c in color) for color in colors]

# Generate a color swatch to visualize the color map
distinctipy.color_swatch(colors)

test_path = '/home/user4/Desktop/programs roots/Datasets/unpack 4d'
img_file_paths = get_img_files(test_path)
output_path = '/home/user4/Desktop/programs roots/Datasets/masked'
output_path_roots = '/home/user4/Desktop/programs roots/Datasets/roots'
output_path_leaves = '/home/user4/Desktop/programs roots/lines'

for img in img_file_paths:
    print(img)
    input_color_mask_path = predict_img(img, output_path)
    mask_extractor(input_color_mask_path, output_path_roots, output_path_leaves)    

