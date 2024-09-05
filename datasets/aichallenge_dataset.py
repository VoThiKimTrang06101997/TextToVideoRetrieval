import os
import torch
import numpy as np
import json
import pandas as pd

import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.nn import functional as F

import clip
import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("RN50x16", device=device)

# Set device to CUDA if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to your ViT-B/32 checkpoint
checkpoint_path = "D:/Course_hoc_mien_phi_workshop/AI Hackkathon/SourceCode/TextToVideo/data/CLIP/ViT-B-32.pt"

# Load the ViT-B/32 model architecture from Hugging Face and set the device
model, preprocess = clip.load("ViT-B/32", device=device)

# Instead of using load_state_dict, directly load the TorchScript model
# TorchScript models contain both the model and the weights
model = torch.jit.load(checkpoint_path, map_location=device)

# Ensure the model is in evaluation mode
model.eval()

print("CLIP model loaded successfully from checkpoint!")




class CustomDataset(Dataset):
    def __init__(self, config, split_type='train', img_transforms=None, clip_model_name="RN50x16"):
        self.config = config
        self.split_type = split_type
        self.img_transforms = img_transforms

        # Load CLIP model and preprocess
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        

        # Define directories for your dataset
        self.keyframes_dir = config.keyframes_dir
        self.metadata_dir = config.metadata_dir
        self.map_keyframes_dir = config.map_keyframes_dir

        # Automatically load keyframe list from the directory
        self.video_list = self._load_keyframe_list()

        print(f"Loaded {len(self.video_list)} keyframe sequences for split: {split_type}")
        
    def _load_keyframe_list(self):
        """
        This function reads the keyframe list from the keyframe subdirectories.
        """
        keyframe_list = []
        
        # Traverse through the keyframe directories
        for subdir in os.listdir(self.keyframes_dir):
            subdir_path = os.path.join(self.keyframes_dir, subdir)
            if os.path.isdir(subdir_path):  # Ensure it's a directory
                # Iterate through video keyframe subdirectories
                for video_subdir in os.listdir(subdir_path):
                    video_subdir_path = os.path.join(subdir_path, video_subdir)
                    if os.path.isdir(video_subdir_path):
                        # Collect keyframe files with both .jpg and .png extensions
                        keyframe_files = [f for f in os.listdir(video_subdir_path) if f.endswith(('.png', '.jpg'))]
                        if keyframe_files:
                            print(f"Found {len(keyframe_files)} keyframes for video {video_subdir}")
                        for keyframe_file in keyframe_files:
                            keyframe_list.append((subdir, video_subdir, keyframe_file))

        if len(keyframe_list) == 0:
            print(f"No keyframe files found in directory: {self.keyframes_dir}")

        return keyframe_list


    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        subdir, video_subdir, keyframe_file = self.video_list[index]
        print(f"Fetching item {index}, Video ID: {video_subdir}, Keyframe: {keyframe_file}")

        # Path for keyframes and metadata
        keyframe_path = os.path.join(self.keyframes_dir, subdir, video_subdir, keyframe_file)
        metadata_path = os.path.join(self.metadata_dir, f"{video_subdir}.json")
        map_keyframes_path = os.path.join(self.map_keyframes_dir, f"{video_subdir}.csv")

        # Ensure the keyframe file exists
        if not os.path.exists(keyframe_path):
            raise FileNotFoundError(f"Keyframe {keyframe_path} does not exist.")

        # Load keyframe image
        keyframe_image = Image.open(keyframe_path).convert('RGB')

        # Preprocess image using CLIP preprocess
        keyframe_image = self.clip_preprocess(keyframe_image).unsqueeze(0).to(self.device)

        # Extract CLIP features using OpenAI's CLIP model
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(keyframe_image).squeeze(0).cpu().numpy()

        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Load keyframes information from the map_keyframes CSV file
        keyframe_data = pd.read_csv(map_keyframes_path)

        return {
            'video_id': video_subdir,
            'clip_features': clip_features,
            'metadata': metadata,
            'keyframes': keyframe_data,
        }

        
        


# Config class to store paths and other configurations
class Config:
    def __init__(self):
        self.keyframes_dir = "D:/Course_hoc_mien_phi_workshop/AI Hackkathon/Dataset/keyframes"
        self.metadata_dir = "D:/Course_hoc_mien_phi_workshop/AI Hackkathon/Dataset/metadata"
        self.map_keyframes_dir = "D:/Course_hoc_mien_phi_workshop/AI Hackkathon/Dataset/map-keyframes"

# Example usage:
config = Config()

# Create dataset object
dataset = CustomDataset(config, split_type='train')

# Example of loading and processing the first 5 items
for i in range(min(5, len(dataset))):
    data = dataset[i]
    print(f"Video {data['video_id']}: CLIP feature shape = {data['clip_features'].shape}")
