import torch
import os
import numpy as np
from PIL import Image

class VideoCapture:

    @staticmethod
    def load_frames_from_keyframes(keyframes_dir, video_id, num_frames, size=(224, 224), img_transforms=None, normalize=True):
        """
        Load keyframes from the directory instead of extracting frames from video.
        
        Args:
            keyframes_dir (str): Path to the keyframes directory.
            video_id (str): Video ID for which keyframes are to be loaded (e.g., 'L01_V001').
            num_frames (int): Number of frames to load.
            size (tuple): Desired size (width, height) for the frames.
            img_transforms (callable, optional): Optional transformations to apply to the frames.
            normalize (bool): Whether to normalize pixel values to [0, 1] range.
            
        Returns:
            torch.Tensor: Loaded keyframes stacked as a tensor.
            list: List of keyframe file names.
        """
        # Path to the specific video folder containing keyframes
        keyframe_path = os.path.join(keyframes_dir, video_id)

        # Check if the directory exists
        if not os.path.exists(keyframe_path):
            raise FileNotFoundError(f"Keyframe directory {keyframe_path} does not exist.")

        # Get the list of keyframe files (assuming PNG or JPG format)
        keyframe_files = [f for f in os.listdir(keyframe_path) if f.endswith(('.png', '.jpg'))]
        keyframe_files = sorted(keyframe_files)  # Ensure the files are loaded in order

        # Handle case when no keyframes are found
        if not keyframe_files:
            raise ValueError(f"No keyframe files found in directory {keyframe_path}")

        # If the number of requested frames exceeds available keyframes, take available frames
        acc_samples = min(num_frames, len(keyframe_files))
        selected_keyframes = keyframe_files[:acc_samples]

        frames = []

        # Load each keyframe image
        for frame_file in selected_keyframes:
            frame_path = os.path.join(keyframe_path, frame_file)
            try:
                # Load image using PIL and convert to RGB
                frame = Image.open(frame_path).convert('RGB')

                # Apply any image transformations if specified
                if img_transforms:
                    frame = img_transforms(frame)
                else:
                    # Resize manually if no transform is provided
                    frame = frame.resize(size)

                # Convert the frame to a tensor (C, H, W format)
                frame_tensor = torch.from_numpy(np.array(frame)).permute(2, 0, 1)  # Convert to (C, H, W)
                frames.append(frame_tensor)

            except Exception as e:
                print(f"Failed to load keyframe {frame_file} from {frame_path}: {str(e)}")
                continue  # Skip the failed keyframe and continue

        if not frames:
            raise ValueError(f"No valid frames loaded for video ID {video_id} in directory {keyframe_path}")

        # Stack frames
        frames = torch.stack(frames).float()

        # Normalize pixel values to [0, 1] if specified
        if normalize:
            frames /= 255.0

        return frames, selected_keyframes

# # Example usage of VideoCapture
# keyframes_dir = "path/to/keyframes"
# video_id = "L01_V001"
# num_frames = 10
# size = (224, 224)

# # Load frames with optional transformations
# frames, keyframe_files = VideoCapture.load_frames_from_keyframes(keyframes_dir, video_id, num_frames, size)

# print(f"Loaded {len(frames)} frames from video {video_id}")
