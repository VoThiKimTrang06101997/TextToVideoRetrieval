import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
import clip
from transformers import AutoTokenizer, AutoModel
import logging
import torch.nn.functional as F
import torchvision.transforms as transforms

from pytube import YouTube

from torch.utils.checkpoint import checkpoint

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda" 
print(f'Using device: {device}')

# Load CLIP model and ensure it runs on the chosen device (GPU or CPU)
# clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
# print("CLIP model loaded successfully!")

from transformers import AutoTokenizer, AutoModel

# Use cached tokenizers
def initialize_phobert_tokenizer():
    """Initialize the PhoBERT tokenizer with caching for faster loads."""
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", cache_dir='./cache')
    model = AutoModel.from_pretrained("vinai/phobert-base-v2", cache_dir='./cache')
    return tokenizer, model

# Initialize PhoBERT model and tokenizer
phobert_tokenizer, phobert_model = initialize_phobert_tokenizer()


class Config:
    def __init__(self):
        self.videos_dir = "/content/drive/MyDrive/AI_Hackkathon/Dataset/video"
        self.keyframes_dir = "/content/drive/MyDrive/AI_Hackkathon/Dataset/keyframes"
        self.metadata_dir = "/content/drive/MyDrive/AI_Hackkathon/Dataset/metadata"
        self.map_keyframes_dir = "/content/drive/MyDrive/AI_Hackkathon/Dataset/map-keyframes"
        self.output_dir = "/content/drive/MyDrive/AI_Hackkathon/output"  
        
        self.clip_arch = 'ViT-B/32'  # Model architecture
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
        self.input_res = 224  # Input resolution for images
        self.num_frames = 16  # Number of frames to use
        self.max_token_length = 256  # Set max token length for PhoBERT embeddings
        self.seed = 24  # Define the seed here

        print(f"Keyframes directory: {self.keyframes_dir}")
        print(f"Metadata directory: {self.metadata_dir}")
        print(f"Map-keyframes directory: {self.map_keyframes_dir}")
        print(f"Using device: {self.device}")
    
# Create an instance of Config
config = Config()

# Set seed for reproducibility
torch.manual_seed(config.seed)  # Use instance attribute 'config.seed'
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)

# Function to download transcription from YouTube
import os
import json
import urllib.error
from youtube_transcript_api import YouTubeTranscriptApi, CouldNotRetrieveTranscript
from pytube import YouTube
import torch
from PIL import Image

# Function to load already processed keyframes
def load_processed_keyframes(processed_keyframes_file):
    if os.path.exists(processed_keyframes_file):
        with open(processed_keyframes_file, 'r') as f:
            return set(json.load(f))  # Load as a set for faster lookups
    else:
        return set()

# Function to save processed keyframe
def save_processed_keyframe(keyframe_id, processed_keyframes_file, processed_keyframes):
    processed_keyframes.add(keyframe_id)
    with open(processed_keyframes_file, 'w') as f:
        json.dump(list(processed_keyframes), f)

def download_youtube_transcription_from_json(json_path, output_dir):
    """Download a YouTube video transcription from a JSON file's 'watch_url' field."""
    try:
        # Check if the metadata file exists
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Metadata file does not exist: {json_path}")
        
        # Load the metadata JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Extract the YouTube URL from the 'watch_url' field
        youtube_url = metadata.get('watch_url', None)

        if youtube_url is None:
            raise ValueError(f"No 'watch_url' found in the metadata file: {json_path}")
        
        # Extract the video ID from the URL
        video_id = youtube_url.split("v=")[-1]

        # Try downloading the official transcription using YouTubeTranscriptApi
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['vi'])
            transcript_text = " ".join([t['text'] for t in transcript])
            print(f"Transcription for {youtube_url} in Vietnamese: {transcript_text}")

            # Save the transcription to a file
            output_file = os.path.join(output_dir, f"{video_id}_transcription.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            print(f"Transcription saved to {output_file}")
            return transcript_text

        except Exception as e:
            if "age-restricted" in str(e).lower():
                print(f"Skipping age-restricted video {video_id}: {youtube_url}")
            else:
                print(f"Error retrieving transcription: {e}")
            return None

        except CouldNotRetrieveTranscript as transcript_error:
            print(f"Error downloading transcription: {transcript_error}")
            print(f"Attempting to download auto-generated captions...")

            # Fall back to auto-generated captions via pytube
            try:
                yt = YouTube(youtube_url)
                # captions = yt.captions['vi']
                captions = yt.captions.get_by_language_code('vi')

                if 'vi' not in yt.captions:
                  print(f"No auto-generated captions found for {youtube_url}")
                  return None  # Skip this item as no captions are available

                if captions:
                    transcript_text = captions.generate_srt_captions()
                    print(f"Auto-generated captions found for {youtube_url}.")
                    
                    # Save the auto-generated captions
                    output_file = os.path.join(output_dir, f"{video_id}_auto_captions.txt")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(transcript_text)
                    print(f"Auto-generated captions saved to {output_file}")
                    return transcript_text
                else:
                    print(f"No auto-generated captions found for {youtube_url}.")
                    return None
            except Exception as e:
                print(f"Error downloading auto-generated captions: {e}")
                return None

    except Exception as e:
        print(f"An error occurred while downloading transcription: {e}")
        return None


class CustomDataset(Dataset):
    def __init__(self, config, split_type='train', img_transforms=None, clip_model_name="ViT-B/32"):
        self.config = config
        self.split_type = split_type
        self.img_transforms = img_transforms        

        # Load CLIP model and preprocess
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set to GPU if available
        print(f"Using device: {self.device}")

        # Load the CLIP model and its preprocess function
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        print("CLIP model and preprocess function loaded.")
        
        # Load PhoBERT model and tokenizer
        self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2")
        
        # Define directories for your dataset
        self.keyframes_dir = config.keyframes_dir
        self.metadata_dir = config.metadata_dir
        self.map_keyframes_dir = config.map_keyframes_dir

        # Load processed videos
        self.processed_keyframes_file = os.path.join(config.output_dir, '/content/drive/MyDrive/AI_Hackkathon/save_processed_keyframes/processed_keyframes.json')  # Path to save processed keyframes
        self.processed_videos = load_processed_keyframes(self.processed_keyframes_file)

        # Automatically load keyframe list from the directory
        self.video_list = self._load_keyframe_list()
        print(f"Loaded {len(self.video_list)} keyframe sequences for split: {split_type}")
        
    def _load_keyframe_list(self):
        keyframe_list = []
        for subdir in os.listdir(self.keyframes_dir):
            subdir_path = os.path.join(self.keyframes_dir, subdir)
            if os.path.isdir(subdir_path):
                for video_subdir in os.listdir(subdir_path):
                    video_subdir_path = os.path.join(subdir_path, video_subdir)
                    if os.path.isdir(video_subdir_path):
                        keyframe_files = [f for f in os.listdir(video_subdir_path) if f.endswith(('.png', '.jpg'))]
                        for keyframe_file in keyframe_files:
                            keyframe_list.append((subdir, video_subdir, keyframe_file))

        if len(keyframe_list) == 0:
            print(f"No keyframe files found in directory: {self.keyframes_dir}")

        return keyframe_list

    def validate_tokens(self, input_ids):
      """
      Validate token IDs to ensure they are within the vocabulary range.
      """
      vocab_size = self.phobert_tokenizer.vocab_size  # Get the vocab size from the tokenizer
      invalid_token_flag = False
      for token in input_ids['input_ids'].view(-1):  # Access the 'input_ids' field in the dictionary
          if token >= vocab_size or token < 0:
              print(f"Invalid token {token} out of vocab size {vocab_size}")
              invalid_token_flag = True
      return not invalid_token_flag  # Return False if any invalid tokens are found

      if not self.validate_tokens(input_ids):
        raise ValueError(f"Invalid tokens found in input: {input_ids}")


    def resize_embeddings(self, embeddings, target_size):
        if embeddings.size(1) > target_size:
            embeddings = embeddings[:, :target_size]
        elif embeddings.size(1) < target_size:
            padding_size = target_size - embeddings.size(1)
            embeddings = F.pad(embeddings, (0, padding_size))
        return embeddings

    def resize_keyframe_images(self, keyframe_images, target_size=224):
      """
      Resize keyframe images to the target size expected by CLIP (224x224).
      
      Arguments:
      - keyframe_images: The keyframe image tensor of shape [1, 3, H, W]
      - target_size: The target size (height, width), default is 224 for CLIP.
      
      Returns:
      - Resized keyframe image of shape [1, 3, target_size, target_size]
      """
      # If the target size is different from the current size, resize the image
      resize_transform = transforms.Resize((target_size, target_size))
      
      # Apply the transformation to resize the image
      resized_keyframe_images = resize_transform(keyframe_images.squeeze(0))  # Remove the batch dimension
      
      # Restore the batch dimension
      resized_keyframe_images = resized_keyframe_images.unsqueeze(0)
      
      return resized_keyframe_images


    
    def validate_and_run_model(self, input_ids):
        try:
            # Validate token IDs
            max_token_id = torch.max(input_ids['input_ids'])
            vocab_size = self.phobert_tokenizer.vocab_size
            print(f"Max token ID: {max_token_id}, Vocab size: {vocab_size}")
            
            if max_token_id >= vocab_size:
                raise ValueError(f"Token ID {max_token_id} exceeds vocabulary size {vocab_size}")

            # Validate sequence length
            max_length = self.phobert_model.config.max_position_embeddings
            print(f"Sequence length: {input_ids['input_ids'].size(1)}, Max length: {max_length}")
            
            if input_ids['input_ids'].size(1) > max_length:
                raise ValueError(f"Sequence length {input_ids['input_ids'].size(1)} exceeds max length {max_length}")
            
            # Run model and get outputs
            model_output = self.phobert_model(input_ids['input_ids'].to(self.device))
            return model_output
        
        except Exception as e:
            print(f"Error encountered: {e}")
            return None

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
      try:
          # Check if the data exists in the video_list or another attribute
          if not hasattr(self, 'video_list'):
              raise ValueError(f"No attribute 'video_list' found in dataset")
              
          subdir, video_subdir, keyframe_file = self.video_list[index]
          # Check if the video has already been processed
          if video_subdir in self.processed_videos:
              print(f"Video {video_subdir} already processed. Skipping.")
              return None

          keyframe_path = os.path.join(self.keyframes_dir, subdir, video_subdir, keyframe_file)
          
          keyframe_image = Image.open(keyframe_path).convert('RGB')
          # keyframe_image = keyframe_image.resize((256, 256))  # Resize to a fixed resolution
          keyframe_image = self.clip_preprocess(keyframe_image).unsqueeze(0)

          # Resize the keyframe image to the target size (256)
          keyframe_image = self.resize_keyframe_images(keyframe_image, target_size=224)
          

          print(f"Shape of keyframe_image: {keyframe_image.shape}")
          
          try:
              keyframe_image = keyframe_image.to(self.device)
          except RuntimeError as e:
              print(f"CUDA error during keyframe image processing: {e}")
              raise


          with torch.no_grad():
              clip_features = self.clip_model.encode_image(keyframe_image).squeeze(0).cpu().numpy()

          # Load metadata based on the correct video ID (use `video_subdir` instead of hardcoding)
          metadata_path = os.path.join(self.metadata_dir, f"{video_subdir}.json")
          # Load metadata
          with open(metadata_path, 'r', encoding='utf-8') as f:
              metadata = json.load(f)
          
          # Ensure the metadata file exists for this video
          if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found for video {video_subdir} at {metadata_path}")

          description_text = download_youtube_transcription_from_json(metadata_path, self.config.output_dir)
          if description_text is None:
            print(f"Skipping item {index} due to missing transcription.")
            raise ValueError(f"Could not retrieve transcription from {metadata_path}")
            return None  # Skip this item if transcription is missing
          

          # Tokenize and validate tokens
          input_ids = self.phobert_tokenizer(description_text, return_tensors='pt', 
                                            truncation=True, padding='max_length', 
                                            max_length=self.config.max_token_length)
          print(f"Input token IDs: {input_ids['input_ids']}")

          # Validate token IDs and ensure they are within the vocabulary
          if not self.validate_tokens(input_ids):
            raise ValueError(f"Invalid tokens found in transcription: {description_text}")
          
          # Validate tokens
          for token in input_ids['input_ids'].view(-1):
              if token >= self.phobert_tokenizer.vocab_size:
                  raise ValueError(f"Invalid token ID {token} out of vocab range {self.phobert_tokenizer.vocab_size}")

          input_ids = input_ids.to(self.device)
          # Ensure the inputs and the model are on the same device
          input_ids = {key: val.to(self.device) for key, val in input_ids.items()}
          self.phobert_model = self.phobert_model.to(self.device)  # Move model to the correct device
          print(f"Shape of input_ids['input_ids']: {input_ids['input_ids'].shape}")

          
          if not self.validate_tokens(input_ids):
            raise ValueError(f"Invalid tokens found in description: {description_text}")

          if input_ids['input_ids'].size(1) > self.phobert_model.config.max_position_embeddings:
            logger.error(f"Input ID sequence is too long: {input_ids['input_ids'].size(1)}")

          # Kiểm tra token ID có nằm trong phạm vi từ vựng của PhoBERT không
          max_token_id = torch.max(input_ids['input_ids'])
          vocab_size = self.phobert_tokenizer.vocab_size
          if max_token_id >= vocab_size:
              raise ValueError(f"Token ID {max_token_id} vượt quá kích thước từ vựng {vocab_size}")

          # Kiểm tra chiều dài sequence có vượt quá giới hạn không
          max_length = self.phobert_model.config.max_position_embeddings
          if input_ids['input_ids'].size(1) > max_length:
              raise ValueError(f"Chiều dài sequence {input_ids['input_ids'].size(1)} vượt quá giới hạn {max_length}")

           # Get text embeddings
          with torch.no_grad():
              model_output = self.phobert_model(**input_ids)

          if not hasattr(model_output, 'last_hidden_state'):
              raise ValueError(f"Invalid model output at index {index}: {model_output}")

          description_embeddings = model_output.last_hidden_state

          # Resize the embeddings to the expected size
          description_embeddings = self.resize_embeddings(description_embeddings, self.config.max_token_length)

          print(f"Shape of description embeddings: {description_embeddings.shape}")

          # Get text embeddings
          # Validate tokens and run model
          # model_output = self.validate_and_run_model(input_ids)
          # if model_output is None:
          #     raise ValueError(f"Invalid tokens or sequence at index {index}")


          # # Ensure 'last_hidden_state' is available
          # if not hasattr(model_output, 'last_hidden_state'):
          #     logger.error(f"Invalid model output at index {index}: {model_output}")
          #     return None  # Skip this sample
          # else:
          #     description_embeddings = model_output.last_hidden_state
          #     logger.info(f"Description embeddings shape: {description_embeddings.shape}")

          # Assuming you have keyframe images loaded as 'keyframe_image'
          logger.info(f"Keyframe image shape: {keyframe_image.shape}")
          logger.info(f"Input token IDs shape: {input_ids['input_ids'].shape}")
          logger.info(f"Description embeddings shape: {description_embeddings.shape}")

          # Resize the embeddings to the expected size
          description_embeddings = self.resize_embeddings(description_embeddings, self.config.max_token_length)


          # Get text embeddings
          # Wrap your model forward pass with checkpoint
          # description_embeddings = checkpoint(self.phobert_model, input_ids['input_ids'])

          # Resize the embeddings to the expected size
          # description_embeddings = self.resize_embeddings(description_embeddings, self.config.max_token_length)
          print(f"Shape of description embeddings: {description_embeddings.shape}")

          logger.info(f"Processing item {index}, input size: {input_ids['input_ids'].size()}, vocab size: {self.phobert_tokenizer.vocab_size}")

           # Print the video ID being processed
          print(f"Processing video ID: {video_subdir}, at index: {index}")

          # Save processed keyframe ID
          keyframe_id = f"{subdir}_{video_subdir}_{keyframe_file}"
          save_processed_keyframe(keyframe_id, self.processed_keyframes_file, self.processed_videos)

          return {
              'video_id': video_subdir,
              'keyframe_image': keyframe_image,
              'clip_features': clip_features,
              'metadata': metadata,
              # 'description_embeddings': input_ids['input_ids']
              'description_embeddings': description_embeddings,
          }

          print(f"Keyframe image shape: {keyframe_image.shape}")
          print(f"Description embeddings shape: {description_embeddings.shape}")

          

      except Exception as e:
          logger.error(f"Error processing item {index}: {e}", exc_info=True)
          return None



# Example usage:
if __name__ == "__main__":
    processed_keyframes_file = "/content/drive/MyDrive/AI_Hackkathon/save_processed_keyframes/processed_keyframes.json"
    config = Config()

    # Create dataset object
    dataset = CustomDataset(config, split_type='train')

    print(f"Number of videos in dataset: {len(dataset)}")

    # Load a few examples from the dataset
    for i in range(min(5, len(dataset))):
        data = dataset[i]
        if data is None:
            print(f"Warning: Data at index {i} is None.")
        else:
            print(f"Data at index {i} loaded successfully: {data['video_id']}")

    

