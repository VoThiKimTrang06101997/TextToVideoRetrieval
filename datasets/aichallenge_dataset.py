import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
import clip
from transformers import AutoTokenizer, AutoModel
import logging
import torch.nn.functional as F

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
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print("CLIP model loaded successfully!")

# Load PhoBERT tokenizer and model
phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2")
phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")


class Config:
    def __init__(self):
        self.keyframes_dir = "/content/drive/MyDrive/AI_Hackkathon/Dataset/keyframes"
        self.metadata_dir = "/content/drive/MyDrive/AI_Hackkathon/Dataset/metadata"
        self.map_keyframes_dir = "/content/drive/MyDrive/AI_Hackkathon/Dataset/map-keyframes"
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

class CustomDataset(Dataset):
    def __init__(self, config, split_type='train', img_transforms=None, clip_model_name="ViT-B/32"):
        self.config = config
        self.split_type = split_type
        self.img_transforms = img_transforms

        # Load CLIP model and preprocess
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set to GPU if available
        print(f"Using device: {self.device}")

        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)

        # Load PhoBERT model and tokenizer
        self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2")
        
        # Define directories for your dataset
        self.keyframes_dir = config.keyframes_dir
        self.metadata_dir = config.metadata_dir
        self.map_keyframes_dir = config.map_keyframes_dir

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
          subdir, video_subdir, keyframe_file = self.video_list[index]
          keyframe_path = os.path.join(self.keyframes_dir, subdir, video_subdir, keyframe_file)
          
          keyframe_image = Image.open(keyframe_path).convert('RGB')
          keyframe_image = self.clip_preprocess(keyframe_image).unsqueeze(0)

          print(f"Shape of keyframe_image: {keyframe_image.shape}")
          
          try:
              keyframe_image = keyframe_image.to(self.device)
          except RuntimeError as e:
              print(f"CUDA error during keyframe image processing: {e}")
              raise


          with torch.no_grad():
              clip_features = self.clip_model.encode_image(keyframe_image).squeeze(0).cpu().numpy()

          metadata_path = os.path.join(self.metadata_dir, f"{video_subdir}.json")
          # Load metadata
          with open(metadata_path, 'r', encoding='utf-8') as f:
              metadata = json.load(f)

          description_text = metadata.get('description', '')


          # Tokenize and validate tokens
          input_ids = self.phobert_tokenizer(description_text, return_tensors='pt', 
                                            truncation=True, padding=True, 
                                            max_length=self.config.max_token_length)
          print(f"Input token IDs: {input_ids['input_ids']}")

          # Validate token IDs and ensure they are within the vocabulary
          if not self.validate_tokens(input_ids):
            raise ValueError(f"Invalid tokens in input: {input_ids}")
          
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


          return {
              'video_id': video_subdir,
              'keyframe_image': keyframe_image,
              'clip_features': clip_features,
              'metadata': metadata,
              'description_embeddings': description_embeddings,
          }

          print(f"Keyframe image shape: {keyframe_image.shape}")
          print(f"Description embeddings shape: {description_embeddings.shape}")

      except Exception as e:
          logger.error(f"Error processing item {index}: {e}", exc_info=True)
          return None



# Example usage:
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

    

