import torch.nn as nn
import torch
from transformers import RobertaModel, RobertaTokenizer, CLIPModel
from modules.transformer import Transformer
from modules.stochastic_module import StochasticText
from config.base_config import Config
from modules.tokenization_clip import CLIPTokenizer
import torch.nn.functional as F

import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'


class CLIPStochastic(nn.Module):
    def __init__(self, config: Config):
        super(CLIPStochastic, self).__init__()
        self.config = config

        # Load CLIP model for video feature extraction
        if config.clip_arch == 'ViT-B/32':
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        elif config.clip_arch == 'ViT-B/16':
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        else:
            raise ValueError("Unsupported CLIP architecture")
        
        # Load PhoBERT using RobertaModel for Vietnamese text processing
        phobert_model_path = "vinai/phobert-base-v2"  # The pre-trained PhoBERT model path from HuggingFace
        vocab_file = "/content/drive/MyDrive/AI_Hackkathon/PhoBert/CLIP/vocab.txt"
        merges_file = "/content/drive/MyDrive/AI_Hackkathon/PhoBert/CLIP/bpe.codes"

        # Load the tokenizer and model from the specified local paths
        self.phobert_tokenizer = RobertaTokenizer(vocab_file=vocab_file, merges_file=merges_file)
        self.phobert = RobertaModel.from_pretrained(phobert_model_path)


        # Pooling and stochastic modules
        self.pool_frames = Transformer(config)  # Custom Transformer for pooling
        self.stochastic = StochasticText(config)  # Stochastic text embeddings

    def resize_text_embeddings(self, text_embeddings, target_size=258):
        """
        Resize or pad the text embeddings to the target size.
        """
        current_size = text_embeddings.shape[1]
        if current_size > target_size:
            text_embeddings = text_embeddings[:, :target_size]  # Truncate
        elif current_size < target_size:
            padding_size = target_size - current_size
            text_embeddings = F.pad(text_embeddings, (0, padding_size), "constant", 0)  # Pad with zeros
        return text_embeddings

    def resize_clip_features(self, clip_features, target_size=258):
        """
        Resize or truncate the clip features to the target size.
        """
        current_size = clip_features.shape[0]
        if current_size > target_size:
            clip_features = clip_features[:target_size]  # Truncate
        elif current_size < target_size:
            clip_features = F.interpolate(clip_features.unsqueeze(0), size=(target_size,), mode='linear', align_corners=False).squeeze(0)
        return clip_features

    # def validate_tokens(self, input_ids):
    #   """
    #   Validates token IDs and checks if they are within the vocabulary size.
    #   Also checks if the sequence length exceeds the model's max position embeddings.
    #   """
    #   # Get the vocab size from the PhoBERT tokenizer
    #   vocab_size = self.phobert_tokenizer.vocab_size
    #   print(f"Vocab size: {vocab_size}, Input token IDs: {input_ids['input_ids']}")

    #   # Validate each token
    #   for token in input_ids['input_ids'].view(-1):
    #       if token >= vocab_size or token < 0:
    #           print(f"Invalid token {token} out of vocab size {vocab_size}")
    #           return False
      
    #   # Check if any token exceeds the vocabulary size
    #   max_token_id = torch.max(input_ids['input_ids'])
    #   if max_token_id >= vocab_size:
    #       raise ValueError(f"Token ID {max_token_id} exceeds vocabulary size {vocab_size}")
      
    #   # Check if the sequence length exceeds the max position embeddings of the model
    #   max_length = self.phobert_model.config.max_position_embeddings
    #   if input_ids['input_ids'].size(1) > max_length:
    #       raise ValueError(f"Sequence length {input_ids['input_ids'].size(1)} exceeds max length {max_length}")

    #   return True

    def validate_tokens(self, input_ids):
      vocab_size = self.phobert_tokenizer.vocab_size
      max_token_id = torch.max(input_ids['input_ids'])
      if max_token_id >= vocab_size:
          raise ValueError(f"Invalid token {max_token_id} exceeds vocabulary size {vocab_size}")

      max_length = self.phobert_tokenizer.model_max_length
      if input_ids['input_ids'].size(1) > max_length:
          raise ValueError(f"Sequence length {input_ids['input_ids'].size(1)} exceeds max length {max_length}")

      return True


    def forward(self, data, return_all_frames=False, is_train=True):
        batch_size = data['video'].shape[0]

        # Process text data using PhoBERT
        description_text = data['description']

        # Tokenize Vietnamese text using PhoBERT
        max_length = 256  # Define max_length for PhoBERT tokenization
        text_inputs = self.phobert_tokenizer(
            description_text,
            return_tensors="pt",
            truncation=True,  # Enable truncation
            padding=True,     # Enable padding
            max_length=256  # Specify max_length explicitly
        )

        # Move tokenized inputs to device
        text_inputs = {key: val.to(self.config.device) for key, val in text_inputs.items()}

        # Pass tokenized input to PhoBERT model to get embeddings
        text_embeddings = self.phobert(**text_inputs).last_hidden_state

        # Resize text embeddings to a fixed size
        text_embeddings = self.resize_text_embeddings(text_embeddings, target_size=258)
        
        # Print the text embeddings shape for debugging
        print(f"Text embeddings shape after resizing: {text_embeddings.shape}")

        # Process video data using CLIP
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)  # Reshape for CLIP's input format
        
        # Extract video features using CLIP
        video_features = self.clip.get_image_features(video_data)
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        # Resize video features to match text embedding dimensions
        video_features = self.resize_clip_features(video_features, target_size=258)

        # Print video features shape for debugging
        print(f"Video features shape after resizing: {video_features.shape}")

        # Pool frames and apply stochastic text embeddings
        if is_train:
            video_features_pooled = self.pool_frames(text_embeddings, video_features)
            text_features_stochastic, text_mean, log_var = self.stochastic(text_embeddings, video_features)

            if return_all_frames:
                return text_embeddings, video_features, video_features_pooled, text_features_stochastic, text_mean, log_var

            return text_embeddings, video_features_pooled, text_features_stochastic, text_mean, log_var

        else:
            # Pool frames for evaluation
            video_features_pooled = self.pool_frames(text_embeddings, video_features)
            text_features_stochastic, _, _ = self.stochastic(text_embeddings, video_features)

            if return_all_frames:
                return text_embeddings, video_features, video_features_pooled, text_features_stochastic

            return text_embeddings, video_features_pooled, text_features_stochastic

        print(f"Input token IDs shape: {data['input_ids'].shape}")
        print(f"Keyframe image shape: {data['keyframe_image'].shape}")

