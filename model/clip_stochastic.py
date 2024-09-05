import torch.nn as nn

import sys
import os

sys.path.append('D:/Course_hoc_mien_phi_workshop/AI Hackkathon/SourceCode/TextToVideo')
from config.base_config import Config

from modules.transformer import Transformer
from modules.stochastic_module import StochasticText

# Import RobertaModel and RobertaTokenizer for PhoBERT
from transformers import RobertaModel, RobertaTokenizer

class CLIPStochastic(nn.Module):
    def __init__(self, config: Config):
        super(CLIPStochastic, self).__init__()
        self.config = config
        
        from transformers import CLIPModel
        
        # Load CLIP model for video feature extraction
        if config.clip_arch == 'ViT-B/32':
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        elif config.clip_arch == 'ViT-B/16':
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        else:
            raise ValueError("Unsupported CLIP architecture")
        
        # Load PhoBERT using RobertaModel for Vietnamese text processing
        self.tokenizer = RobertaTokenizer.from_pretrained("vinai/phobert-base")
        self.phobert = RobertaModel.from_pretrained("vinai/phobert-base")
        
        # Pooling and stochastic modules
        config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)
        self.stochastic = StochasticText(config)

    def forward(self, data, return_all_frames=False, is_train=True):
        batch_size = data['video'].shape[0]
        
        # Process text data using PhoBERT
        text_data = data['text']
        inputs = self.tokenizer(text_data, return_tensors="pt", padding=True, truncation=True, max_length=512)
        text_features = self.phobert(**inputs).last_hidden_state  # Using PhoBERT for text embeddings
        
        # Process video data using CLIP
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        video_features = self.clip.get_image_features(video_data)
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)  # [bs, #F, 512]

        if is_train:
            # Pool frames and apply stochastic text
            video_features_pooled = self.pool_frames(text_features, video_features)
            text_features_stochastic, text_mean, log_var = self.stochastic(text_features, video_features)

            if return_all_frames:
                return text_features, video_features, video_features_pooled, text_features_stochastic, text_mean, log_var

            return text_features, video_features_pooled, text_features_stochastic, text_mean, log_var

        else:
            # Pool frames for evaluation
            video_features_pooled = self.pool_frames(text_features, video_features)
            text_features_stochastic, _, _ = self.stochastic(text_features, video_features)

            if return_all_frames:
                return text_features, video_features, video_features_pooled, text_features_stochastic

            return text_features, video_features_pooled, text_features_stochastic
