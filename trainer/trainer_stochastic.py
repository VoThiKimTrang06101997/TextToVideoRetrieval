import os
import gc
import time
import torch
import numpy as np
import pandas as pd  # For writing the CSV
from tqdm import tqdm
from collections import defaultdict, deque
from transformers import AutoTokenizer, AutoModel

from config.all_config import gen_log
from config.base_config import Config
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference_stochastic, sim_matrix_inference_stochastic_light_allops, generate_embeds_per_video_id_stochastic, np_softmax

from datasets.aichallenge_dataset import CustomDataset 
from model.clip_stochastic import CLIPStochastic

import torch.nn.functional as F

import logging

from torch.utils.data import DataLoader
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load PhoBERT tokenizer
phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader,
                 valid_data_loader, tokenizer=None, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer

        # Load PhoBERT tokenizer and model
        self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(self.device)

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0
        self.loss_log = []  # Store loss for logging


    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps - 1, self.evals_per_epoch + 1, dtype=int)[1:]

        for batch_idx, data in enumerate(self.train_data_loader):       
            video_ids = data.get('video_ids', 'Unknown')
            logger.info(f"Batch {batch_idx}: Processing video IDs {video_ids}")

            print(f"Data keys: {data.keys()}")  # This will show all keys in the 'data' dict
            logger.info(f"Batch {batch_idx}: Processing video IDs {data['video_id']}")
            
            if data is None:
              logger.warning(f"Batch {batch_idx} is None")
              continue

            logger.info(f"Batch {batch_idx}: Processing video IDs {data['video_id']}")
            
            try:
              # Forward pass
              video = data['clip_features'].to(self.device)
              text_embeds = data['description_embeddings'].to(self.device)

              logger.info(f"Clip features shape: {video.shape}, Text embeddings shape: {text_embeds.shape}")

              # text_embeds, video_embeds_pooled, text_embeds_stochastic, text_mean, text_log_var = self.model(
              #     {'text_embeds': text_embeds, 'video': video}, is_train=True
              # )

              # Ensure both 'text_embeds' and 'video' are present in data
              if 'text_embeds' not in data or 'video' not in data:
                  logger.error(f"Missing required data keys at batch {batch_idx}. Skipping this batch.")
                  continue

              text_inputs = data['text_embeds']  # Replace 'text_embeds' with your actual text input key if necessary
              video_inputs = data['video']       # Replace 'video' with your actual video input key

              # Ensure inputs are on the correct device
              text_inputs = text_inputs.to(self.device)
              video_inputs = video_inputs.to(self.device)

              # Forward pass through the model with both text and video inputs
              text_embeds, video_embeds_pooled, text_embeds_stochastic, text_mean, text_log_var = self.model(
                  text_inputs, video_inputs, is_train=True  # Pass both inputs and other necessary arguments
              )
              
              # Compute loss
              output = sim_matrix_training(text_embeds_stochastic, video_embeds_pooled, self.pooling_type)
              loss = self.loss(output, self.model.clip.logit_scale)

              # Support text embedding regularization
              video_embeds_pooled_avg = torch.mean(video_embeds_pooled, dim=1).squeeze()
              pointer = video_embeds_pooled_avg - text_embeds
              text_support = pointer / pointer.norm(dim=-1, keepdim=True) * torch.exp(text_log_var) + text_embeds
              output_support = sim_matrix_training(text_support, video_embeds_pooled, self.pooling_type)
              loss_support = self.loss(output_support, self.model.clip.logit_scale)

              loss_all = loss + loss_support * self.config.support_loss_weight
              loss_all.backward()

              torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
              self.optimizer.step()
              if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
              

              # Backward pass and optimization step
              loss.backward()
              self.optimizer.step()
              

              torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))
              self.global_step += 1
              total_loss += loss_all.detach().item()

              # Log loss
              if batch_idx % self.log_step == 0:
                msg = (f'Train Epoch: {epoch} dl: {batch_idx}/{num_steps - 1} '
                    f'Total Loss: {loss_all.item()}, Original Loss: {loss.item()}, Support Loss: {loss_support.item()}')
                gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=msg)

              # Perform validation at eval steps
              if batch_idx in eval_steps:
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps - 1)
                self.model.train()
                if val_res['R1-window'] > self.best_window:
                  self.best_window = val_res['R1-window']
                  self._save_checkpoint(epoch, save_best=True)
                if val_res['R1'] > self.best:
                  self.best = val_res['R1']

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}", exc_info=True)

        avg_loss = total_loss / num_steps
        return avg_loss
         
    def _valid_epoch_step(self, epoch, step, num_steps):
      self.model.eval()
      total_val_loss = 0.0
      text_embed_arr = []
      vid_embed_arr = []
      all_vid_ids = []

      with torch.no_grad():
          for idx, data in tqdm(enumerate(self.valid_data_loader)):
              try:
                  # Ensure that the required keys are present in the data
                  required_keys = ['video_id', 'keyframe_image', 'clip_features', 'metadata', 'description_embeddings']
                  for key in required_keys:
                      if key not in data:
                          logger.error(f"Missing required data key: {key} at batch {idx}. Skipping this batch.")
                          continue  # Skip this batch if any required key is missing

                  # Validate data
                  if data is None or any([d is None for d in data.values()]):
                      logger.warning(f"Data at index {idx} is None or invalid. Skipping.")
                      continue

                  # Extract description text and video features
                  description_text = data['description_embeddings']
                  video = data['clip_features'].to(self.device)
                  
                  # Skip batch if description text is invalid
                  if description_text is None or len(description_text.strip()) == 0:
                      logger.error(f"Invalid description text at index {idx}. Skipping this entry.")
                      continue

                  # Tokenize description text using the tokenizer
                  input_ids = self.tokenizer(
                      description_text,
                      return_tensors='pt',
                      truncation=True,
                      padding='max_length',  # Ensure padding to max_length
                      max_length=256         # Set max_length to 256
                  )
                  if not self.validate_tokens(input_ids):
                    print("Invalid tokens or sequence length. Skipping this sample.")

                  # Log the shape of the tokenized input
                  logger.info(f"Tokenized input_ids shape: {input_ids['input_ids'].shape}")                

                  # Check token ID validity before passing to PhoBERT
                  max_token_id = torch.max(input_ids)
                  vocab_size = self.phobert_model.config.vocab_size
                  if torch.max(input_ids) >= vocab_size:
                      raise ValueError(f"Token ID {torch.max(input_ids)} vượt quá kích thước từ vựng {vocab_size}")

                  # Log to check if any token exceeds the vocabulary size
                  if max_token_id >= vocab_size:
                      logger.error(f"Token ID exceeds vocabulary size: {max_token_id} >= {vocab_size}. Skipping batch.")
                      continue
                  else:
                      logger.info(f"All token IDs are within the valid range.")

                  max_length = self.phobert_model.config.max_position_embeddings
                  if input_ids.size(1) > max_length:
                      raise ValueError(f"Input token length {input_ids.size(1)} vượt quá giới hạn {max_length}")

                  self.phobert_model.config.use_nested_tensor = False


                  # Move the tokenized inputs to the appropriate device
                  input_ids = input_ids['input_ids'].to(self.device)

                  # Ensure the input tensor has a valid shape
                  if input_ids.shape[1] > 256:
                      logger.warning(f"Input shape exceeds expected length: {input_ids.shape}. Truncating.")
                      input_ids = input_ids[:, :256]  # Truncate to 256 tokens

                  # Use the tokenized inputs for the PhoBERT model
                  model_output = self.phobert_model(input_ids)
                  
                  if input_ids is None or input_ids.shape[1] == 0:
                    logger.error(f"Invalid input_ids at index {idx}. Skipping this entry.")
                    continue

                  # Ensure 'last_hidden_state' is available
                  if not hasattr(model_output, 'last_hidden_state'):
                      logger.error(f"Invalid model output, missing 'last_hidden_state'. Skipping batch.")
                      continue
                  else:
                      description_embeddings = model_output.last_hidden_state
                      logger.info(f"Description embeddings shape: {description_embeddings.shape}")

                  # Ensure valid sizes before proceeding
                  if description_embeddings.size(0) == 0 or video.size(0) == 0:
                      logger.warning(f"Invalid input size at batch {idx}. Skipping.")
                      continue
                  
                  video = data['clip_features'].to(self.device)

                  # Ensure valid sizes before proceeding
                  if description_embeddings.size(0) == 0 or video.size(0) == 0:
                      logger.warning(f"Invalid input size at batch {idx}. Skipping.")
                      continue

                  # Append embeddings for further processing
                  text_embed_arr.append(description_embeddings.cpu())
                  vid_embed_arr.append(video.cpu())
                  for v_id in data['video_id']:
                      all_vid_ids.append(v_id)

              except Exception as e:
                  logger.error(f"Error processing batch {idx}: {e}")
                  continue

          # If no valid data, return early
          if not text_embed_arr or not vid_embed_arr:
              logger.error("No valid data available for validation. Returning.")
              return {}

          # Concatenate embeddings
          text_embeds = torch.cat(text_embed_arr)
          vid_embeds = torch.cat(vid_embed_arr)

          # Verify sizes before pooling
          logger.info(f"text_embeds shape: {text_embeds.shape}, vid_embeds shape: {vid_embeds.shape}")

          # Pool video embeddings based on video IDs
          vid_embeds_per_video_id = {v_id: vid_embeds[idx] for idx, v_id in enumerate(all_vid_ids)}
          vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])

          # Ensure embeddings are on the CPU before pooling
          self.model.pool_frames.cpu()
          vid_embeds_pooled = self.model.pool_frames(text_embeds, vid_embeds)
          self.model.pool_frames.cuda()

          # Prepare stochastic embeddings
          self.model.stochastic.cpu()
          text_embeds_stochastic_allpairs = torch.zeros(size=(vid_embeds.shape[0], text_embeds.shape[0], text_embeds.shape[1]))

          for (idx_vid, single_vid), single_vid_embed_pooled in tqdm(zip(enumerate(vid_embeds), vid_embeds_pooled)):
              single_vid_vec = single_vid.unsqueeze(0)
              single_vid_repeat = single_vid_vec.tile((text_embeds.shape[0], 1, 1))  # [bs_t, #F, dim]

              all_text_embed_stochastic = []
              for trial in range(self.config.stochastic_trials):
                  all_text_embed_stochastic, _, _ = self.model.stochastic(text_embeds, single_vid_repeat)  # [bs_t, dim]
                  all_text_embed_stochastic.append(all_text_embed_stochastic)

              all_text_embed_stochastic_arr = torch.stack(all_text_embed_stochastic, dim=0)  # [#trials, bs_t, dim]
              all_text_embed_stochastic_arr = all_text_embed_stochastic_arr / all_text_embed_stochastic_arr.norm(dim=-1, keepdim=True)
              single_vid_embed_pooled = single_vid_embed_pooled / single_vid_embed_pooled.norm(dim=-1, keepdim=True)

              sim_select = torch.sum(torch.mul(all_text_embed_stochastic_arr, single_vid_embed_pooled), dim=-1)  # [#trial, bs_t]
              max_indices = torch.argmax(sim_select, dim=0)  # [bs_t]

              selected_plane = torch.ones((all_text_embed_stochastic_arr.shape[1], all_text_embed_stochastic_arr.shape[2]))
              for i in range(all_text_embed_stochastic_arr.shape[1]):
                  selected_plane[i, :] = all_text_embed_stochastic_arr[max_indices[i], i, :]
              text_embeds_stochastic_allpairs[idx_vid, :, :] = selected_plane

          self.model.stochastic.cuda()

          # Generate embeddings per video ID
          text_embeds_per_video_id, vid_embeds_pooled_per_video_id = generate_embeds_per_video_id_stochastic(
              text_embeds_stochastic_allpairs, vid_embeds_pooled, all_vid_ids, self.pooling_type
          )

          # Compute similarity matrix
          if self.config.save_memory_mode:
              sims = sim_matrix_inference_stochastic_light_allops(
                  text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type, self.config.batch_size_split, self.config
              )
          else:
              sims = sim_matrix_inference_stochastic(
                  text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type
              )

          total_val_loss = total_val_loss / len(self.valid_data_loader)
          res = self.metrics(sims)

          # Log results
          for m in res:
              self.window_metric[m].append(res[m])
          for m in self.window_metric:
              res[m + "-window"] = np.mean(self.window_metric[m])

          msg = (f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n"
                f"R@1: {res['R1']} (window: {res['R1-window']})\n"
                f"R@5: {res['R5']} (window: {res['R5-window']})\n"
                f"R@10: {res['R10']} (window: {res['R10-window']})\n"
                f"MedR: {res['MedR']} (window: {res['MedR-window']})\n"
                f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n")
          gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=msg)

          res['loss_val'] = total_val_loss
          return res


    def _save_checkpoint(self, epoch, save_best=False):
        checkpoint_path = os.path.join(self.config.model_path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
        if save_best:
            best_checkpoint_path = os.path.join(self.config.model_path, 'best_model.pth')
            torch.save(self.model.state_dict(), best_checkpoint_path)
            print(f"Best model saved at {best_checkpoint_path}")

    def log_loss_to_csv(self, epoch, train_loss, val_loss):
        log_path = os.path.join(self.config.model_path, 'training_log.csv')
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write('epoch,train_loss,val_loss\n')
        with open(log_path, 'a') as f:
            f.write(f'{epoch},{train_loss},{val_loss}\n')

def tokenize_text(description_text, tokenizer, max_length=256):
    input_ids = tokenizer(description_text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
    return input_ids


import torch
import logging

# Logger setup
logger = logging.getLogger(__name__)

def pad_collate_fn(batch):
   # Filter out 'None' data points
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
      print("Warning: All items in the batch are None. Skipping this batch.")
      return None  # Optionally, return None or raise an error
      
    return torch.utils.data.dataloader.default_collate(batch)

  
    try:
        # Extract clip features, description embeddings, metadata, and video IDs
        keyframe_images = [item['keyframe_image'] for item in batch if item is not None]
        clip_features = [torch.tensor(item['clip_features']) for item in batch if item is not None]
        description_embeddings = [torch.tensor(item['description_embeddings']) for item in batch if item is not None]
        metadata = [item['metadata'] for item in batch if item is not None]
        video_ids = [item['video_id'] for item in batch if item is not None]

        # Determine the maximum length for description embeddings and clip features
        max_clip_len = max([feat.shape[0] for feat in clip_features])
        max_desc_len = max([desc.shape[1] for desc in description_embeddings])

        # Pad or truncate the clip features and description embeddings to the same length
        padded_clip_features = [torch.nn.functional.pad(feat, (0, 0, 0, max_clip_len - feat.shape[0])) for feat in clip_features]
        padded_description_embeddings = [torch.nn.functional.pad(desc, (0, 0, 0, max_desc_len - desc.shape[1])) for desc in description_embeddings]

        # Stack all the tensors into batches
        keyframe_images_batch = torch.stack(keyframe_images)
        clip_features_batch = torch.stack(padded_clip_features)
        description_embeddings_batch = torch.stack(padded_description_embeddings)

        # # Resize embeddings
        # resized_clip_features = [resize_clip_features(feat, target_size=258) for feat in clip_features]
        # resized_description_embeddings = [resize_text_embeddings(desc, target_size=258) for desc in description_embeddings]

        # # Ensure description embeddings have the same shape by padding or truncating
        # # max_len = max([emb.shape[1] for emb in description_embeddings])
        # # description_embeddings = [torch.nn.functional.pad(emb, (0, 0, 0, max_len - emb.shape[1])) for emb in description_embeddings]
        # # description_embeddings = torch.stack(description_embeddings)

        # # Stack all the tensors into batches
        # keyframe_images = torch.stack(keyframe_images)
        # clip_features_batch = torch.stack(resized_clip_features)        
        # description_embeddings_batch = torch.stack(resized_description_embeddings)

        # Print the processed video IDs in the batch
        print(f"Batch contains video IDs: {video_ids}")

        # Return the collated batch
        return {
            'video_ids': video_ids,
            'keyframe_image': keyframe_images_batch,
            'clip_features': clip_features_batch,
            'metadata': metadata,
            'description_embeddings': description_embeddings_batch,
         
        }

    except Exception as e:
        # Log any errors that occur during the collation process
        logger.error(f"Error during collation: {e}")
        return None

def resize_clip_features(clip_features, target_size):
    # Resize or truncate the clip features to the target size
    current_size = clip_features.shape[0]
    if current_size > target_size:
        clip_features = clip_features[:target_size]  # Truncate
    elif current_size < target_size:
        padding_size = target_size - current_size
        clip_features = F.pad(clip_features, (0, 0, 0, padding_size))  # Pad along the appropriate dimensions
    return clip_features

def resize_text_embeddings(text_embeddings, target_size):
    # Resize or pad the text embeddings to the target size
    current_size = text_embeddings.shape[1]
    if current_size > target_size:
        text_embeddings = text_embeddings[:, :target_size]  # Truncate
    elif current_size < target_size:
        padding_size = target_size - current_size
        text_embeddings = F.pad(text_embeddings, (0, 0, 0, padding_size))  # Pad along the appropriate dimensions
    return text_embeddings




if __name__ == "__main__":
    config = Config()

    # Ensure the model path exists
    config.model_path = "/content/drive/MyDrive/AI_Hackkathon/output"
    os.makedirs(config.model_path, exist_ok=True)

    # Initialize datasets
    train_dataset = CustomDataset(config, split_type='train')
    valid_dataset = CustomDataset(config, split_type='valid')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=pad_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=pad_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    # Initialize model, optimizer, and loss function
    model = CLIPStochastic(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        loss=criterion,
        metrics=None,
        optimizer=optimizer,
        config=config,
        train_data_loader=train_loader,
        # valid_data_loader=valid_loader,
        tokenizer=None
    )

    # Training loop
    for epoch in range(config.num_epochs):
        print(f"Training epoch {epoch}...")
        train_loss = trainer._train_epoch(epoch)
        val_loss = trainer._valid_epoch_step(epoch, 0, 1)
        trainer.log_loss_to_csv(epoch, train_loss, val_loss)

    # Save final checkpoint
    final_checkpoint_path = os.path.join(config.model_path, 'final_checkpoint.pth')
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model checkpoint saved at {final_checkpoint_path}")


