import os
import gc
import time
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque

import os
import sys

# Add the path to the source code if needed
sys.path.append('D:/Course_hoc_mien_phi_workshop/AI Hackkathon/SourceCode/TextToVideo')

# Import CustomDataset and transformation functions
from datasets.aichallenge_dataset import CustomDataset  
from config.base_config import Config
from model.clip_stochastic import CLIPStochastic
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference_stochastic, sim_matrix_inference_stochastic_light_allops, generate_embeds_per_video_id_stochastic, np_softmax


class Trainer(BaseTrainer):
    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader,
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None):
        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0

    def _train_epoch(self, epoch):
        self.device = "cpu"  # Force using CPU if memory is a problem
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]

        for batch_idx, data in enumerate(self.train_data_loader):
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
            
            data['video'] = data['video'].to(self.device)

            text_embeds, video_embeds_pooled, text_embeds_stochastic, text_mean, text_log_var = self.model(data, is_train=True)

            output = sim_matrix_training(text_embeds_stochastic, video_embeds_pooled, self.pooling_type)
            loss = self.loss(output, self.model.clip.logit_scale)

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

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))
            self.global_step += 1
            total_loss += loss_all.detach().item()

            if batch_idx in eval_steps:
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
                self.model.train()

                if val_res['R1-window'] > self.best_window:
                    self.best_window = val_res['R1-window']
                    self._save_checkpoint(epoch, save_best=True)

                if val_res['R1'] > self.best:
                    self.best = val_res['R1']

        return {'loss_train': total_loss / num_steps}

    def _valid_epoch_step(self, epoch, step, num_steps):
        self.model.eval()
        text_embed_arr, vid_embed_arr, all_vid_ids = [], [], []
        with torch.no_grad():
            for idx, data in tqdm(enumerate(self.valid_data_loader)):
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

                data['video'] = data['video'].to(self.device)
                text_embed, vid_embed, vid_embed_pooled, text_embed_stochastic = self.model(data, return_all_frames=True, is_train=False)

                text_embed_arr.append(text_embed.cpu())
                vid_embed_arr.append(vid_embed.cpu())

                for v_id in data['video_id']:
                    all_vid_ids.append(v_id)

            text_embeds = torch.cat(text_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)
            vid_embeds_per_video_id = {v_id: vid_embeds[idx] for idx, v_id in enumerate(all_vid_ids)}
            vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])

            self.model.pool_frames.cpu()
            vid_embeds_pooled = self.model.pool_frames(text_embeds, vid_embeds)
            self.model.pool_frames.cuda()

            self.model.stochastic.cpu()
            text_embeds_stochastic_allpairs = torch.zeros(size=(vid_embeds.shape[0], text_embeds.shape[0], text_embeds.shape[1]))

            for idx_vid, single_vid, single_vid_embed_pooled in zip(range(len(vid_embeds)), vid_embeds, vid_embeds_pooled):
                single_vid_vec = single_vid.unsqueeze(0)
                single_vid_repeat = single_vid_vec.tile((text_embeds.shape[0], 1, 1))
                all_text_embed_stochstic = [self.model.stochastic(text_embeds, single_vid_repeat)[0] for _ in range(self.config.stochasic_trials)]
                all_text_embed_stochstic_arr = torch.stack(all_text_embed_stochstic, dim=0)

                all_text_embed_stochstic_arr = all_text_embed_stochstic_arr / all_text_embed_stochstic_arr.norm(dim=-1, keepdim=True)
                single_vid_embed_pooled = single_vid_embed_pooled / single_vid_embed_pooled.norm(dim=-1, keepdim=True)

                sim_select = torch.sum(torch.mul(all_text_embed_stochstic_arr, single_vid_embed_pooled), dim=-1)
                max_indices = torch.argmax(sim_select, dim=0)

                selected_plane = torch.stack([all_text_embed_stochstic_arr[max_indices[i], i] for i in range(all_text_embed_stochstic_arr.shape[1])])
                text_embeds_stochastic_allpairs[idx_vid, :, :] = selected_plane

            self.model.stochastic.cuda()

            text_embeds_per_video_id, vid_embeds_pooled_per_video_id = generate_embeds_per_video_id_stochastic(
                text_embeds_stochastic_allpairs, vid_embeds_pooled, all_vid_ids, self.pooling_type)

            if self.config.save_memory_mode:
                sims = sim_matrix_inference_stochastic_light_allops(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type, self.config.batch_size_split, self.config)
            else:
                sims = sim_matrix_inference_stochastic(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type)

            metrics = self.metrics
            res = metrics(sims)

            return res

import torch
from torch.nn import functional as F

def pad_collate_fn(batch):
    """
    Custom collate function to resize and/or pad images and clip features in a batch to have the same size.
    """
    # Extract all images and other data
    keyframes = [item['keyframe_image'] for item in batch]
    clip_features = [torch.tensor(item['clip_features']) for item in batch]
    metadata = [item['metadata'] for item in batch]
    video_ids = [item['video_id'] for item in batch]
    keyframes_data = [item['keyframes'] for item in batch]

    # Find the largest height and width in the batch for images
    max_height = max([img.shape[1] for img in keyframes])
    max_width = max([img.shape[2] for img in keyframes])

    # Find the largest sequence length in the batch for CLIP features
    max_feature_length = max([features.shape[0] for features in clip_features])

    # Pad all images to the size of the largest image
    padded_keyframes = []
    for img in keyframes:
        padding = (0, max_width - img.shape[2], 0, max_height - img.shape[1])  # (left, right, top, bottom)
        padded_img = F.pad(img, padding, value=0)  # Pad with zeros
        padded_keyframes.append(padded_img)

    # Pad all CLIP features to the length of the largest feature sequence
    padded_clip_features = []
    for features in clip_features:
        pad_size = (0, 0, 0, max_feature_length - features.shape[0])  # Pad along the sequence length dimension
        padded_features = F.pad(features, pad_size, value=0)  # Pad with zeros
        padded_clip_features.append(padded_features)

    # Stack padded images and CLIP features into batch tensors
    keyframe_images = torch.stack(padded_keyframes)
    clip_features_batch = torch.stack(padded_clip_features)

    # Return a dictionary with the batch data
    return {
        'keyframe_images': keyframe_images,
        'clip_features': clip_features_batch,
        'metadata': metadata,
        'video_ids': video_ids,
        'keyframes_data': keyframes_data
    }



# Set up your config, data loading, and start the training
if __name__ == "__main__":
    config = Config()
    
    # Set the output path for checkpoints
    config.model_path = "D:/Course_hoc_mien_phi_workshop/AI Hackkathon/SourceCode/TextToVideo/output"
    os.makedirs(config.model_path, exist_ok=True)  # Ensure output directory exists

    tokenizer = None  # Update this with your tokenizer if needed

    # Prepare the datasets
    train_dataset = CustomDataset(config, split_type='train')
    valid_dataset = CustomDataset(config, split_type='valid')

    # Dataloaders
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4,
    # collate_fn= pad_collate_fn)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, collate_fn= pad_collate_fn)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
                        

    # Initialize model, optimizer, loss function
    model = CLIPStochastic(config)  # Assuming CLIPStochastic is defined
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        loss=criterion,
        metrics=None,  # Update with your custom metrics
        optimizer=optimizer,
        config=config,
        train_data_loader=train_loader,
        valid_data_loader=valid_loader,
        tokenizer=tokenizer
    )

    # Start training
    trainer.train()

    # Save the final model checkpoint
    final_checkpoint_path = os.path.join(config.model_path, 'final_checkpoint.pth')
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model checkpoint saved at {final_checkpoint_path}")
