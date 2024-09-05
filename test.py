import os
import torch
import random
import numpy as np
import pandas as pd
from config.all_config import AllConfig
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.loss import LossFactory
from trainer.trainer_stochastic import Trainer
from config.all_config import gen_log
import json

# @WJM: solve num_workers
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def extract_keyframe_info(map_keyframes_dir, video_id, subdir):
    """
    Extract frame index information from map-keyframes CSV file for a specific video_id.
    Adjusted to reflect the subdirectory structure.
    """
    map_keyframes_path = os.path.join(map_keyframes_dir, subdir, f"{video_id}.csv")
    if os.path.exists(map_keyframes_path):
        df = pd.read_csv(map_keyframes_path)
        return df['frame_idx'].tolist()
    else:
        print(f"No keyframes file found for video_id: {video_id} in subdir: {subdir}")
        return []


def extract_metadata(metadata_dir, video_id):
    """
    Extract description information from metadata JSON file for a specific video_id.
    """
    metadata_path = os.path.join(metadata_dir, f"{video_id}.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata.get('title', 'No description')
    else:
        print(f"No metadata file found for video_id: {video_id}")
        return 'No description'


def main():
    # config
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    writer = None

    # GPU
    if config.gpu is not None and config.gpu != '99':
        print('set GPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception('NO GPU!')

    # @WJM: add log
    msg = f'model pth = {config.model_path}'
    gen_log(model_path=config.model_path, log_name='log_trntst', msg=msg)
    gen_log(model_path=config.model_path, log_name='log_trntst', msg='record all training and testing results')

    # seed
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # CLIP
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)

    # data I/O
    test_data_loader = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config)

    # metric
    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented

    loss = LossFactory.get_loss(config.loss)

    trainer = Trainer(model=model,
                      loss=loss,
                      metrics=metrics,
                      optimizer=None,
                      config=config,
                      train_data_loader=None,
                      valid_data_loader=test_data_loader,
                      lr_scheduler=None,
                      writer=writer,
                      tokenizer=tokenizer)

    if config.load_epoch is not None:
        if config.load_epoch > 0:
            trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
        else:
            trainer.load_checkpoint("model_best.pth")

    # Validate the model and generate predictions
    predictions = trainer.validate()

    # Extract results and save to CSV
    output_file = os.path.join(config.model_path, 'results.csv')
    results = []

    for batch in test_data_loader:
        for video_id in batch['video_id']:
            # Extract subdir (e.g., Videos_L01) from video_id and assume video_id has format like L01_V001
            subdir = f"Keyframes_{video_id[:4]}"

            # Get frame_idx from keyframes
            frame_idxs = extract_keyframe_info(config.map_keyframes_dir, video_id, subdir)
            
            # Get description from metadata
            description = extract_metadata(config.metadata_dir, video_id)
            
            # Add results for each frame_idx
            for frame_idx in frame_idxs:
                results.append({
                    'video_id': video_id,
                    'frame_idx': frame_idx,
                    'description': description
                })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
