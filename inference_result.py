import pandas as pd
import json
import os
from config.base_config import Config
from model.clip_stochastic import CLIPStochastic
import torch
from datasets.aichallenge_dataset import CustomDataset

class InferenceHandler:
    def __init__(self, config):
        self.map_keyframes_dir = config.map_keyframes_dir
        self.metadata_dir = config.metadata_dir
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_inference(self, model, dataset, tokenizer=None):
        """
        Perform inference and save the output results to an Excel file.
        """
        # Placeholder for storing inference results
        results = []

        # Iterate over the dataset for inference
        for i in range(len(dataset)):
            data = dataset[i]
            video_id = data['video_id']
            keyframe_image = data['keyframe_image'].unsqueeze(0)  # Add batch dimension
            
            # Perform model inference
            with torch.no_grad():
                # Assuming your model's forward pass returns relevant embeddings
                if tokenizer is not None:
                    data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                output = model(data)

            # Retrieve the frame indices from the map-keyframes CSV
            map_keyframes_path = os.path.join(self.map_keyframes_dir, f"{video_id}.csv")
            frame_indices = self._load_frame_indices(map_keyframes_path)

            # Retrieve the metadata (description) from the metadata JSON
            metadata_path = os.path.join(self.metadata_dir, f"{video_id}.json")
            description = self._load_metadata_description(metadata_path)

            # Append the result to the list (video_id, frame_idx, description, inference_output)
            for frame_idx in frame_indices:
                results.append({
                    'video_id': video_id,
                    'frame_idx': frame_idx,
                    'description': description,
                    'inference_output': output.cpu().numpy()  # Example of adding inference output
                })

        # Convert the results to a DataFrame and save as Excel
        results_df = pd.DataFrame(results)
        excel_output_path = os.path.join(self.output_dir, "inference_results.xlsx")
        results_df.to_excel(excel_output_path, index=False)
        print(f"Inference results saved at: {excel_output_path}")

    def _load_frame_indices(self, map_keyframes_path):
        """
        Load frame indices from the map-keyframes CSV file.
        """
        try:
            df = pd.read_csv(map_keyframes_path)
            frame_indices = df['frame_idx'].tolist()  # Assuming 'frame_idx' column exists
            return frame_indices
        except FileNotFoundError:
            print(f"Map keyframes file not found: {map_keyframes_path}")
            return []

    def _load_metadata_description(self, metadata_path):
        """
        Load description from the metadata JSON file.
        """
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            description = metadata.get('description', '')  # Assuming 'description' key exists
            return description
        except FileNotFoundError:
            print(f"Metadata file not found: {metadata_path}")
            return "No description available"

# After training, you can use the inference handler like this:
if __name__ == "__main__":
    config = Config()
    tokenizer = None  # Define or load your tokenizer if necessary

    # Load the trained model (ensure it's in evaluation mode)
    model = CLIPStochastic(config)
    model.load_state_dict(torch.load("path_to_trained_checkpoint.pth"))
    model.eval()

    # Load dataset for inference
    inference_dataset = CustomDataset(config, split_type='valid')

    # Initialize the inference handler
    inference_handler = InferenceHandler(config)

    # Run inference and output results
    inference_handler.run_inference(model, inference_dataset, tokenizer=tokenizer)
