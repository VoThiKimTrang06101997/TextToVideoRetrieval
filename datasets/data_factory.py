import os
import sys
from torch.utils.data import DataLoader

# Add the path to the source code if needed
sys.path.append('D:/Course_hoc_mien_phi_workshop/AI Hackkathon/SourceCode/TextToVideo')

# Import CustomDataset and transformation functions
from datasets.aichallenge_dataset import CustomDataset  
from datasets.model_transforms import init_transform_dict

class DataFactory:
    @staticmethod
    def get_data_loader(config, split_type='train'):
        # Initialize image transformations
        img_transforms = init_transform_dict(config.input_res)
        train_img_tfms = img_transforms['clip_train']
        test_img_tfms = img_transforms['clip_test']

        # Use CustomDataset for your custom dataset
        if split_type == 'train':
            print(f"Initializing dataset for training with {split_type}")
            dataset = CustomDataset(config, split_type='train', img_transforms=train_img_tfms)
            if len(dataset) == 0:
                print(f"Dataset for {split_type} is empty!")
                return None
            print(f"Training dataset initialized with {len(dataset)} samples.")
            return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

        elif split_type == 'test':
            print(f"Initializing dataset for testing with {split_type}")
            dataset = CustomDataset(config, split_type='test', img_transforms=test_img_tfms)
            if len(dataset) == 0:
                print(f"Dataset for {split_type} is empty!")
                return None
            print(f"Testing dataset initialized with {len(dataset)} samples.")
            return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        else:
            raise NotImplementedError(f"Dataset {config.dataset_name} not implemented.")

        


