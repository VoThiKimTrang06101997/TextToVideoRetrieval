from torchvision import transforms
from PIL import Image

def init_transform_dict(input_res=224):
    tsfm_dict = {
        'clip_test': transforms.Compose([
            transforms.Resize(input_res, interpolation=Image.BICUBIC),
            transforms.CenterCrop(input_res),
            transforms.ToTensor(),  # Chuyển đổi ảnh sang tensor
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]),
        'clip_train': transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.2),  # Điều chỉnh độ sáng, độ bão hòa, và sắc độ
            transforms.ToTensor(),  # Chuyển đổi ảnh sang tensor
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    }

    return tsfm_dict
