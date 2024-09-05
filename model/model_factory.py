import sys
import os

sys.path.append('D:/Course_hoc_mien_phi_workshop/AI Hackkathon/SourceCode/TextToVideo')

from config.base_config import Config
from model.clip_stochastic import CLIPStochastic

# model_factory.py
from model.clip_transfomer import CLIPTransformer  # Assuming this is the new model


class ModelFactory:
    @staticmethod
    def get_model(config):
        if config.arch == 'clip_transformer':
            return CLIPTransformer(config)  # Newly added model
        elif config.arch == 'clip_stochastic':
            return CLIPStochastic(config)
        else:
            raise NotImplementedError(f"Model architecture {config.arch} not implemented.")

