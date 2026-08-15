from src.configs import ModelConfig
from src.models.timesfm import ZeroShotTimesFM
from src.models.custom_decoder import Decoder


def create_model(model_type:str, model_config: ModelConfig):

    if model_type.upper() == "DECODER":
        return Decoder(model_config)

    elif model_type.upper() == "TIMESFM":
        return ZeroShotTimesFM()
        
    else:
        raise ValueError("Unknown model type")
