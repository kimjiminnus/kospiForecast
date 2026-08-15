def create_model(model_type:str, model_config: ModelConfig):

    if model_type.upper() == "DECODER":
        return Decoder(model_config)

    elif model_type.upper() == "TIMESFM":
        return ZeroShotTimesFM()

    elif model_type.upper() == "LSTM":
        return LSTM(model_config)
    else:
        raise ValueError("Unknown model type")
