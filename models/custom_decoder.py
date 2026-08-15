from src.configs import ModelConfig
from src.models.input_embeddings import InputEmbedding
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, config:ModelConfig):
        super(Decoder, self).__init__()
        self.config = config
        self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                dim_feedforward=config.dim_ffn,
                nhead=config.num_heads,
                batch_first=True,
                #norm_first=True,
                dropout=config.dropout)

        self.encoder = nn.TransformerEncoder(
                encoder_layer=self.encoder_layer,
                num_layers=config.num_layers)

        self.input_embedding = InputEmbedding(
                d_model=config.d_model,
                max_window=config.max_window,
                num_vars=config.num_vars,
                dropout=config.dropout)

        self.output_head = nn.Linear(config.d_model, 1)

    def forward(self, x):
        x = self.input_embedding(x)
        attention_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1))
        x = self.encoder(x, mask=attention_mask, is_causal=True)
        x = self.output_head(x)
        return x
