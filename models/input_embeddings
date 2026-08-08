class VectorEmbedding(nn.Module):
    def __init__(self, num_vars, emb_size):
        super(VectorEmbedding, self).__init__()
        self.linear = nn.Linear(num_vars, emb_size)

    def forward(self, x):
        return self.linear(x)



class PositionalEncoding(nn.Module):
    def __init__(self ,emb_size:int, max_window:int, dropout:float):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_window).reshape(max_window, 1)
        pos_embedding = torch.zeros((max_window, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.register_buffer("pos_embedding", pos_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vector_embedding:Tensor, window:int):
        return self.dropout(vector_embedding + self.pos_embedding[:, :window,:])



class InputEmbedding(nn.Module):
    def __init__(self, d_model, max_window, num_vars, dropout:float):
        super(InputEmbedding, self).__init__()

        self.embedding = VectorEmbedding(num_vars, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_window, dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x, x.size(1))
        return x
