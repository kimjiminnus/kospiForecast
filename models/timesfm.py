class ZeroShotTimesFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TimesFm2_5ModelForPrediction.from_pretrained("google/timesfm-2.5-200m-transformers")

    def forward(self, x:Tensor):
        forecast = self.model(x).mean_predictions
        return forecast[:,:1]
