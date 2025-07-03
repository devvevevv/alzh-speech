import torch

class AlzhSpeechNN(torch.nn.Module):
    def __init__(self, input_dim = 45):
        super(AlzhSpeechNN, self).__init__()

        self.linear_stack = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(0.2),

            torch.nn.Linear(64, 128),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(0.4),

            torch.nn.Linear(128, 256),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(0.4),

            torch.nn.Linear(256, 64),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(0.2),

            torch.nn.Linear(64, 2),
            # Use softmax if NOT using cross entropy loss
            # torch.nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.linear_stack(x)

