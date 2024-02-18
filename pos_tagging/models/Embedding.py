import torch

class Embedding(torch.nn.Module):
    def __init__(self, vocabSize : int, embeddingSize : int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabSize, embeddingSize)

    def forward(self, x):
        """
        x: one-hot vector of the word
        """
        return self.embedding(torch.argmax(x, dim=1))