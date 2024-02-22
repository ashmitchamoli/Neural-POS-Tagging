import torch

class AnnClassifier(torch.nn.Module):
    def __init__(self, vocabSize : int, 
                 embeddingSize : int, 
                 futureContextSize : int, 
                 pastContextSize : int,
                 outChannels : int, 
                 hiddenLayers : list[int], 
                 activation : str) -> None:
        super().__init__()

        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh()

        self.embeddingLayer = torch.nn.Embedding(vocabSize, embeddingSize)

        inChannels = embeddingSize * (pastContextSize + futureContextSize + 1)
        self.feedForward = torch.nn.Sequential()
        self.nHiddenLayers = len(hiddenLayers)
        for i in range(self.nHiddenLayers):
            linearLayer = torch.nn.Linear(inChannels if i == 0 else hiddenLayers[i-1],
                                          outChannels if i == self.nHiddenLayers - 1 else hiddenLayers[i])

            self.feedForward.append(linearLayer)

            if i == self.nHiddenLayers - 1:
                self.feedForward.append(torch.nn.Softmax(dim=1))
            else:
                self.feedForward.append(self.activation)

        if self.nHiddenLayers == 0: # the above loop will not run
            linearLayer = torch.nn.Linear(inChannels, outChannels)
            self.feedForward.append(linearLayer)
            self.feedForward.append(torch.nn.Softmax(dim=1))

    def forward(self, x):
        x = self.embeddingLayer(x)
        # flatten x into va 1d arrray
        x = x.view(x.size(0), -1)
        return self.feedForward(x)