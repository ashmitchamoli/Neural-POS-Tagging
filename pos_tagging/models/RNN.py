import torch

class RnnClassifier(torch.nn.Module):
    def __init__(self, vocabSize : int, 
                 embeddingSize : int, 
                 outChannels : int, 
                 hiddenEmbeddingSize : int, 
                 numLayers : int = 1, 
                 activation : str = 'relu', 
                 bidirectional : bool = False,
                 linearHiddenLayers : list[int] = []) -> None:
        """
        activation: 'relu', 'tanh'.
        linearHiddenLayers: list of hidden layer sizes for the linear classifier.
        """
        super().__init__()
        
        self.vocabSize = vocabSize
        self.embeddingSize = embeddingSize
        self.outChannels = outChannels
        self.hiddenEmbeddingSize = hiddenEmbeddingSize
        self.numLayers = numLayers
        self.activation = activation
        self.bidirectional = bidirectional
        self.linearHiddenLayers = linearHiddenLayers

        self.embeddingLayer = torch.nn.Embedding(vocabSize, embeddingSize)

        self.rnn = torch.nn.RNN(input_size=self.embeddingSize,
                                hidden_size=self.hiddenEmbeddingSize,
                                num_layers=self.numLayers,
                                nonlinearity=self.activation,
                                bidirectional=self.bidirectional)

        self.linearClassifier = torch.nn.Sequential()
        for i in range(len(self.linearHiddenLayers)):
            linearLayer = torch.nn.Linear( self.hiddenEmbeddingSize * (int(self.bidirectional) + 1) if i == 0 else self.linearHiddenLayers[i-1],
                                               self.outChannels if i == len(self.linearHiddenLayers) - 1 else self.linearHiddenLayers[i] )
            self.linearClassifier.append(linearLayer)
            if i == len(self.linearHiddenLayers) - 1:
                self.linearClassifier.append(torch.nn.Softmax(dim=1))
            else:
                self.linearClassifier.append(torch.nn.ReLU())
        
        if len(self.linearHiddenLayers) == 0:
            linearLayer = torch.nn.Linear( self.hiddenEmbeddingSize * (int(self.bidirectional) + 1),
                                               self.outChannels )
            self.linearClassifier.append(linearLayer)
            self.linearClassifier.append(torch.nn.Softmax(dim=1))

    def forward(self, x):
        x = self.embeddingLayer(x)
        outputs, _ = self.rnn(x)
        return self.linearClassifier(outputs)