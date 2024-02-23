import torch

class BaseClassifier(torch.nn.Module):
    def __init__(self, vocabSize : int, 
                 embeddingSize : int, 
                 outChannels : int, 
                 hiddenEmbeddingSize : int, 
                 numLayers : int,
                 bidirectional : bool,
                 linearHiddenLayers : list[int],
                 activation : str) -> None:
        super().__init__()

        self.vocabSize = vocabSize
        self.embeddingSize = embeddingSize
        self.outChannels = outChannels
        self.hiddenEmbeddingSize = hiddenEmbeddingSize
        self.numLayers = numLayers
        self.bidirectional = bidirectional
        self.linearHiddenLayers = linearHiddenLayers
        self.activation = activation
        
        if self.activation == 'tanh':
            self.activationFn = torch.nn.Tanh()
        elif self.activation == 'relu':
            self.activationFn = torch.nn.ReLU()

        self.embeddingLayer = torch.nn.Embedding(vocabSize, embeddingSize)

        self.linearClassifier = torch.nn.Sequential()
        for i in range(len(self.linearHiddenLayers)):
            linearLayer = torch.nn.Linear( self.hiddenEmbeddingSize * (int(self.bidirectional) + 1) if i == 0 else self.linearHiddenLayers[i-1],
                                               self.outChannels if i == len(self.linearHiddenLayers) - 1 else self.linearHiddenLayers[i] )
            self.linearClassifier.append(linearLayer)
            if i == len(self.linearHiddenLayers) - 1:
                self.linearClassifier.append(torch.nn.Softmax(dim=1))
            else:
                self.linearClassifier.append(self.activationFn)
        
        if len(self.linearHiddenLayers) == 0:
            linearLayer = torch.nn.Linear( self.hiddenEmbeddingSize * (int(self.bidirectional) + 1),
                                               self.outChannels )
            self.linearClassifier.append(linearLayer)
            self.linearClassifier.append(torch.nn.Softmax(dim=1))

class RnnClassifier(BaseClassifier):
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
        super().__init__(vocabSize, embeddingSize, outChannels, hiddenEmbeddingSize, numLayers, bidirectional, linearHiddenLayers, activation)
        
        self.rnn = torch.nn.RNN(input_size=self.embeddingSize,
                                hidden_size=self.hiddenEmbeddingSize,
                                num_layers=self.numLayers,
                                nonlinearity=self.activation,
                                bidirectional=self.bidirectional,
                                batch_first=True)

    def forward(self, x):
        x = self.embeddingLayer(x)
        outputs, _ = self.rnn(x, torch.zeros(size=(self.numLayers * (1 + self.bidirectional), x.shape[0], self.hiddenEmbeddingSize)))
        return self.linearClassifier(outputs)
    
class LstmClassifier(BaseClassifier):
    def __init__(self, vocabSize : int, 
                 embeddingSize : int, 
                 outChannels : int, 
                 hiddenEmbeddingSize : int, 
                 numLayers : int = 1,
                 bidirectional : bool = False,
                 activation : str = 'relu',
                 linearHiddenLayers : list[int] = []) -> None:
        """
        activation: 'relu', 'tanh'. activation to be used in the linear decoder layer.
        linearHiddenLayers: list of hidden layer sizes for the linear classifier.
        """
        super().__init__(vocabSize, embeddingSize, outChannels, hiddenEmbeddingSize, numLayers, bidirectional, linearHiddenLayers, activation)

        self.lstm = torch.nn.LSTM(input_size=self.embeddingSize,
                                  hidden_size=self.hiddenEmbeddingSize,
                                  num_layers=self.numLayers,
                                  batch_first=True,
                                  bidirectional=self.bidirectional)
        
    
    def forward(self, x):
        x = self.embeddingLayer(x)
        # print(x.shape)
        outputs, _ = self.lstm(x) # , torch.zeros(size=(self.numLayers * (1 + self.bidirectional), x.shape[0], self.hiddenEmbeddingSize))
        # print(outputs.shape)
        return self.linearClassifier(outputs)