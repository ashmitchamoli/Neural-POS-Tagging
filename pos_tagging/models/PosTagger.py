import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tag_datasets.TagData import TagDataset
from tag_datasets.DataHandling import AnnPosDataset, RnnPosDataset
from models.ANN import AnnClassifier
from models.RNN import RnnClassifier

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
import hashlib
import pickle

ANN_MODEL_SAVE_DIR = './model_checkpoints/ann/'
RNN_MODEL_SAVE_DIR = './model_checkpoints/rnn/'

_SUCCESS = True
_FAILURE = False

class NeuralPosTagger:
    def __init__(self, trainData : TagDataset, devData : TagDataset, activation : str, embeddingSize : int, batchSize : int) -> None:
        self.trainData = trainData
        self.classes = trainData.classes
        self.vocabulary = trainData.vocabulary
        self.vocabSize = len(self.vocabulary)
        self.numClasses = len(self.classes)
        self.indexClassDict = { value: key for key, value in self.classes.items() }

        self.devData = devData
        self.activation = activation
        self.embeddingSize = embeddingSize
        self.batchSize = batchSize
        self.strConfig = ""
        self.MODEL_SAVE_DIR = ""
 
    def getConfigHash(self):
        return hashlib.sha256(self.strConfig.encode()).hexdigest()

    def saveModel(self) -> None:
        path = None
        try:
            if not os.path.exists(self.MODEL_SAVE_DIR):
                os.makedirs(self.MODEL_SAVE_DIR)
            
            path = f"{self.MODEL_SAVE_DIR}/ann_pos_tagger_{self.getConfigHash()}.pt"

        except Exception as e:
            print("Unable to save model.")
            print(e)
            return _FAILURE
        
        torch.save(self.classifier.state_dict(), path)
        return _SUCCESS

    def loadFromCheckpoint(self) -> None:
        selfHash = self.getConfigHash()

        for filename in os.listdir(self.MODEL_SAVE_DIR):
            f = os.path.join(self.MODEL_SAVE_DIR, filename)

            if os.path.isfile(f):
                modelHash = f.rstrip('.pt').lstrip(os.path.join(self.MODEL_SAVE_DIR, 'ann_pos_tagger_'))
                if selfHash == modelHash:
                    self.classifier.load_state_dict(torch.load(f))
                    return _FAILURE

        return _SUCCESS


class AnnPosTagger(NeuralPosTagger):
    def __init__(self, trainData : TagDataset, devData : TagDataset, contextSize : int = 2, activation : str = 'relu', embeddingSize : int = 128, hiddenLayers : list[int] = [128], batchSize : int = 64) -> None:
        """
        activation : 'relu', 'sigmoid', 'tanh'.
        """
        super().__init__(trainData, devData, activation, embeddingSize, batchSize)

        self.contextSize = contextSize
        self.MODEL_SAVE_DIR = ANN_MODEL_SAVE_DIR

        # classifier
        self.outputSize = len(self.trainData.classes)
        self.hiddenLayers = hiddenLayers
        self.classifier = AnnClassifier(vocabSize=self.vocabSize,
                                        embeddingSize=embeddingSize,
                                        contextSize=self.contextSize,
                                        outChannels=self.outputSize,
                                        hiddenLayers=self.hiddenLayers,
                                        activation=self.activation)

        self.strConfig = str(self.contextSize) + str(self.activation) + str(self.embeddingSize) + str(self.hiddenLayers) + str(self.batchSize) + str(self.outputSize)

    def train(self, epochs : int, learningRate : float) -> None:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learningRate)
        
        self.trainLoss = []
        self.devLoss = []

        trainDataset = AnnPosDataset(self.trainData, self.classes, self.contextSize, self.vocabulary)
        devDataset = AnnPosDataset(self.devData, self.trainData.classes, self.contextSize, self.vocabulary)

        trainLoader = DataLoader(trainDataset, batch_size=self.batchSize)

        for epoch in range(epochs):
            for X_batch, y_batch in trainLoader:
                # forward pass
                outputs = self.classifier(X_batch)

                # calculate loss
                loss = criterion(outputs, y_batch)
                
                # back propagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.trainLoss.append(criterion(self.classifier(trainDataset.X), trainDataset.y).item())
            self.devLoss.append(criterion(self.classifier(devDataset.X), devDataset.y).item())

        self.saveModel()

    def __getContext(self, sentence : list[tuple[int, str, str]], i : int, vocabulary : dict[str, int]) -> list[int]:
        pastContextIds = [0] * max(0, self.contextSize - i) + \
                            [ vocabulary[word[1]] for word in sentence[max(0, i - self.contextSize) : i] ]

        currWordId = vocabulary[sentence[i][1]]

        futureContextIds = [ vocabulary[word[1]] for word in sentence[i+1 : min(len(sentence) - 1, i + self.contextSize + 1)] ]
        futureContextIds = futureContextIds + [0] * max(0, self.contextSize - len(futureContextIds))

        return pastContextIds, currWordId, futureContextIds

    def predict(self, sentence : list[str]) -> list[str]:
        preds = []
        
        newSentence = [ (0, word if word in self.vocabulary else "<UNK>", "") for word in sentence ]
        for i in range(len(newSentence)):

            pastContextIds, currWordId, futureContextIds = self.__getContext(newSentence, i, self.vocabulary)

            x = pastContextIds + [currWordId] + futureContextIds
            x = torch.tensor(x, dtype=torch.long).reshape(1, -1)
            outputs = self.classifier(x)

            y_pred = outputs.argmax()

            preds.append(self.indexClassDict[y_pred.item()])

        return preds

    def evaluateModel(self, testData : TagDataset) -> float:
        preds = []
        y_true = []

        for sentence in testData.dataset:
            newSentence = [ word[1] for word in sentence ]
            tags = [ word[2] for word in sentence ]
            sentencePreds = self.predict(newSentence)
            preds.extend(sentencePreds)
            y_true.extend( tags )
        
        self.classificationReport = classification_report(y_true, preds, zero_division=0)

        # print(self.classificationReport)

        self.scores = {
            'accuracy' : accuracy_score(y_true, preds),
            'precision' : precision_score(y_true, preds, average='weighted', zero_division=0),
            'recall' : recall_score(y_true, preds, average='weighted', zero_division=0),
            'f1' : f1_score(y_true, preds, average='weighted', zero_division=0)
        }

        return self.scores

class RnnPosTagger(NeuralPosTagger):
    def __init__(self, trainData : TagDataset, devData : TagDataset, activation : str = 'relu', embeddingSize : int = 128, hiddenSize : int = 128, numLayers : int = 1, bidirectional : bool = False, linearHiddenLayers : list[int] = [], batchSize : int = 32) -> None:        
        
        super().__init__(trainData, devData, activation, embeddingSize, batchSize)

        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.bidirectional = bidirectional
        self.linearHiddenLayers = linearHiddenLayers

        self.MODEL_SAVE_DIR = RNN_MODEL_SAVE_DIR

        self.classifier = RnnClassifier(vocabSize=self.vocabSize,
                                        embeddingSize=self.embeddingSize,
                                        outChannels=self.numClasses,
                                        hiddenEmbeddingSize=self.hiddenSize,
                                        numLayers=self.numLayers,
                                        activation=self.activation,
                                        bidirectional=self.bidirectional,
                                        linearHiddenLayers=self.linearHiddenLayers)
        
        self.strConfig = str(self.numClasses) + str(self.vocabSize) + self.activation + str(self.embeddingSize) + \
                         str(self.hiddenSize) + str(self.numLayers) + str(self.bidirectional) + str(self.linearHiddenLayers)
    
    def train(self, epochs : int = 5, learningRate : float = 0.001) -> None:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learningRate)

        self.trainLoss = []
        self.devLoss = []

        trainDataset = RnnPosDataset(self.trainData, self.classes, self.vocabulary)
        devDataset = RnnPosDataset(self.devData, self.classes, self.vocabulary)

        trainLoader = DataLoader(trainDataset, batch_size=self.batchSize, collate_fn=trainDataset.collate_fn)
        devLoader = DataLoader(devDataset, batch_size=self.batchSize, collate_fn=devDataset.collate_fn)

        for epoch in range(epochs):
            runningLoss = 0
            for X_batch, y_batch in trainLoader:
                # forward pass
                outputs = self.classifier(X_batch)

                # calculate loss
                loss = criterion(outputs, y_batch)
                runningLoss += loss.item()

                # back prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(outputs.shape, y_batch.shape)
            
            self.trainLoss.append(runningLoss / len(trainLoader))
            for X, y in devLoader:
                outputs = self.classifier(X)
                loss = criterion(outputs, y)
                self.devLoss.append(loss.item())

        self.saveModel()
