import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tag_datasets.TagData import TagDataset
from tag_datasets.DataHandling import AnnPosDataset
from models.ANN import AnnClassifier

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score

class NeuralPosTagger:
    def __init__(self, trainData : TagDataset, devData : TagDataset) -> None:
        self.trainData = trainData
        self.devData = devData

class AnnPosTagger(NeuralPosTagger):
    def __init__(self, trainData : TagDataset, devData : TagDataset, contextSize : int = 2, activation : str = 'relu', embeddingSize : int = 128, hiddenLayers : list[int] = [128], batchSize : int = 64) -> None:
        """
        activation : 'relu', 'sigmoid', 'tanh'.
        """
        super().__init__(trainData, devData)

        self.contextSize = contextSize
        self.batchSize = batchSize

        # inverse class dictionary
        self.trainData.indexClassDict = { value: key for key, value in trainData.classes.items() }

        # classifier
        self.outputSize = len(self.trainData.classes)
        self.hiddenLayers = hiddenLayers
        self.activation = activation 
        self.classifier = AnnClassifier(vocabSize=len(self.trainData.vocabulary),
                                        embeddingSize=embeddingSize,
                                        contextSize=self.contextSize,
                                        outChannels=self.outputSize,
                                        hiddenLayers=self.hiddenLayers,
                                        activation=self.activation)
    
    def train(self, epochs : int, learningRate : float) -> None:
        criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(list(self.embeddingLayer.parameters()) + list(self.classifier.parameters()), lr=learningRate)
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learningRate)
        
        self.trainLoss = []
        self.devLoss = []

        trainDataset = AnnPosDataset(self.trainData, self.trainData.classes, self.contextSize)
        devDataset = AnnPosDataset(self.devData, self.trainData.classes, self.contextSize)

        trainLoader = DataLoader(trainDataset, batch_size=self.batchSize)

        # X_train, y_train = self.__prepareData(self.trainData)
        # X_dev, y_dev = self.__prepareData(self.devData)
        
        # print(X_train.shape, y_train.shape)
        # print(X_dev.shape, y_dev.shape)

        for epoch in range(epochs):
            for X_batch, y_batch in trainLoader:
                # forward pass
                outputs = self.classifier(X_batch)
                
                # calculate loss
                loss = criterion(outputs, y_batch)
                self.trainLoss.append(loss.item())

                
                # back propagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # # forward pass
            # outputs = self.classifier(X_train)
            
            # # calculate loss
            # loss = criterion(outputs, y_train)
            # self.trainLoss.append(loss.item())

            # # calculate dev loss
            # # devOutputs = self.classifier(X_dev)
            # # devLoss = criterion(devOutputs, y_dev)
            # # # devLoss = criterion(self.classifier(X_dev), y_dev)
            # # self.devLoss.append(devLoss.item())

            # # back propagate
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

    def __getContext(self, sentence : list[tuple[int, str, str]], i : int, vocabulary : dict[str, int]) -> list[int]:
        pastContextIds = [0] * max(0, self.contextSize - i) + \
                            [ vocabulary[word[1]] for word in sentence[max(0, i - self.contextSize) : i] ]

        currWordId = vocabulary[sentence[i][1]]

        futureContextIds = [ vocabulary[word[1]] for word in sentence[i+1 : min(len(sentence) - 1, i + self.contextSize + 1)] ]
        futureContextIds = futureContextIds + [0] * max(0, self.contextSize - len(futureContextIds))

        return pastContextIds, currWordId, futureContextIds

    def predict(self, sentence : list[str]) -> list[str]:
        preds = []
        
        newSentence = [ (0, word if word in self.trainData.vocabulary else "<UNK>", "") for word in sentence ]
        for i in range(len(newSentence)):

            pastContextIds, currWordId, futureContextIds = self.__getContext(newSentence, i, self.trainData.vocabulary)

            x = pastContextIds + [currWordId] + futureContextIds
            x = torch.tensor(x, dtype=torch.long).reshape(1, -1)
            outputs = self.classifier(x)

            y_pred = outputs.argmax()

            preds.append(self.trainData.indexClassDict[y_pred.item()])

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

        print(self.classificationReport)

        self.scores = {
            'accuracy' : accuracy_score(y_true, preds),
            'precision' : precision_score(y_true, preds, average='weighted'),
            'recall' : recall_score(y_true, preds, average='weighted'),
            'f1' : f1_score(y_true, preds, average='weighted')
        }

        return self.scores

class RnnPosTagger(NeuralPosTagger):
    def __init__(self, trainData : TagDataset, devData : TagDataset) -> None:
        super().__init__(trainData, devData)
