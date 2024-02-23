import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tag_datasets.TagData import TagDataset
from tag_datasets.DataHandling import AnnPosDataset, RnnPosDataset
from models.ANN import AnnClassifier
from models.RNN import RnnClassifier, LstmClassifier

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import hashlib
import pickle

ANN_MODEL_SAVE_DIR = './model_checkpoints/ann/'
RNN_MODEL_SAVE_DIR = './model_checkpoints/rnn/'
LSTM_MODEL_SAVE_DIR = './model_checkpoints/lstm/'

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
                    return _SUCCESS

        return _FAILURE

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

        self.confusionMatrix = confusion_matrix(y_true, preds)

        return self.scores

class AnnPosTagger(NeuralPosTagger):
    def __init__(self, trainData : TagDataset, devData : TagDataset, futureContextSize : int = 2, pastContextSize : int = 2, activation : str = 'relu', embeddingSize : int = 128, hiddenLayers : list[int] = [128], batchSize : int = 64) -> None:
        """
        activation : 'relu', 'sigmoid', 'tanh'.
        """
        super().__init__(trainData, devData, activation, embeddingSize, batchSize)

        self.futureContextSize = futureContextSize
        self.pastContextSize = pastContextSize

        self.MODEL_SAVE_DIR = ANN_MODEL_SAVE_DIR

        # classifier
        self.outputSize = len(self.trainData.classes)
        self.hiddenLayers = hiddenLayers
        self.classifier = AnnClassifier(vocabSize=self.vocabSize,
                                        embeddingSize=embeddingSize,
                                        futureContextSize=self.futureContextSize,
                                        pastContextSize=self.pastContextSize,
                                        outChannels=self.outputSize,
                                        hiddenLayers=self.hiddenLayers,
                                        activation=self.activation)

        self.strConfig = str(self.futureContextSize) + str(self.pastContextSize) + str(self.activation) + str(self.embeddingSize) + str(self.hiddenLayers) + str(self.batchSize) + str(self.outputSize)

    def train(self, epochs : int, learningRate : float, verbose : bool = False, retrain : bool = False) -> None:
        if retrain == False:
            if self.loadFromCheckpoint() == _SUCCESS:
                if verbose:
                    print("Model loaded from checkpoint.")
                return
            else:
                if verbose:
                    print("Model checkpoint not found. Commencing training...")

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learningRate)
        
        self.trainLoss = []
        self.devLoss = []
        self.devAccuracy = []

        trainDataset = AnnPosDataset(self.trainData, self.classes, self.futureContextSize, self.pastContextSize, self.vocabulary)
        devDataset = AnnPosDataset(self.devData, self.trainData.classes, self.futureContextSize, self.pastContextSize, self.vocabulary)

        trainLoader = DataLoader(trainDataset, batch_size=self.batchSize)
        devLoader = DataLoader(devDataset, batch_size=len(devDataset))

        for epoch in range(epochs):
            runningLoss = 0
            for X_batch, y_batch in trainLoader:
                # forward pass
                outputs = self.classifier(X_batch)

                # calculate loss
                loss = criterion(outputs, y_batch)
                runningLoss += loss.item() 

                # back propagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.trainLoss.append(runningLoss / len(trainLoader))

            self.devAccuracy.append(self.evaluateModel(self.devData)['accuracy'])
            if verbose:
                print(f"Epoch {epoch + 1}: train loss = {round(self.trainLoss[-1], 4)}, dev loss = {round(self.devLoss[-1], 4)}, dev accuracy = {round(self.devAccuracy[-1], 4)}")


        self.saveModel()

    def __getContext(self, sentence : list[tuple[int, str, str]], i : int, vocabulary : dict[str, int]) -> list[int]:
        pastContextIds = [0] * max(0, self.pastContextSize - i) + \
                            [ vocabulary[word[1]] for word in sentence[max(0, i - self.pastContextSize) : i] ]

        currWordId = vocabulary[sentence[i][1]]

        futureContextIds = [ vocabulary[word[1]] for word in sentence[i+1 : min(len(sentence) - 1, i + self.futureContextSize + 1)] ]
        futureContextIds = futureContextIds + [0] * max(0, self.futureContextSize - len(futureContextIds))

        return pastContextIds, currWordId, futureContextIds

    def predict(self, sentence : list[str]) -> list[str]:
        preds = []
        
        newSentence = [ (0, word if word in self.vocabulary else "<UNK>", "") for word in sentence ]
        for i in range(len(newSentence)):

            pastContextIds, currWordId, futureContextIds = self.__getContext(newSentence, i, self.vocabulary)

            x = pastContextIds + [currWordId] + futureContextIds
            
            with torch.no_grad():
                x = torch.tensor(x, dtype=torch.long).reshape(1, -1)
                outputs = self.classifier(x)

            y_pred = outputs.argmax()

            preds.append(self.indexClassDict[y_pred.item()])

        return preds

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
    
    def train(self, epochs : int = 5, learningRate : float = 0.001, verbose : bool = False, retrain : bool = False) -> None:

        if retrain == False:
            if self.loadFromCheckpoint() == _SUCCESS:
                if verbose:
                    print("Model loaded from checkpoint.")
                return
            else:
                if verbose:
                    print("Model checkpoint not found. Commencing training...")
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learningRate)

        self.trainLoss = []
        self.devLoss = []
        self.devAccuracy = []

        trainDataset = RnnPosDataset(self.trainData, self.classes, self.vocabulary)
        devDataset = RnnPosDataset(self.devData, self.classes, self.vocabulary)

        trainLoader = DataLoader(trainDataset, batch_size=self.batchSize, collate_fn=trainDataset.collate_fn)
        devLoader = DataLoader(devDataset, batch_size=len(devDataset), collate_fn=devDataset.collate_fn)

        for epoch in range(epochs):
            self.trainLoss.append(0)
            for sentence in self.trainData.dataset:
                tokenIds = []
                tagIds = []
                for word in sentence:
                    tokenIds.append(self.vocabulary[word[1]])
                    tagIds.append(self.classes[word[2]])
                
                x = torch.tensor(tokenIds, dtype=torch.long).reshape(1, -1)
                y = torch.zeros(size=(len(tagIds), self.numClasses))
                for i in range(len(tagIds)):
                    y[i][tagIds[i]] = 1
                
                # forward pass
                outputs = self.classifier(x)[0]

                # print(outputs.shape, outputs[0][0])
                # print(y.shape, y[0][0])

                # calculate loss
                loss = criterion(outputs, y)
                self.trainLoss[-1] += loss.item()

                # back prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.trainLoss[-1] /= len(self.trainData.dataset)       
            
            # ! see what's wrong with the data loader and the RnnPosDataset class inside /pos_tagging/tag_datasets/DataHandling.py
            # runningLoss = 0
            # for X_batch, y_batch in trainLoader:
            #     # forward pass
            #     outputs = self.classifier(X_batch)

            #     # calculate loss
            #     loss = criterion(outputs, y_batch)
            #     runningLoss += loss.item()

            #     # back prop
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            
            # self.trainLoss.append(runningLoss / len(trainLoader))

            for X, y in devLoader:
                outputs = self.classifier(X)
                loss = criterion(outputs, y)
                self.devLoss.append(loss.item())

            devAccuracy = self.evaluateModel(self.devData)['accuracy']
            self.devAccuracy.append(devAccuracy)
            if verbose:
                print(f"Epoch {epoch+1} | Train Loss: {self.trainLoss[-1]:.3f} | Dev Accuracy: {devAccuracy:.3f}")

        self.saveModel()

    def predict(self, sentence : list[str]) -> list[str]:
        tokenIds = []
        for i in range(len(sentence)):
            if sentence[i] not in self.vocabulary:
                sentence[i] = "<UNK>"
            tokenIds.append(self.vocabulary[sentence[i]])
        
        with torch.no_grad():
            tokenIds = torch.tensor(tokenIds, dtype=torch.long).reshape(1, -1)
            outputs = self.classifier(tokenIds)[0]

        y_pred = torch.argmax(outputs, dim=1)

        # return y_pred
        return [ self.indexClassDict[y.item()] for y in y_pred ]


class LstmPosTagger(RnnPosTagger):
    def __init__(self, trainData : TagDataset, 
                 devData : TagDataset, 
                 activation : str = 'relu', 
                 embeddingSize : int = 128, 
                 hiddenSize : int = 128, 
                 numLayers : int = 1, 
                 bidirectional : bool = False, 
                 linearHiddenLayers : list[int] = [], 
                 batchSize : int = 32) -> None:
        super().__init__(trainData, devData, activation, embeddingSize, hiddenSize, numLayers, bidirectional, linearHiddenLayers, batchSize)

        self.MODEL_SAVE_DIR = LSTM_MODEL_SAVE_DIR
        self.classifier = LstmClassifier(self.vocabSize,
                                         self.embeddingSize,
                                         self.numClasses,
                                         self.hiddenSize,
                                         self.numLayers,
                                         self.bidirectional,
                                         self.linearHiddenLayers)