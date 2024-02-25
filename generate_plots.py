import numpy as np
import pickle
import importlib
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns

from pos_tagging.tag_datasets.TagData import TagDataset 
from pos_tagging.models.PosTagger import AnnPosTagger, RnnPosTagger, LstmPosTagger

trainData = TagDataset('./data/UD_English-Atis/en_atis-ud-train.conllu')
devData = TagDataset('./data/UD_English-Atis/en_atis-ud-dev.conllu')
testData = TagDataset('./data/UD_English-Atis/en_atis-ud-test.conllu')

annTagger = AnnPosTagger(trainData, 
                         devData, 
                         futureContextSize=2,
                         pastContextSize=2,
                         activation='tanh', 
                         embeddingSize=512,
                         hiddenLayers=[64, 32],
                         batchSize=128)
print("Training ANN model...")
annTagger.train(epochs=15, learningRate=1e-3, verbose=True, retrain=True)

lstmTagger = LstmPosTagger(trainData,
                           devData,
                           activation='relu',
                           embeddingSize=256,
                           batchSize=1,
                           hiddenSize=64,
                           numLayers=2,
                           bidirectional=True,
                           linearHiddenLayers=[64, 32])
print("Training RNN model...")
lstmTagger.train(epochs=10, learningRate=1e-3, verbose=True, retrain=True)
   

def plotAnnTrainingProgress():
    # ANN Training progress plot
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].plot(annTagger.trainLoss)
    ax[1].plot(annTagger.devLoss)
    ax[2].plot(annTagger.devAccuracy)
    ax[0].set_title('Training Loss')
    ax[1].set_title('Dev Loss')
    ax[2].set_title('Dev Accuracy')
    plt.show()

def plotRnnTrainingProgress():
    # RNN Training progress plot
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax[0].plot(lstmTagger.trainLoss)
    ax[1].plot(lstmTagger.devAccuracy)
    ax[0].set_title('Training Loss')
    ax[1].set_title('Dev Accuracy')
    plt.show()

def plotConfusionMatrix():
    annTagger.evaluateModel(testData)
    lstmTagger.evaluateModel(testData)

    # Confusion matrix
    plt.figure(figsize=(13, 10))
    sns.heatmap(annTagger.confusionMatrix, annot=True, square=True, fmt='.2f', cmap='Blues', vmax=0.6, xticklabels=lstmTagger.classes, yticklabels=lstmTagger.classes)
    plt.title('ANN Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    plt.figure(figsize=(13, 10))
    sns.heatmap(lstmTagger.confusionMatrix, annot=True, square=True, fmt='.2f', cmap='Blues', vmax=0.6, xticklabels=lstmTagger.classes, yticklabels=lstmTagger.classes)
    plt.title('RNN Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def trainBestModel(contextSize : int) -> None:
    posTagger = AnnPosTagger(trainData, 
                             devData, 
                             futureContextSize=contextSize,
                             pastContextSize=contextSize,
                             activation='tanh', 
                             embeddingSize=512,
                             hiddenLayers=[64, 32],
                             batchSize=128)
    posTagger.train(epochs=30, learningRate=1e-3)
    posTagger.evaluateModel(devData)
    return posTagger

def plotAccuracyVsContextSize():
    model0 = trainBestModel(0)
    model1 = trainBestModel(1)
    model2 = trainBestModel(2)
    model3 = trainBestModel(3)
    model4 = trainBestModel(4)

    plt.plot([ 0, 1, 2, 3, 4 ], [ model0.scores['accuracy'], model1.scores['accuracy'], model2.scores['accuracy'], model3.scores['accuracy'], model4.scores['f1'] ])
    plt.title('Accuracy vs Context Size')
    plt.xlabel('Context Size')
    plt.ylabel('Accuracy')
    plt.show()

def plotDevAccuracyVsEpochs():
    lstmModel1 = LstmPosTagger(trainData, devData, activation='relu', embeddingSize=256, hiddenSize=64, linearHiddenLayers=[64, 32], batchSize=1, bidirectional=True)
    lstmModel1.train(epochs=20, learningRate=1e-3, retrain=True)

    lstmModel2 = LstmPosTagger(trainData, devData, activation='relu', embeddingSize=128, hiddenSize=128, linearHiddenLayers=[64], batchSize=1, bidirectional=True)
    lstmModel2.train(epochs=20, learningRate=1e-3, retrain=True)

    lstmModel3 = LstmPosTagger(trainData, devData, activation='relu', embeddingSize=128, hiddenSize=64, linearHiddenLayers=[64, 32], batchSize=1, bidirectional=True)
    lstmModel3.train(epochs=20, learningRate=1e-3, retrain=True)

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].plot(lstmModel1.devAccuracy)
    ax[1].plot(lstmModel2.devAccuracy)
    ax[2].plot(lstmModel3.devAccuracy)
    ax[0].set_title('Top Model 1')
    ax[1].set_title('Top Model 2')
    ax[2].set_title('Top Model 3')
    plt.show()

plotAnnTrainingProgress()
plotRnnTrainingProgress()
plotConfusionMatrix()
plotAccuracyVsContextSize()
plotDevAccuracyVsEpochs()
