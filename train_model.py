from pos_tagging.tag_datasets.TagData import TagDataset
from pos_tagging.models.PosTagger import AnnPosTagger, LstmPosTagger

trainData = TagDataset('./data/UD_English-Atis/en_atis-ud-train.conllu')
devData = TagDataset('./data/UD_English-Atis/en_atis-ud-dev.conllu')
testData = TagDataset('./data/UD_English-Atis/en_atis-ud-test.conllu')

modelType = input("Enter model type (ann/rnn): ")
activation = input(f"Enter activation function (relu/sigmoid/tanh): ")
embeddingSize = int(input("Enter embedding size: "))
hiddenLayers = []
nHiddenLayers = int(input("Enter number of hidden layers: "))
for i in range(nHiddenLayers):
    hiddenLayers.append(int(input(f"Enter size of hidden layer {i+1}: ")))
batchSize = int(input("Enter batch size: "))

if modelType == 'ann':
    futureContextSize = int(input("Enter future context size: "))
    pastContextSize = int(input("Enter past context size: "))
    annTagger = AnnPosTagger(trainData, 
                             devData, 
                             futureContextSize=futureContextSize,
                             pastContextSize=pastContextSize,
                             activation=activation, 
                             embeddingSize=embeddingSize,
                             hiddenLayers=hiddenLayers,
                             batchSize=batchSize)
    annTagger.train(epochs=30, learningRate=1e-3)
    annTagger.evaluateModel(devData)

elif modelType == 'rnn':
    hiddenSize = int(input("Enter hidden size: "))
    numStacks = int(input("Enter number of stacks: "))
    bidirectional = input("Enter bidirectional (y/n): ")
    if bidirectional == 'y':
        bidirectional = True
    else:
        bidirectional = False
    lstmTagger = LstmPosTagger(trainData,
                               devData,
                               activation=activation,
                               embeddingSize=embeddingSize,
                               batchSize=batchSize,
                               hiddenSize=hiddenSize,
                               numLayers=numStacks,
                               bidirectional=bidirectional,
                               linearHiddenLayers=hiddenLayers)
    lstmTagger.train(epochs=10, learningRate=1e-3)
    lstmTagger.evaluateModel(devData)