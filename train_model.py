from pos_tagging.tag_datasets.TagData import TagDataset
from pos_tagging.models.PosTagger import AnnPosTagger, LstmPosTagger
import nltk

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

retrain = input("Do you want to load from checkpoint? (y/n): ")
print("Note: if model checkpoint is not found, training will commence from scratch.")
if retrain == 'y':
    retrain = False
else:
    retrain = True

if modelType == 'ann':
    futureContextSize = int(input("Enter future context size: "))
    pastContextSize = int(input("Enter past context size: "))
    model = AnnPosTagger(trainData, 
                             devData, 
                             futureContextSize=futureContextSize,
                             pastContextSize=pastContextSize,
                             activation=activation, 
                             embeddingSize=embeddingSize,
                             hiddenLayers=hiddenLayers,
                             batchSize=batchSize)
    model.train(epochs=15, learningRate=1e-3, verbose=True, retrain=retrain)
    model.evaluateModel(devData)

elif modelType == 'rnn':
    hiddenSize = int(input("Enter hidden size: "))
    numStacks = int(input("Enter number of stacks: "))
    bidirectional = input("Enter bidirectional (y/n): ")
    if bidirectional == 'y':
        bidirectional = True
    else:
        bidirectional = False
    model = LstmPosTagger(trainData,
                               devData,
                               activation=activation,
                               embeddingSize=embeddingSize,
                               batchSize=batchSize,
                               hiddenSize=hiddenSize,
                               numLayers=numStacks,
                               bidirectional=bidirectional,
                               linearHiddenLayers=hiddenLayers)
    model.train(epochs=15, learningRate=1e-3, verbose=True, retrain=retrain)
    model.evaluateModel(devData)

print("Model training has been completed. Do you wish to infer? (y/n)")
infer = input()
if infer == 'y':
    while True:
        sentence = input("Enter sentence: ")
        tokenizedSentence = [ word.lower() for word in list(nltk.word_tokenize(sentence)) ]
        preds = model.predict(tokenizedSentence)

        for token in [ word.lower() for word in list(nltk.word_tokenize(sentence)) ]:
            print(token, preds.pop(0))
        
        q = input("Continue? (y/n): ")
        if q == 'n':
            break