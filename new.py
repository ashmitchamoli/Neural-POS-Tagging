from pos_tagging.models.PosTagger import LstmPosTagger
from pos_tagging.tag_datasets.TagData import TagDataset

trainData = TagDataset('./data/UD_English-Atis/en_atis-ud-train.conllu')
devData = TagDataset('./data/UD_English-Atis/en_atis-ud-dev.conllu')
testData = TagDataset('./data/UD_English-Atis/en_atis-ud-test.conllu')

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
lstmTagger.train(epochs=10, learningRate=1e-3, verbose=True)

import matplotlib.pyplot as plt

print(lstmTagger.evaluateModel(trainData))

