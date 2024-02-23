from pos_tagging.models.PosTagger import AnnPosTagger, RnnPosTagger, LstmPosTagger
from pos_tagging.tag_datasets.TagData import TagDataset

import sys
import nltk

trainData = TagDataset('./data/UD_English-Atis/en_atis-ud-train.conllu')
devData = TagDataset('./data/UD_English-Atis/en_atis-ud-dev.conllu')

# command line args
model = None
if sys.argv[1] == '-f':
    model = AnnPosTagger(trainData,
                         devData,
                         futureContextSize=1,
                         pastContextSize=1,
                         activation='sigmoid', 
                         embeddingSize=128,
                         hiddenLayers=[64],
                         batchSize=128)

if sys.argv[1] == '-r':
    model = LstmPosTagger(trainData,
                          devData,
                          activation='relu',
                          embeddingSize=128,
                          batchSize=1,
                          hiddenSize=128,
                          numLayers=2,
                          bidirectional=True,
                          linearHiddenLayers=[32])
model.train(epochs=20, learningRate=1e-3, verbose=False)

sentence = input()
tokenizedSentence = [ word.lower() for word in list(nltk.word_tokenize(sentence)) ]
preds = model.predict(tokenizedSentence)

for token in [ word.lower() for word in list(nltk.word_tokenize(sentence)) ]:
    print(token, preds.pop(0))