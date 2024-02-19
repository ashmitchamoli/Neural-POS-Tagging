from pos_tagging.models.PosTagger import AnnPosTagger
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
                         contextSize=2, 
                         activation='relu', 
                         embeddingSize=128,
                         hiddenLayers=[32],
                         batchSize=128)

    status = model.loadFromCheckpoint()
    # print(status)

sentence = input()
tokenizedSentence = [ word.lower() for word in list(nltk.word_tokenize(sentence)) ]
preds = model.predict(tokenizedSentence)

for token in tokenizedSentence:
    print(token, preds.pop(0))