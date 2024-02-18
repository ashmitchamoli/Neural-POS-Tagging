import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from conllu import parse_incr

class TagDataset:
    def __init__(self, inputDataPath : str) -> None:
        self.__inputDataPath = inputDataPath
        self.vocabulary = {"<PAD>" : 0}
        self.classes = {}
        self.dataset = self.__getTagData()
    
    def __readData(self):
        dataFile = open(self.__inputDataPath, 'r', encoding='utf-8')
        return parse_incr(dataFile)
    
    def __getTagData(self) -> list[list[tuple[int, str, str]]]:
        """
        Returns a list of sentences. Each sentence is a list of word-tag information.
        """
        data = self.__readData()
        tagData = []
        for tokenList in data:
            # iterate over each sentence
            tagData.append([])

            # iterate over each token
            for token in tokenList:
                tagData[-1].append((token['id'], token['form'], token['upos'])) # store information about this token

                # update vocabulary
                if token['form'] not in self.vocabulary:
                    self.vocabulary[token['form']] = len(self.vocabulary)
                
                # update classes
                if token['upos'] not in self.classes:
                    self.classes[token['upos']] = len(self.classes)

        self.vocabulary["<UNK>"] = len(self.vocabulary)

        return tagData