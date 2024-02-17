import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from conllu import parse_incr

class TagData:
    def __init__(self, inputDataPath : str) -> None:
        self.__inputDataPath = inputDataPath
        self.vocabulary = set("<PAD>")
        self.dataset = self.__getTagData()
    
    def __readData(self):
        dataFile = open(self.__inputDataPath, 'r', encoding='utf-8')
        return parse_incr(dataFile)
    
    def __getTagData(self) -> list[list[tuple[str, str]]]:
        """
        Returns a list of sentences. Each sentence is a list of word-tag information.
        """
        data = self.__readData()
        tagData = []
        for tokenList in data:
            tagData.append([])
            for token in tokenList:
                tagData[-1].append((token['form'], token['upos']))
                self.vocabulary.add(token['form'])

        return tagData