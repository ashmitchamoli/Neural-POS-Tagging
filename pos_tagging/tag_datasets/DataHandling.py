import torch
from torch.utils.data import Dataset
from tag_datasets.TagData import TagDataset

class AnnPosDataset(Dataset):
    def __init__(self, data : TagDataset, classes : dict[str, int], contextSize : int) -> None:
        self.classes = classes
        self.contextSize = contextSize
        self.X, self.y = self.__prepareData(data)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.X.shape[0]

    def __getContext(self, sentence : list[tuple[int, str, str]], i : int, vocabulary : dict[str, int]) -> list[int]:
        pastContextIds = [0] * max(0, self.contextSize - i) + \
                            [ vocabulary[word[1]] for word in sentence[max(0, i - self.contextSize) : i] ]

        currWordId = vocabulary[sentence[i][1]]

        futureContextIds = [ vocabulary[word[1]] for word in sentence[i+1 : min(len(sentence) - 1, i + self.contextSize + 1)] ]
        futureContextIds = futureContextIds + [0] * max(0, self.contextSize - len(futureContextIds))

        return pastContextIds, currWordId, futureContextIds
    
    def __prepareData(self, data : TagDataset):
        X = []
        y = []

        for sentence in data.dataset:
            for i in range(len(sentence)):
                if sentence[i][2] not in self.classes:
                    continue
                
                pastContextIds, currWordId, futureContextIds = self.__getContext(sentence, i, data.vocabulary)

                X.append(pastContextIds + [currWordId] + futureContextIds)
                y.append(torch.zeros(len(self.classes)))
                y[-1][data.classes[sentence[i][2]]] = 1

        X = torch.tensor(X, dtype=torch.long)
        y = torch.stack(y)

        return X, y
