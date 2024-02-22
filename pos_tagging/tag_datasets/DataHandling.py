import torch
from torch.utils.data import Dataset
from tag_datasets.TagData import TagDataset

class PosDataset(Dataset):
    def __init__(self, data : TagDataset, classes : dict[str, int], vocabulary : dict[str, int]) -> None:
        self.classes = classes
        self.vocabulary = vocabulary
        self.numClasses = len(classes)

    def __getitem__(self, index):
        return self.X[index], torch.stack(self.y[index], dim=0)
    
    def __len__(self):
        return len(self.X)

class AnnPosDataset(PosDataset):
    def __init__(self, data : TagDataset, 
                 classes : dict[str, int], 
                 futureContextSize : int,
                 pastContextSize : int, 
                 vocabulary : dict[str, int]) -> None:
        super().__init__(data, classes, vocabulary)
        
        self.futureContextSize = futureContextSize
        self.pastContextSize = pastContextSize

        self.X, self.y = self.__prepareData(data)

    def __getContext(self, sentence : list[tuple[int, str, str]], i : int, vocabulary : dict[str, int]) -> list[int]:
        pastContextIds = [0] * max(0, self.pastContextSize - i) + \
                            [ vocabulary[word[1]] for word in sentence[max(0, i - self.pastContextSize) : i] ]

        currWordId = vocabulary[sentence[i][1]]

        futureContextIds = [ vocabulary[word[1]] for word in sentence[i+1 : min(len(sentence) - 1, i + self.futureContextSize + 1)] ]
        futureContextIds = futureContextIds + [0] * max(0, self.futureContextSize - len(futureContextIds))

        return pastContextIds, currWordId, futureContextIds
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __prepareData(self, data : TagDataset):
        X = []
        y = []

        for sentence in data.dataset:
            for i in range(len(sentence)):
                if sentence[i][1] not in self.vocabulary:
                    sentence[i] = (sentence[i][0], "<UNK>", sentence[i][2])

            for i in range(len(sentence)):
                if sentence[i][2] not in self.classes:
                    continue

                pastContextIds, currWordId, futureContextIds = self.__getContext(sentence, i, self.vocabulary)

                X.append(pastContextIds + [currWordId] + futureContextIds)
                y.append(torch.zeros(len(self.classes)))
                y[-1][self.classes[sentence[i][2]]] = 1

        X = torch.tensor(X, dtype=torch.long)
        y = torch.stack(y)

        return X, y

class RnnPosDataset(PosDataset):
    def __init__(self, data: TagDataset, classes: dict[str, int], vocabulary: dict[str, int]) -> None:
        super().__init__(data, classes, vocabulary)

        self.X, self.y = self.__prepareData(data)

    def collate_fn(self, batch):
        X, y = zip(*batch)
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)

        # yNew = []

        # sequenceLength = X.shape[1]
        # for groundTruth in y:
        #     paddingLength = sequenceLength - len(groundTruth)
        #     groundTruth = groundTruth + [ torch.tensor([0] * self.numClasses, dtype=torch.float) ] * paddingLength
        #     yNew.append(torch.stack(groundTruth))

        return X, y

    def __prepareData(self, data : TagDataset):
        X = []
        y = []
        
        for sentence in data.dataset:
            for i in range(len(sentence)):
                if sentence[i][1] not in self.vocabulary:
                    sentence[i] = (sentence[i][0], "<UNK>", sentence[i][2])

            X.append([]) # new sentence
            y.append([])
            for i in range(len(sentence)):
                if sentence[i][2] not in self.classes:
                    continue

                X[-1].append(self.vocabulary[sentence[i][1]])
                y[-1].append(torch.zeros(len(self.classes), dtype=torch.float))
                y[-1][-1][self.classes[sentence[i][2]]] = 1
            
            X[-1] = torch.tensor(X[-1], dtype=torch.long)

        # X = torch.tensor(X, dtype=torch.long)
        # y = torch.stack(y)

        return X, y