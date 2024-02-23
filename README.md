# Neural-POS-Tagging
This repository provides 2 different architectures for Neural PoS Tagging implemented using PyTorch.

## ANN Architecture
The fist layer is an embedding layer followed by a simple feed forward architecture with Softmax activation in the final layer. 

### Embedding Layer
This layer is a simple matrix of size `|V|*|X|`, where `|V|` is the vocabulary size and `|X|` is the desired embedding size. This layer takes a word index as input and returns the corresponding row vector as the embedding for this word. All the entries in this matrix are trainable weights and are trained along with the entire pipeline. For this layer, the implementation provided by PyTorch, `torch.nn.Embedding`, has been used.

### Feed Forward Layer
The embedding layer is simply followed by a feed forward network, with desired hidden layer sizes and activation function. The final layer is of size `numClasses` with a softmax activation and the output is interpreted as class probabilities.

### Code Implementation

#### AnnPosDataset
This class inherits from `torch.utils.data.Dataset` and provides the dataset class for ease of data loading during training.

Each data point in the training data consists of a word from a sentence `w`, `p` words that come before `w` in the sentence and `s` words that come after `w` in the sentence. The data point is stored as a list of `p + s + 1` word indices. The corresponding label is stored as a list of `p + s + 1` one-hot vectors indicating the class of the word `w`.

Source File: [`pos_tagger/tag_datasets/DataHandling.py`](./pos_tagging/tag_datasets/DataHandling.py).

#### AnnClassifier
This class inherits from `torch.nn.Module` and provides the implementaion of the ANN architecture discussed [above](#ann-architecture).

Source file: [`pos_tagger/models/ANN.py`](./pos_tagging/models/ANN.py)

#### AnnPosTagger
This class uses the above 2 classes and takes `TagDataset` (see [TagDataset.py](./pos_tagging/models/PosTagger.py)) objects as training and validation data. It trains the model on the training data and evaluates on the validation data after each epoch. It also provides methods for saving and loading models from file and evaluating the model on test data.

Source file: [`pos_tagger/models/PosTagger.py`](./pos_tagging/models/PosTagger.py)


## RNN Architecture
The fist layer is an embedding layer followed by an LSTM (or a vanilla RNN) with desired stack size. The hidden output embedding is fed into the feed forward layer containing the desired hidden layer sizes and activation function. The final layer is of size `numClasses` with a softmax activation and the output is interpreted as class probabilities.

### Embedding Layer
The same implementation as the ANN architecture is used here.

### LSTM/RNN Layer
This layer is a LSTM (or RNN) layer with desired stack size and hidden size. The final hidden output embedding is fed into the feed forward layer.

### Code Implementation

#### RnnPosDataset
This class inherits from `torch.utils.data.Dataset` and provides the dataset class for ease of data loading during training.

Each data point is a sequence of word indices forming a sentence. The corresponding label is stored as a list of one-hot vectors indicating the class of each word in the sentence.

Source File: [`pos_tagger/tag_datasets/DataHandling.py`](./pos_tagging/tag_datasets/DataHandling.py).

#### LstmClassifier/RnnClassifier
This class inherits from `torch.nn.Module` and provides the implementaion of the RNN architecture discussed [above](#rnn-architecture).

Source file: [`pos_tagger/models/RNN.py`](./pos_tagging/models/RNN.py)

#### LstmPosTagger/RnnPosTagger
Same as the AnnPosTagger, this class combines the above 2 classes and provides methods for model training, saving, loading and evaluating the model on test data.

Source file: [`pos_tagger/models/PosTagger.py`](./pos_tagging/models/PosTagger.py)

# Running the code
## Inference
To test the model on a single sentence, use the file `pos_tagger.py` as follows:

From the root directory of the project, run: 
```
python pos_tagger.py -<model_type>
```
Here, `model_type` = f for `ANN` or r for `RNN`.

## Evaluation
