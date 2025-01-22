# Speech Commands Recognition v2

Many scientific papers using the [Google Speech Commands Dataset v2](https://arxiv.org/abs/1804.03209) do not follow all provided recommendations for fair results comparison.
This repository serves for creating the train, validation, and test sets following **strictly** all recommendations, using [PyTorch](http://pytorch.org).

## Features
- It automatically loads the 10 classes (words: "Yes", "No", "Up", "Down", "Left", "Right", "On", "Oﬀ", "Stop", and "Go") and generates the "unknown" and "silence" classes data.
- It generates the train, validation, and test sets in Datasets objects.

## Paper recommendations
The [paper](https://arxiv.org/abs/1804.03209) specifies a train/validation/test split of 80/10/10% of the data.

It provides a 'validation_list.txt' and 'test_list.txt' files for meaningful comparisons.

The 'test_list.txt' already provides the 'background' and 'silence' audio samples.

These files can be obtained from the [speech_commands_test_set_v0.02](http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz)

The original audio files are provided in [speech_commands_v0.02](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)

It is **highly recommended** use this audio sample to perform test comparisons.

## Requirements

* Python 3.6+
* [PyTorch](https://github.com/pytorch/pytorch#installation)

## Usage

### Google Speech Commands Dataset (v0.02)
To download and extract the [Google Speech Commands Dataset](https://arxiv.org/abs/1804.03209) run the following python code:

```
import os, pathlib

# download dataset
os.mkdir('dataset')
os.mkdir('dataset_test_set')
! wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz -P dataset
! wget http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz -P dataset_test_set

# extract dataset
! tar -xf dataset/speech_commands_v0.02.tar.gz -C dataset
! tar -xf dataset/speech_commands_test_set_v0.02.tar.gz -C dataset_test_set

# define path
DATASET_PATH = pathlib.Path('dataset')
TESTSET_PATH = pathlib.Path('dataset_test_set')
```

### Example
```
from speech_command_dataset import SpeechCommandDataset

dataset = SpeechCommandDataset(DATASET_PATH,TESTSET_PATH)

test_set = dataset.get_testset()
validation_set = dataset.get_validationset()
train_set = dataset.get_trainset()

print(f"created silence audio directory: {dataset.get_silence_temp_folder()}")
print("name of test audio files:")
print(dataset.get_test_list())
print(f"number of samples in the train set: {len(train_set)}")
```