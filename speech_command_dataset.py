from torch.utils.data import Dataset,ConcatDataset
import pandas as pd
import torchaudio
import os, glob, re, tempfile, random

seed = 42
random.seed(seed)


class AudioDataset(Dataset):
    def __init__(self, audio_path, audio_list):
        self.keywords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        self.__audio_list = audio_list

        self.__audio_path = [f"{str(audio_path)}/{audiofile}" for audiofile in audio_list]

    #len(audio_wav_file)
    def __len__(self):
        return len(self.__audio_list)
    
    #a_list[1] -> a_list.__getitem__(1)
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label
    
    def _get_audio_sample_path(self,index):
        path = self.__audio_path[index]
        return path

    def _get_audio_sample_label(self,index):
        path = self.__audio_path[index]
        parts = path.split(os.path.sep)
        label = parts[-2]

        if label == '_silence_' or label == 'silence':
            return 'silence'
        elif label == '_unkown_':
            return 'unknown'
        elif label in self.keywords:
            return label
        else:
            return 'unknown'
        




class SpeechCommandDataset():
    def __init__(self, audio_dataset_path, audio_test_path):
        self.__dataset_path = audio_dataset_path
        self.__testset_path = audio_test_path

        self.__train_list, self.__validation_list, self.__test_list, self.__testing_audio_list_full = self.__load_audio_sets()

        self.__silence_folder = self.__create_silence_samples() # for train and validation 
        # load silence files
        silence_files=os.listdir(self.__silence_folder)
        idx_split_train_val = int(len(silence_files) * 0.8/ 0.9)
        self.__silence_train_list = silence_files[:idx_split_train_val]
        self.__silence_validation_list = silence_files[idx_split_train_val+1::]

    def get_train_list(self):
        silence_files = [f"silence/{filename}" for filename in self.__silence_train_list]
        return self.__train_list + silence_files
    
    def get_validation_list(self):
        silence_files = [f"silence/{filename}" for filename in self.__silence_validation_list]
        return self.__validation_list + silence_files
    
    def get_test_list(self):
        return self.__test_list
    
    def get_silence_temp_folder(self):
        return self.__silence_folder

    def __create_silence_samples(self):
        temp_dir = tempfile.mkdtemp()
        
        SAMPLE_RATE = 16000  # Sample rate for one second audio clip

        data_dir=self.__dataset_path
        filenames = glob.glob(str(data_dir) + '/*/*.wav')

        validation_paths = [f"{str(data_dir)}/{path}" for path in self.__validation_list]
        testing_paths = [f"{str(data_dir)}/{path}" for path in self.__testing_audio_list_full]
        training_paths = list(set(filenames)-set(validation_paths)-set(testing_paths))

        
        background_noise_folder = os.path.join(data_dir, '_background_noise_')

        wav_files = [os.path.join(background_noise_folder, file) for file in os.listdir(background_noise_folder) if file.endswith('.wav')]

        num_classes = 35
        for idx in range(round((len(training_paths)+len(validation_paths))/num_classes*1.5)): #similar number of audios than other classes
            random_wav_file = random.choice(wav_files)
            waveform, sample_rate = torchaudio.load(random_wav_file)

            audio_length = waveform.size(dim=1)
            start_point = random.randint(0, audio_length - SAMPLE_RATE-1)

            one_second_clip = waveform[:, start_point:start_point + SAMPLE_RATE]

            path = f"{temp_dir}/one_second_clip_{idx}.wav"
            torchaudio.save(path, one_second_clip, sample_rate)

        return temp_dir

    def __load_audio_sets(self):
        TEST_LIST = "testing_list.txt"
        VALIDATION_LIST = "validation_list.txt"

        all_files = glob.glob(self.__dataset_path + '*/*.wav')
        all_files = [os.path.relpath(f, self.__dataset_path) for f in all_files]

        # Step 1: Read the paths from the text files into lists
        with open(self.__dataset_path+(VALIDATION_LIST), 'r') as f:
            validation_audio_list = f.read().splitlines()

        with open(self.__dataset_path+TEST_LIST, 'r') as f:
            testing_audio_list_full = f.read().splitlines()

        with open(self.__testset_path+TEST_LIST, 'r') as f:
            testing_audio_list = f.read().splitlines()

        train_audio_list = list(set(all_files)-set(validation_audio_list)-set(testing_audio_list_full))
        train_audio_list = [item for item in train_audio_list if not re.search(r"_background_noise_", item)]


        all_test_files = glob.glob(self.__testset_path + '*/*.wav')
        all_test_files = [os.path.relpath(f, self.__testset_path) for f in all_test_files]

        return train_audio_list, validation_audio_list, testing_audio_list, testing_audio_list_full

    def __create_dataset(self, file_list, path):
        return AudioDataset(path , file_list)
    
    def get_trainset(self):
        train_files = self.__train_list
        path = self.__dataset_path
        keywordsDataset = self.__create_dataset(train_files , path)
        silenceDataset = self.__create_dataset(self.__silence_train_list , self.__silence_folder)
        return ConcatDataset([keywordsDataset, silenceDataset])

    def get_validationset(self):
        val_files = self.__validation_list
        path = self.__dataset_path
        keywordsDataset = self.__create_dataset(val_files , path)
        silenceDataset = self.__create_dataset(self.__silence_validation_list , self.__silence_folder)
        return ConcatDataset([keywordsDataset, silenceDataset])

    def get_testset(self):
        test_files = self.__test_list
        path = self.__testset_path
        return self.__create_dataset(test_files, path)