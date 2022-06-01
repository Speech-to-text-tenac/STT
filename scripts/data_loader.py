import os
import librosa


class DataLoader:

    def load_audios(self, audio_path:str, rate:int, mono:bool, no_of_audios:int) -> tuple:
        """Loads the audio files from a folder.

        Args:
            audio_path (str): Path to the audio files
            rate (int): The sampling rate to be used for each audio
            mono (bool): 
            no_of_audios (int): Number of files to load

        Returns:
            tuple: Dictionary of sampled audios and Maximum duration of audios
        """

        audio_data = {}
        max_duration = 0
        for i,file in enumerate(os.listdir(audio_path)):
            if i > no_of_audios:
                return audio_data, max_duration
            sampled_audio, sample_rate = librosa.load(audio_path+file, sr=rate, mono=mono)
            max_duration = max(len(sampled_audio)/sample_rate,max_duration)
            audio_data[file.split('.')[0]] = sampled_audio
        
        return audio_data, max_duration