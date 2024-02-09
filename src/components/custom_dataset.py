import torchaudio
import torch
import os
import pandas as pd
from torch.utils.data import Dataset


class IndianLanguageDataset(Dataset):
  def __init__(self, metadata, audio_dir, target_sample_rate, num_samples, transformation):
    self.annotations = pd.read_csv(metadata)
    self.audio_dir = audio_dir
    self.target_sample_rate = target_sample_rate
    self.num_samples = num_samples
    self.transformation = transformation
    
  def __len__(self):
    return len(self.annotations)

  def __getitem__(self,idx):
    audio_sample_path = self._get_audio_sample_path(idx)
    label = self._get_audio_sample_label(idx)
    signal, sr = torchaudio.load(audio_sample_path)
    signal = self.resample_audio(signal, sr)
    signal = self.mix_down_channels(signal)
    signal = self.cut_if_needed(signal)
    signal = self.right_padding(signal)
    signal = self.transformation(signal)
    return signal , label
  
  def _get_audio_sample_path(self,idx):
    """
    > It takes the index of the audio file and returns the path of the audio file
    """
    class_name = f"{self.annotations.iloc[idx, 1]}"
    path = os.path.join(self.audio_dir, class_name, self.annotations.iloc[idx, 0])
    return path
  
  def _get_audio_sample_label(self, idx):
    """
    > This function returns the label of the audio sample at the given index
    """
    return self.annotations.iloc[idx, 2]
  
  def resample_audio(self, signal, sr):
    """
    > Resample the audio signal to the target sample rate
    """
    if sr != self.target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
        signal = resampler(signal)
    return signal
  
  def mix_down_channels(self, signal):
    """
    > If the signal has more than one channel, average the channels together
    """
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim = 0, keepdim=True)
    return signal
  
  def cut_if_needed(self, signal):
    """
    > If the signal is longer than the number of samples, cut it down to the number of samples
    """
    if signal.shape[1] > self.num_samples:
        signal = signal[:, :self.num_samples]
    return signal 
  
  def right_padding(self, signal):
    """
    > If the signal is shorter than the number of samples, pad the signal with zeros
    """
    length_signal = signal.shape[1]
    if length_signal < self.num_samples:
        num_missing = self.num_samples - length_signal
        last_dim_padding = (0, num_missing)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal