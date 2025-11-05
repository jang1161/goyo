import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import csv

#from scipy.io import wavfile
#from scipy import signal

# Load the model.
model = hub.load('./yamnet_1')



# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])
  return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)



# def ensure_sample_rate(original_sample_rate, waveform,
#                        desired_sample_rate=16000):
#   """Resample waveform if required."""
#   if original_sample_rate != desired_sample_rate:
#     desired_length = int(round(float(len(waveform)) /original_sample_rate * desired_sample_rate))
#     waveform = signal.resample(waveform, desired_length)
#   return desired_sample_rate, waveform



# wav_file_name = 'speech_whistling2.wav'
# 이따가 파일이 들어갈 경로를 저기다가 해야함
wav_file_name = 'vibration.mp3'

try:
    
    wav_data, sample_rate = librosa.load(wav_file_name, sr=16000, mono=True)
    # sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

    # Show some basic information about the audio.
    duration = len(wav_data)/sample_rate
    print(f'Sample rate: {sample_rate} Hz')
    print(f'Total duration: {duration:.2f}s')
    print(f'Size of the input: {len(wav_data)}')

    # if wav_data.ndim != 1:  #dimension 1차원으로 통일?
    #   wav_data = np.mean(wav_data, axis=1)
      
    waveform = wav_data
    waveform = tf.cast(waveform, tf.float32)

    # Run the model, check the output.
    scores, embeddings, spectrogram = model(waveform)


    scores_np = scores.numpy()
    spectrogram_np = spectrogram.numpy()
    infered_class = class_names[scores_np.mean(axis=0).argmax()]
    print(f'The main sound is: {infered_class}')

except FileNotFoundError:
    print(f"에러 : {wav_file_name}을 찾을 수 없습니다.")
    print(f"파일을 다운로드하거나, {wav_file_name}의 경로를 확인하세요.")
except Exception as e:
    print(f"에러발생 : {e}")
    
    