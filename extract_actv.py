import torch
import zipfile
import torchaudio
from glob import glob
from os.path import join, isfile
from os import listdir

DATADIR = '/Users/gt/Documents/GitHub/aud-dnn/data/stimuli/165_natural_sounds_16kHz/'
RESULTDIR = '/Users/gt/Documents/GitHub/aud-dnn/aud_dnn/model-actv/silero/'

files = [f for f in listdir(DATADIR) if isfile(join(DATADIR, f))]
wav_files = [f for f in files if f.endswith('wav')]


device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details

# download a single file, any format compatible with TorchAudio
torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav',
                               dst ='speech_orig.wav', progress=True)
test_files = glob('speech_orig.wav')
test_file = join(DATADIR, wav_files[0])
batches = split_into_batches(test_files, batch_size=10)
input = prepare_model_input(read_batch(batches[0]),
                            device=device)

output = model(input)
for example in output:
    print(decoder(example.cpu()))









# # silero imports
# import torch
# import random
# from glob import glob
# from omegaconf import OmegaConf
# from utils import (init_jit_model,
#                    split_into_batches,
#                    read_audio,
#                    read_batch,
#                    prepare_model_input)
# # from colab_utils import (record_audio,
# #                          audio_bytes_to_np,
# #                          upload_audio)
#
# device = torch.device('cpu')   # you can use any pytorch device
# models = OmegaConf.load('models.yml')
#
# # imports for uploading/recording
# import numpy as np
# import ipywidgets as widgets
# from scipy.io import wavfile
# from IPython.display import Audio, display, clear_output
# from torchaudio.functional import vad
#
#
# # wav to text method
# def wav_to_text(f='test.wav'):
#   batch = read_batch([f])
#   input = prepare_model_input(batch, device=device)
#   output = model(input)
#   return decoder(output[0].cpu())