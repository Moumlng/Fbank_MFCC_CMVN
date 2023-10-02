import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
from utils import get_filter_bank

fs, wav = scipy.io.wavfile.read('PHONE_001.wav')
print(f'wave length = {len(wav) / fs}s.')
filter_id, t, Fbank = get_filter_bank(wav, fs, nperseg = 512)
np.save('Fbank', Fbank)

#plt.show() #show mel bank filters and filter_bank