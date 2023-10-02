import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
from utils import get_filter_bank, MFCC_from_Fbank, diff_3

wave_name = 'PHONE_001.wav'
fs, wav = scipy.io.wavfile.read(wave_name)
print(f'wave length = {len(wav) / fs}s.')
filter_id, t, Fbank = get_filter_bank(wav, fs, nperseg = 512)
np.save(f'{wave_name}_Fbank', Fbank)

c_id, MFCC = MFCC_from_Fbank(Fbank)
np.save(f'{wave_name}_MFCC', MFCC)

MFCC_diff_3 = diff_3(MFCC)
mean = np.mean(MFCC_diff_3)
std = np.std(MFCC_diff_3)
MFCC_diff_3 = (MFCC_diff_3 - mean) / std
np.save(f'{wave_name}_MFCC_diff_3', MFCC_diff_3)
#plt.show()