import scipy.signal as signal
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def mel(freq):
    return 2595 * np.log10(1 + freq / 700)

def imel(melfreq):
    return (10**(melfreq / 2595) - 1) * 700

def diff_3(data):
    data = np.array(data)
    def diff_1(data):
        column1 = data[:,0]
        column1 = np.expand_dims(column1, axis=1)
        data_extended = np.concatenate((column1, data), axis = 1)
        data_diff = data_extended[:,1:]-data_extended[:,:-1]
        return data_diff
    return(diff_1(diff_1(diff_1(data))))

def get_filter_bank(wav, fs, nperseg = 512, freq_range = [60, 3400], n_banks = 15, window = 'hann', miu = 0.95):
    
    mel_freq_range = [mel(freq) for freq in freq_range]
    mel_bank_o = [mel_freq_range[0] + i * (mel_freq_range[1] - mel_freq_range[0])/(n_banks + 1) for i in range(n_banks + 2)]
    bank_o = [imel(mel_freq) for mel_freq in mel_bank_o]
    
    wav_preprocessed = [0 if i == 0 else wav[i] - miu * wav[i-1] for i in range(len(wav))] #预加重
    
    f, t, zxx = signal.stft(wav_preprocessed, fs, window, nperseg, noverlap = nperseg / 2)
    bank_filters = [[0 if delta < 0 else delta for delta in [(freq - bank_o[i])/(bank_o[i+1] - bank_o[i]) if freq < bank_o[i+1] else (bank_o[i+2] - freq)/(bank_o[i+2] - bank_o[i+1]) for freq in f]] for i in range(n_banks)]
    bank_filters = np.array(bank_filters)
    plt.figure(figsize=[12,6])
    for i in range(n_banks):
        plt.plot(f, bank_filters[i,:])
        plt.xlabel('f/Hz')
    plt.title('Amplitude Frequency Response of Filter Bank')
    plt.legend([f'filter {i}' for i in range(n_banks)])
    zxx = (abs(zxx)) ** 2
    filter_bank = np.matmul(bank_filters, zxx)
    filter_bank = np.log(filter_bank)
    filter_id = [i for i in range(n_banks)]
    
    return filter_id, t, filter_bank

def MFCC_from_Fbank(Fbank, nDCT=None):
    
    if nDCT == None:
        nDCT = Fbank.shape[0]
    else:
        assert type(nDCT) == int, f'The nDCT shoule be int type, not {type(nDCT)} type.'
        
    DCT_matrix = [[np.cos(np.pi*(i + 1)/Fbank.shape[0]*(j + 0.5)) for j in range(Fbank.shape[0])]for i in range(nDCT)]
    DCT_matrix = np.sqrt(2/Fbank.shape[0]) * np.array(DCT_matrix)
    
    MFCC = np.matmul(DCT_matrix, Fbank)
    c_id = np.array([i+1 for i in range(nDCT)])
    
    return c_id, MFCC


if __name__ == '__main__':
    fs, wav = scipy.io.wavfile.read('PHONE_001.wav')
    
    fid, t, Fbank = get_filter_bank(wav, fs, nperseg = 512, n_banks=15)
    
    ff,tt = np.meshgrid(fid, t)
    plt.figure(figsize=[10,5])
    plt.pcolor(tt.T[:,500:600], ff.T[:,500:600], Fbank[:,500:600], cmap='Reds', shading='auto')
    plt.xlabel('time/s')
    plt.ylabel('filter number')
    plt.title('Filter bank Log Amptitude in the first few seconds')
    plt.colorbar()
    
    c_id, MFCC = MFCC_from_Fbank(Fbank)
    cc,tt = np.meshgrid(c_id, t)
    plt.figure(figsize=[10,5])
    norm = colors.TwoSlopeNorm(vmin=MFCC.min(), vmax=MFCC.max(), vcenter=0)
    plt.pcolor(tt.T[:,500:600], cc.T[:,500:600], MFCC[:,500:600], cmap='bwr', shading='auto', norm=norm)
    plt.xlabel('time/s')
    plt.ylabel('MFCC_id')
    plt.title('MFCC')
    plt.colorbar()
    
    MFCC_diff_3 = diff_3(MFCC)
    mean = np.mean(MFCC_diff_3)
    std = np.std(MFCC_diff_3)
    MFCC_diff_3 = (MFCC_diff_3 - mean) / std
    cc,tt = np.meshgrid(c_id, t)
    plt.figure(figsize=[10,5])
    norm = colors.TwoSlopeNorm(vmin=MFCC_diff_3.min(), vmax=MFCC_diff_3.max(), vcenter=0)
    plt.pcolor(tt.T[:,:100], cc.T[:,:100], MFCC_diff_3[:,:100], cmap='bwr', shading='auto', norm=norm)
    plt.xlabel('time/s')
    plt.ylabel('MFCC_id')
    plt.title('MFCC_diff3')
    plt.colorbar()
    
    plt.show()