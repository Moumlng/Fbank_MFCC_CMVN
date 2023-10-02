import scipy.signal as signal
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt

def mel(freq):
    return 2595 * np.log10(1 + freq / 700)

def imel(melfreq):
    return (10**(melfreq / 2595) - 1) * 700

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
    filter_id = [i for i in range(n_banks)]
    ff,tt = np.meshgrid(filter_id, t)
    #print(bank_filter.shape, zxx.shape, filter_bank.shape)
    plt.figure(figsize=[10,5])
    plt.pcolor(tt.T[:,:100], ff.T[:,:100], np.log(filter_bank[:,:100]), cmap='Reds', shading='auto')
    plt.xlabel('time/s')
    plt.ylabel('filter number')
    plt.title('Filter bank Log Amptitude in the first few seconds')
    plt.colorbar()
    
    return filter_id, t, filter_bank


if __name__ == '__main__':
    fs, wav = scipy.io.wavfile.read('PHONE_001.wav')
    get_filter_bank(wav, fs, nperseg = 512, n_banks=15)
    plt.show()