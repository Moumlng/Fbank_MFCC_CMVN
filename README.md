# Fbank_MFCC_CMVN
语音信号处理第一次作业

## 文件清单

│  main.py 

│  PHONE_001.pkf

│  PHONE_001.wav

│  PHONE_001.wav_Fbank.npy

│  PHONE_001.wav_MFCC.npy

│  PHONE_001.wav_MFCC_diff3_CMVN.npy

│  README.md

│  utils.py

│  requirements.txt

└─img

│      bank_filters.png

│      Fbank_preprocessed.png

│      Fbank_unpreprocessed.png

│      MFCC.png

│      MFCC_diff3.png

- utils.py文件内定义并编写了若干用于特征提取的函数。

- main.py文件用于调用函数并保存语音特征。

## 使用方式

- python == 3.8 (or other versions)
- pip install -r requirements.txt

- python main.py

  在根目录下生成语音对应的Fbank、MFCC和三阶差分后归一化的MFCC特征，均以numpy矩阵形式储存在.npy拓展名的文件中。

- (optional) python utils.py

  可获得若干张示意图，已保存在img文件夹内。

  

## 主要代码说明

主要函数均存放在utils.py文件中：

### get_filter_bank():

```python
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
```

分预加重、加窗STFT、过mel滤波器组、平方取能量、取对数等步骤，返回值为特征__filter_bank__和用于绘图的__filter_id, t__。

### MFCC_from_Fbank():

```python
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
```

只进行一步DCT变换，返回值为特征__MFCC__和用于绘图的__c_id__。

三阶差分和离线归一化操作容易实现，未单设函数。

## 特征文件说明

### ***_Fbank.npy

保存了filter bank特征，其中使用的滤波器组幅频响应见下图。

![滤波器组的幅频响应](/img/bank_filters.png)

获得的filter bank特征见下图。

![Fbank](/img/Fbank_preprocessed.png)

### ***_MFCC.npy

保存了由Fbank特征经DCT变换后得到的倒谱特征，见下图。

![MFCC](/img/MFCC.png)

### ***_MFCC_diff3_CMVN.npy

保存了MFCC特征经三阶差分后全局z-score归一化的结果，见下图。

![MFCC_diff3](/img/MFCC_diff3.png)



