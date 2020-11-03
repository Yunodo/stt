import main

def load_txt_data(path):

    """
    Inputs:
        path  - string containing path to file, e.g.'/content/drive/My Drive/txt'

    Outputs:
        texts - an array of strings representing text data
    """
    texts = []
    for filename in sorted(os.listdir(path)):
        f = open(filename, encoding = "ISO-8859-1")
        string = f.read()
        texts.append(string[:-3].lower()) #removing newline characters
    return texts

def load_wav_data(path):

    """
    Inputs:
        path - string containing path to file, e.g.'/content/drive/My Drive/wav'

    Outputs:
        wavs - Pytorch tensor of .wav data; shape [B, N], where B - Batch Size, N - length of audio .wav
    """
    wavs _ = torchaudio.load('/content/drive/My Drive/wav/4 Advances in Immunotherapy for Cancer Treatment_part1.wav')
    for filename in sorted(os.listdir(path)):
        waveform, _ = torchaudio.load(filename)
        wavs = torch.cat((wavs,waveform),0)
    wavs = wavs[1:,:]
    return wavs
