# Colaboratory Set-up
import os
assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'

!pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.6-cp36-cp36m-linux_x86_64.whl

import torch

import torch_xla
import torch_xla.core.xla_model as xm

!pip install fairseq
!pip3 install torchaudio
import torchaudio

from google.colab import drive
drive.mount('/content/drive')


#Loading data - it has been processed before


from load_data import load_txt_data, load_wav_data
labels = load_txt_data(path = '/content/drive/My Drive/txt')
inputs = load_wav_data(path = '/content/drive/My Drive/wav')


#Loading models
from Wav2Vec import load_wav2vec
Wav2Vec = load_wav2vec(path = '/content/drive/My Drive/wav2vec.pt', map_location = torch.device('gpu'))


from dict import create_dictionary
d = create_dictionary()


from encode_text import encode_labels

labels, max_length = encode_text(labels, d)


from tensor_conversion import text_into_tensor
targets, target_lengths = text_into_tensor(labels, max_length)

input_lengths = torch.ones(len(labels), dtype = torch.int16) * 500



import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, num_classes = 29):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=512, out_channels=250, kernel_size=48, stride=2, padding=23)
        self.conv2 = nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3)
        self.conv6 =nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3)
        self.conv7 =nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3)
        self.conv8 =nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3)
        self.conv9 =nn.Conv1d(in_channels=250, out_channels=2000, kernel_size=32, stride=1, padding=16)
        self.conv10 =nn.Conv1d(in_channels=2000, out_channels=2000, kernel_size=1, stride=1, padding=0)
        self.conv11 =nn.Conv1d(in_channels=2000, out_channels=num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.log_softmax(x, dim = 1)
        x = torch.transpose(x,0,1)
        x = torch.transpose(x,0,2)
        return x



model = Net()
for param in model.parameters():
    param.requires_grad = True



import torch.optim as optim
criterion = torch.nn.CTCLoss(blank = 28)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
ds = torch.utils.data.TensorDataset(inputs, targets, input_lengths, target_lengths)
dataloader = torch.utils.data.DataLoader(ds, batch_size= 16, shuffle = True)




for epoch in range(5):  # loop over the dataset multiple times
    # get the inputs; data is a list of [inputs, labels]
  for i, data in enumerate(dataloader, 0):
    inputs, labels, input_lengths, target_lengths = data

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    #outputs5.requires_grad_(True)
    loss = criterion(outputs, labels, input_lengths, target_lengths5)
    loss.backward(retain_graph=True)
    optimizer.step()
    print("epoch" + str(epoch) + "i" + str(i))
    print(loss)
