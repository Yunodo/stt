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
