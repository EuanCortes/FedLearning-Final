from torch import nn


class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()

        self.relu = nn.ReLU()               # activation function

        # convcolutional layers
        self.ConvLayers = nn.ModuleList([
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # N x 3 x 32 x 32 -> N x 32 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # N x 32 x 16 x 16 -> N x 64 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # N x 64 x 8 x 8 -> N x 128 x 8 x 8
        ])

        self.fc1 = nn.Linear(128 * 4 * 4, 256)      # first fully connected layer after convolutional layers
        self.fc2 = nn.Linear(256, 10)               # final fully connected layer for output

        self.pool = nn.MaxPool2d(2, 2)      # max pooling layer for regularization
        self.dropout = nn.Dropout(0.2)      # dropout layer for regularization

    def forward(self, x):
        for conv in self.ConvLayers:
            # apply convolutional layer, then batch normalization, then ReLU, then max pooling
            x = conv(x)
            x = self.pool(self.relu(x))       # final shape: N x 128 x 4 x 4

        x = x.view(x.size(0), -1)       # reshape to N x 128*4*4
        x = self.relu(self.fc1(x))      # fully connected layer and ReLU
        x = self.dropout(x)             # apply dropout for regularization
        x = self.fc2(x)                 # final fully connected layer for output
        return x