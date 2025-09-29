import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.model = nn.Sequential(
            # Input: (3, 128, 128)
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),   # -> (32, 128, 128)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),  # -> (32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),  # -> (64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # -> (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),  # -> (128, 16, 16)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# -> (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),  # -> (256, 8, 8)

            nn.AdaptiveAvgPool2d((2, 2)),  # -> (256, 1, 1)
            nn.Flatten(),                  # -> (256,)
            nn.Linear(1024, 256),    # final classifier
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)    # final classifier
        )

    def forward(self, x):
        return self.model(x)