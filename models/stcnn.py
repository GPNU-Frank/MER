import torch
import torch.nn as nn


class MySTCNN(nn.Module):
    def __init__(self):
        super(MySTCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(3, 64, (3, 3, 3), (2, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, (3, 3, 3), (2, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(128, 256, (1, 3, 3), (1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3 * 3, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 5),
        )
    
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
    
    def reset_all_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            # elif isinstance(m,nn.Linear):
            #     m.weight.data.normal_()  # 全连接层参数初始化

if __name__ == '__main__':
    inputs = torch.rand((4, 1, 15, 224, 224))
    model = MySTCNN()
    inputs = inputs.cuda()
    model = model.cuda()
    outputs = model(inputs)
    print(outputs)