## MP4-Q1 by Jongwon Lee (jongwon5)

## 1. **Implement BaseNet [5 pts].**

### Implementation

Below is implementation for BaseNet, a vary simple deep-NN model for image classification.

```
import torch.nn as nn
import torch.nn.functional as F

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        # convolutional kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # affine transformation y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        # layer 1 to layer 3
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # layer 4 to layer 6
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # flatten the output
        x = x.view(-1, 16 * 5 * 5)
        # layer 7 and 8
        x = F.relu(self.fc1(x))
        # layer 9
        x = self.fc2(x)

        return x
```

### Result 

The above model can be visualized by using `print(net)` command as follows:

```
# Create an instance of the nn.module class defined above:
net = BaseNet()
print(net)
```

By doing so, we are able to obtain all the layers consisting of the model like below. Please note that other operations, such as activation layers or flattening, are not revealed using this function.

```
BaseNet(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=200, bias=True)
  (fc2): Linear(in_features=200, out_features=10, bias=True)
)
```

By following a sequence of training and validation cycle, I was able to observe a set of evaluation results on the validation dataset.

```
Accuracy of the final network on the val images: 60.0 %
Accuracy of airplane : 71.8 %
Accuracy of automobile : 76.8 %
Accuracy of  bird : 45.3 %
Accuracy of   cat : 35.5 %
Accuracy of  deer : 60.5 %
Accuracy of   dog : 43.5 %
Accuracy of  frog : 74.0 %
Accuracy of horse : 64.1 %
Accuracy of  ship : 58.5 %
Accuracy of truck : 70.2 %
```

## 2. **Improve BaseNet [20 pts].** 

### Implementation

I came up with VGGNet-like structure due to its simple structure but powerful performance. The detailed implementation is as follow:

```
cfg = {
    '1': [16, 'M', 32, 'M', 64, 'M', 128, 'M'], # 81.2%
    '2': [32, 'M', 64, 'M', 128, 'M', 256, 'M', 512, 'M'], # 86.2%
    '3': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'], # 85.9%
}

class ImprovedNet(nn.Module):
    def __init__(self, model_name):
        super(ImprovedNet, self).__init__()
        # normalize input tensor
        self.normalize_input = transforms.Compose([
              transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])])
        # model's layers
        self.convnet = self._make_layers(cfg[model_name])
        self.fc = nn.Linear(in_features=512, out_features=10)

        # arbitrary variable to save the number of input channels for each layer
        self.in_channels = 3

    def _make_layers(self, cfg):
        layers = []
        for param in cfg:
            if param == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif param == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, param, kernel_size=3, padding=1),
                           nn.BatchNorm2d(param),
                           nn.ReLU(inplace=True)]
                self.in_channels = param
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.normalize_input(x)
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
```

Note that `conv2d + batchnorm + relu` and `maxpool2d` are stacked alternatively to reduce the tensor's size gradually. Also, there exists three options to create an ImprovedNet with different depth; ImprovedNet-V1, ImprovedNet-V2, and ImprovedNet-V3 from shallower to deeper.

Architecture for ImprovedNet-V1 (which can be rendered by `net = ImprovedNet('1')`) is provided below:

| Layer No. | Layer Type                | Kernel Size | Input Dim | Output Dim  | Input Channels | Output Channels |
| --------- | ------------------------- | ----------- | --------- | ----------- | -------------- | --------------- |
| 1         | conv2d + batchnorm + relu | 3           | 32        | 32          | 3              | 16              |
| 2         | maxpool2d                 | 2           | 32        | 16          | 16             | 16              |
| 3         | conv2d + batchnorm + relu | 3           | 16        | 16          | 16             | 32              |
| 4         | maxpool2d                 | 2           | 16        | 8           | 32             | 32              |
| 5         | conv2d + batchnorm + relu | 3           | 8         | 8           | 32             | 64              |
| 6         | maxpool2d                 | 2           | 8         | 4           | 64             | 64              |
| 7         | conv2d + batchnorm + relu | 3           | 4         | 4           | 64             | 128             |
| 8         | maxpool2d                 | 2           | 4         | 2           | 128            | 128             |
| 9         | flatten                   | -           | 2         | 1           | 128            | 512             |
| 9         | linear                    | -           | 1         | 1           | 512            | 10              |

Architecture for ImprovedNet-V2 (which can be rendered by `net = ImprovedNet('2')`) is provided below:

| Layer No. | Layer Type                | Kernel Size | Input Dim | Output Dim  | Input Channels | Output Channels |
| --------- | ------------------------- | ----------- | --------- | ----------- | -------------- | --------------- |
| 1         | conv2d + batchnorm + relu | 3           | 32        | 32          | 3              | 32              |
| 2         | maxpool2d                 | 2           | 32        | 16          | 32             | 32              |
| 3         | conv2d + batchnorm + relu | 3           | 16        | 16          | 32             | 64              |
| 4         | maxpool2d                 | 2           | 16        | 8           | 64             | 64              |
| 5         | conv2d + batchnorm + relu | 3           | 8         | 8           | 64             | 128             |
| 6         | maxpool2d                 | 2           | 8         | 4           | 128            | 128             |
| 7         | conv2d + batchnorm + relu | 3           | 4         | 4           | 128            | 256             |
| 8         | maxpool2d                 | 2           | 4         | 2           | 256            | 256             |
| 9         | conv2d + batchnorm + relu | 3           | 2         | 2           | 256            | 512             |
| 10        | maxpool2d                 | 2           | 2         | 1           | 512            | 512             |
| 11        | flatten                   | -           | 2         | 1           | 512            | 512             |
| 12        | linear                    | -           | 1         | 1           | 512            | 10              |

Architecture for ImprovedNet-V2 (which can be rendered by `net = ImprovedNet('3')`) is provided below:

| Layer No. | Layer Type                | Kernel Size | Input Dim | Output Dim  | Input Channels | Output Channels |
| --------- | ------------------------- | ----------- | --------- | ----------- | -------------- | --------------- |
| 1         | conv2d + batchnorm + relu | 3           | 32        | 32          | 3              | 32              |
| 2         | maxpool2d                 | 2           | 32        | 16          | 32             | 32              |
| 3         | conv2d + batchnorm + relu | 3           | 16        | 16          | 32             | 64              |
| 4         | maxpool2d                 | 2           | 16        | 8           | 64             | 64              |
| 5         | conv2d + batchnorm + relu | 3           | 8         | 8           | 64             | 128             |
| 6         | conv2d + batchnorm + relu | 3           | 8         | 8           | 128            | 128             |
| 7         | maxpool2d                 | 2           | 8         | 4           | 128            | 128             |
| 8         | conv2d + batchnorm + relu | 3           | 4         | 4           | 128            | 256             |
| 9         | conv2d + batchnorm + relu | 3           | 4         | 4           | 256            | 256             |
| 9         | maxpool2d                 | 2           | 4         | 2           | 256            | 256             |
| 10        | conv2d + batchnorm + relu | 3           | 2         | 2           | 256            | 512             |
| 11        | conv2d + batchnorm + relu | 3           | 2         | 2           | 512            | 512             |
| 12        | maxpool2d                 | 2           | 2         | 1           | 512            | 512             |
| 13        | flatten                   | -           | 2         | 1           | 512            | 512             |
| 14        | linear                    | -           | 1         | 1           | 512            | 10              |


### Result 

Below is a table for comparing the aforementioned three models with different depth serving image classification task. Data augmentation with random cropping and horizontal flipping (`transforms.Compose([transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip()])`) were utilized to maximize the training performance.

| model | accuracy [%] |
| ----- | ------------ |
| ImprovedNet-V1 | 81.2 |
| ImprovedNet-V2 | 86.2 |
| ImprovedNet-V3 | 85.9 |

It could be observed that ImprovedNet-V2, a model with intermediate depth, showed the best validation accuracy among three options. The reason why the deepest model (ImprovedNet-V3) does not demonstrate the best performance can be explained in various ways: deeper model typically takes more data and time to learn the task, too deeper model can encounter gradient vanishing, or etc.

## 3. **Secret test set [5 pts].**

