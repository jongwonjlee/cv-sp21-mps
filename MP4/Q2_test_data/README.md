## MP4-Q2 by Jongwon Lee (jongwon5)

## 1. **Implement training cycle:**

### Implementation

```
# essential hyperparameters for training
learning_rate = 1e-3    # learning rate
loss_name = 'L1'        # choose one in ['L1', 'cos']
decoder_name = 'none'   # choose one in ['none', 'basic', 'unet']
num_epochs = 200        # number of epochs to go through
val_period = 5          # period of validation over epochs
num_early_stop = 10     # number of patiences over validation to quit training
resume_path = None      # path to load a model to be resumed

# directories and file names to save or load model
model_dir = "part2_model/{0}_{1}_{2:.0e}/".format(decoder_name, loss_name, learning_rate)
best_model_fname = "model_best.pth"
best_model_path = os.path.join(model_dir, best_model_fname)

# create model instance and load weights if it is in the case
model = MyModel(decoder_type=decoder_name).to(device)

if resume_path is not None and os.path.exists(resume_path):
    model.load_state_dict(torch.load(resume_path))
    print("Resuming best model from %s" % (resume_path))

# create a directory to save checkpoints
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print("Create %s" % (model_dir))

# create data loader, loss function, optimizer, and scheduler
train_dataset = NormalDataset(split='train')
train_dataloader = data.DataLoader(train_dataset, batch_size=8, 
                                    shuffle=True, num_workers=2, 
                                    drop_last=True)
criterion = MyCriterion(loss_type=loss_name).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.01)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

'''
start training
'''

train_loss_over_epochs = []
val_error_over_epochs = []

for epoch in range(num_epochs):
    # training
    training_loss = simple_train(model, criterion, optimizer, train_dataloader, epoch)
    train_loss_over_epochs.append(training_loss)
    
    # validation
    if epoch % val_period == (val_period-1):
        val_loss, val_error = simple_validation(model, epoch, val_metric='mean_error', visualize=False)
        val_error_over_epochs.append(val_error)

        # step over scheduler
        scheduler.step(np.mean(val_loss))

        # check whether the performance has been improved. if not, quit training
        if val_error == min(val_error_over_epochs):
            torch.save(model.state_dict(), best_model_path)
            print("[%d] Best model saved at %s\n" % (epoch + 1, best_model_path))
            num_no_improvement = 0
        else:
            num_no_improvement += 1
            print("[%d] No improvement (%d / %d)\n" % (epoch + 1, num_no_improvement, num_early_stop))
            if num_no_improvement >= num_early_stop:
                break
```

## 2. **Build on top of ImageNet pre-trained Model [15 pts]:**

### Implementation

```

```

### Result 

(model structure)

```
model = MyModel(decoder_type='none').to(device)
print(model)
```

```
```

```
from torchsummary import summary

model = MyModel(decoder_type='none').to(device)
summary(model, (3, 512, 512))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 256, 256]           9,408
            Conv2d-2         [-1, 64, 256, 256]           9,408
       BatchNorm2d-3         [-1, 64, 256, 256]             128
       BatchNorm2d-4         [-1, 64, 256, 256]             128
              ReLU-5         [-1, 64, 256, 256]               0
              ReLU-6         [-1, 64, 256, 256]               0
         MaxPool2d-7         [-1, 64, 128, 128]               0
         MaxPool2d-8         [-1, 64, 128, 128]               0
            Conv2d-9         [-1, 64, 128, 128]          36,864
           Conv2d-10         [-1, 64, 128, 128]          36,864
      BatchNorm2d-11         [-1, 64, 128, 128]             128
      BatchNorm2d-12         [-1, 64, 128, 128]             128
             ReLU-13         [-1, 64, 128, 128]               0
             ReLU-14         [-1, 64, 128, 128]               0
           Conv2d-15         [-1, 64, 128, 128]          36,864
           Conv2d-16         [-1, 64, 128, 128]          36,864
      BatchNorm2d-17         [-1, 64, 128, 128]             128
      BatchNorm2d-18         [-1, 64, 128, 128]             128
             ReLU-19         [-1, 64, 128, 128]               0
             ReLU-20         [-1, 64, 128, 128]               0
       BasicBlock-21         [-1, 64, 128, 128]               0
       BasicBlock-22         [-1, 64, 128, 128]               0
           Conv2d-23         [-1, 64, 128, 128]          36,864
           Conv2d-24         [-1, 64, 128, 128]          36,864
      BatchNorm2d-25         [-1, 64, 128, 128]             128
      BatchNorm2d-26         [-1, 64, 128, 128]             128
             ReLU-27         [-1, 64, 128, 128]               0
             ReLU-28         [-1, 64, 128, 128]               0
           Conv2d-29         [-1, 64, 128, 128]          36,864
           Conv2d-30         [-1, 64, 128, 128]          36,864
      BatchNorm2d-31         [-1, 64, 128, 128]             128
      BatchNorm2d-32         [-1, 64, 128, 128]             128
             ReLU-33         [-1, 64, 128, 128]               0
             ReLU-34         [-1, 64, 128, 128]               0
       BasicBlock-35         [-1, 64, 128, 128]               0
       BasicBlock-36         [-1, 64, 128, 128]               0
           Conv2d-37          [-1, 128, 64, 64]          73,728
           Conv2d-38          [-1, 128, 64, 64]          73,728
      BatchNorm2d-39          [-1, 128, 64, 64]             256
      BatchNorm2d-40          [-1, 128, 64, 64]             256
             ReLU-41          [-1, 128, 64, 64]               0
             ReLU-42          [-1, 128, 64, 64]               0
           Conv2d-43          [-1, 128, 64, 64]         147,456
           Conv2d-44          [-1, 128, 64, 64]         147,456
      BatchNorm2d-45          [-1, 128, 64, 64]             256
      BatchNorm2d-46          [-1, 128, 64, 64]             256
           Conv2d-47          [-1, 128, 64, 64]           8,192
           Conv2d-48          [-1, 128, 64, 64]           8,192
      BatchNorm2d-49          [-1, 128, 64, 64]             256
      BatchNorm2d-50          [-1, 128, 64, 64]             256
             ReLU-51          [-1, 128, 64, 64]               0
             ReLU-52          [-1, 128, 64, 64]               0
       BasicBlock-53          [-1, 128, 64, 64]               0
       BasicBlock-54          [-1, 128, 64, 64]               0
           Conv2d-55          [-1, 128, 64, 64]         147,456
           Conv2d-56          [-1, 128, 64, 64]         147,456
      BatchNorm2d-57          [-1, 128, 64, 64]             256
      BatchNorm2d-58          [-1, 128, 64, 64]             256
             ReLU-59          [-1, 128, 64, 64]               0
             ReLU-60          [-1, 128, 64, 64]               0
           Conv2d-61          [-1, 128, 64, 64]         147,456
           Conv2d-62          [-1, 128, 64, 64]         147,456
      BatchNorm2d-63          [-1, 128, 64, 64]             256
      BatchNorm2d-64          [-1, 128, 64, 64]             256
             ReLU-65          [-1, 128, 64, 64]               0
             ReLU-66          [-1, 128, 64, 64]               0
       BasicBlock-67          [-1, 128, 64, 64]               0
       BasicBlock-68          [-1, 128, 64, 64]               0
           Conv2d-69          [-1, 256, 32, 32]         294,912
           Conv2d-70          [-1, 256, 32, 32]         294,912
      BatchNorm2d-71          [-1, 256, 32, 32]             512
      BatchNorm2d-72          [-1, 256, 32, 32]             512
             ReLU-73          [-1, 256, 32, 32]               0
             ReLU-74          [-1, 256, 32, 32]               0
           Conv2d-75          [-1, 256, 32, 32]         589,824
           Conv2d-76          [-1, 256, 32, 32]         589,824
      BatchNorm2d-77          [-1, 256, 32, 32]             512
      BatchNorm2d-78          [-1, 256, 32, 32]             512
           Conv2d-79          [-1, 256, 32, 32]          32,768
           Conv2d-80          [-1, 256, 32, 32]          32,768
      BatchNorm2d-81          [-1, 256, 32, 32]             512
      BatchNorm2d-82          [-1, 256, 32, 32]             512
             ReLU-83          [-1, 256, 32, 32]               0
             ReLU-84          [-1, 256, 32, 32]               0
       BasicBlock-85          [-1, 256, 32, 32]               0
       BasicBlock-86          [-1, 256, 32, 32]               0
           Conv2d-87          [-1, 256, 32, 32]         589,824
           Conv2d-88          [-1, 256, 32, 32]         589,824
      BatchNorm2d-89          [-1, 256, 32, 32]             512
      BatchNorm2d-90          [-1, 256, 32, 32]             512
             ReLU-91          [-1, 256, 32, 32]               0
             ReLU-92          [-1, 256, 32, 32]               0
           Conv2d-93          [-1, 256, 32, 32]         589,824
           Conv2d-94          [-1, 256, 32, 32]         589,824
      BatchNorm2d-95          [-1, 256, 32, 32]             512
      BatchNorm2d-96          [-1, 256, 32, 32]             512
             ReLU-97          [-1, 256, 32, 32]               0
             ReLU-98          [-1, 256, 32, 32]               0
       BasicBlock-99          [-1, 256, 32, 32]               0
      BasicBlock-100          [-1, 256, 32, 32]               0
          Conv2d-101          [-1, 512, 16, 16]       1,179,648
          Conv2d-102          [-1, 512, 16, 16]       1,179,648
     BatchNorm2d-103          [-1, 512, 16, 16]           1,024
     BatchNorm2d-104          [-1, 512, 16, 16]           1,024
            ReLU-105          [-1, 512, 16, 16]               0
            ReLU-106          [-1, 512, 16, 16]               0
          Conv2d-107          [-1, 512, 16, 16]       2,359,296
          Conv2d-108          [-1, 512, 16, 16]       2,359,296
     BatchNorm2d-109          [-1, 512, 16, 16]           1,024
     BatchNorm2d-110          [-1, 512, 16, 16]           1,024
          Conv2d-111          [-1, 512, 16, 16]         131,072
          Conv2d-112          [-1, 512, 16, 16]         131,072
     BatchNorm2d-113          [-1, 512, 16, 16]           1,024
     BatchNorm2d-114          [-1, 512, 16, 16]           1,024
            ReLU-115          [-1, 512, 16, 16]               0
            ReLU-116          [-1, 512, 16, 16]               0
      BasicBlock-117          [-1, 512, 16, 16]               0
      BasicBlock-118          [-1, 512, 16, 16]               0
          Conv2d-119          [-1, 512, 16, 16]       2,359,296
          Conv2d-120          [-1, 512, 16, 16]       2,359,296
     BatchNorm2d-121          [-1, 512, 16, 16]           1,024
     BatchNorm2d-122          [-1, 512, 16, 16]           1,024
            ReLU-123          [-1, 512, 16, 16]               0
            ReLU-124          [-1, 512, 16, 16]               0
          Conv2d-125          [-1, 512, 16, 16]       2,359,296
          Conv2d-126          [-1, 512, 16, 16]       2,359,296
     BatchNorm2d-127          [-1, 512, 16, 16]           1,024
     BatchNorm2d-128          [-1, 512, 16, 16]           1,024
            ReLU-129          [-1, 512, 16, 16]               0
            ReLU-130          [-1, 512, 16, 16]               0
      BasicBlock-131          [-1, 512, 16, 16]               0
      BasicBlock-132          [-1, 512, 16, 16]               0
          Conv2d-133            [-1, 3, 16, 16]           1,539
        Upsample-134          [-1, 3, 512, 512]               0
================================================================
Total params: 22,354,563
Trainable params: 22,354,563
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 3.00
Forward/backward pass size (MB): 662.01
Params size (MB): 85.28
Estimated Total Size (MB): 750.28
----------------------------------------------------------------
```

## 3. **Increase your model output resolution [15 pts]:**

### Implementation

### Result 

(model structure)


```
model = MyModel(decoder_type='basic').to(device)
print(model)
```

```
MyModel(
  (encoder_base): ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (layer1): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer2): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer3): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    (fc): Linear(in_features=512, out_features=1000, bias=True)
  )
  (conv0): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (conv1): Sequential(
    (0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (1): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (conv2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (conv3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (conv4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (upsample): Upsample(scale_factor=2.0, mode=bilinear)
  (conv_last): Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))
  (deconv4): Sequential(
    (0): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (deconv3): Sequential(
    (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (deconv2): Sequential(
    (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (deconv1): Sequential(
    (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
)
```

```
from torchsummary import summary

model = MyModel(decoder_type='basic').to(device)
summary(model, (3, 512, 512))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 256, 256]           9,408
            Conv2d-2         [-1, 64, 256, 256]           9,408
       BatchNorm2d-3         [-1, 64, 256, 256]             128
       BatchNorm2d-4         [-1, 64, 256, 256]             128
              ReLU-5         [-1, 64, 256, 256]               0
              ReLU-6         [-1, 64, 256, 256]               0
         MaxPool2d-7         [-1, 64, 128, 128]               0
         MaxPool2d-8         [-1, 64, 128, 128]               0
            Conv2d-9         [-1, 64, 128, 128]          36,864
           Conv2d-10         [-1, 64, 128, 128]          36,864
      BatchNorm2d-11         [-1, 64, 128, 128]             128
      BatchNorm2d-12         [-1, 64, 128, 128]             128
             ReLU-13         [-1, 64, 128, 128]               0
             ReLU-14         [-1, 64, 128, 128]               0
           Conv2d-15         [-1, 64, 128, 128]          36,864
           Conv2d-16         [-1, 64, 128, 128]          36,864
      BatchNorm2d-17         [-1, 64, 128, 128]             128
      BatchNorm2d-18         [-1, 64, 128, 128]             128
             ReLU-19         [-1, 64, 128, 128]               0
             ReLU-20         [-1, 64, 128, 128]               0
       BasicBlock-21         [-1, 64, 128, 128]               0
       BasicBlock-22         [-1, 64, 128, 128]               0
           Conv2d-23         [-1, 64, 128, 128]          36,864
           Conv2d-24         [-1, 64, 128, 128]          36,864
      BatchNorm2d-25         [-1, 64, 128, 128]             128
      BatchNorm2d-26         [-1, 64, 128, 128]             128
             ReLU-27         [-1, 64, 128, 128]               0
             ReLU-28         [-1, 64, 128, 128]               0
           Conv2d-29         [-1, 64, 128, 128]          36,864
           Conv2d-30         [-1, 64, 128, 128]          36,864
      BatchNorm2d-31         [-1, 64, 128, 128]             128
      BatchNorm2d-32         [-1, 64, 128, 128]             128
             ReLU-33         [-1, 64, 128, 128]               0
             ReLU-34         [-1, 64, 128, 128]               0
       BasicBlock-35         [-1, 64, 128, 128]               0
       BasicBlock-36         [-1, 64, 128, 128]               0
           Conv2d-37          [-1, 128, 64, 64]          73,728
           Conv2d-38          [-1, 128, 64, 64]          73,728
      BatchNorm2d-39          [-1, 128, 64, 64]             256
      BatchNorm2d-40          [-1, 128, 64, 64]             256
             ReLU-41          [-1, 128, 64, 64]               0
             ReLU-42          [-1, 128, 64, 64]               0
           Conv2d-43          [-1, 128, 64, 64]         147,456
           Conv2d-44          [-1, 128, 64, 64]         147,456
      BatchNorm2d-45          [-1, 128, 64, 64]             256
      BatchNorm2d-46          [-1, 128, 64, 64]             256
           Conv2d-47          [-1, 128, 64, 64]           8,192
           Conv2d-48          [-1, 128, 64, 64]           8,192
      BatchNorm2d-49          [-1, 128, 64, 64]             256
      BatchNorm2d-50          [-1, 128, 64, 64]             256
             ReLU-51          [-1, 128, 64, 64]               0
             ReLU-52          [-1, 128, 64, 64]               0
       BasicBlock-53          [-1, 128, 64, 64]               0
       BasicBlock-54          [-1, 128, 64, 64]               0
           Conv2d-55          [-1, 128, 64, 64]         147,456
           Conv2d-56          [-1, 128, 64, 64]         147,456
      BatchNorm2d-57          [-1, 128, 64, 64]             256
      BatchNorm2d-58          [-1, 128, 64, 64]             256
             ReLU-59          [-1, 128, 64, 64]               0
             ReLU-60          [-1, 128, 64, 64]               0
           Conv2d-61          [-1, 128, 64, 64]         147,456
           Conv2d-62          [-1, 128, 64, 64]         147,456
      BatchNorm2d-63          [-1, 128, 64, 64]             256
      BatchNorm2d-64          [-1, 128, 64, 64]             256
             ReLU-65          [-1, 128, 64, 64]               0
             ReLU-66          [-1, 128, 64, 64]               0
       BasicBlock-67          [-1, 128, 64, 64]               0
       BasicBlock-68          [-1, 128, 64, 64]               0
           Conv2d-69          [-1, 256, 32, 32]         294,912
           Conv2d-70          [-1, 256, 32, 32]         294,912
      BatchNorm2d-71          [-1, 256, 32, 32]             512
      BatchNorm2d-72          [-1, 256, 32, 32]             512
             ReLU-73          [-1, 256, 32, 32]               0
             ReLU-74          [-1, 256, 32, 32]               0
           Conv2d-75          [-1, 256, 32, 32]         589,824
           Conv2d-76          [-1, 256, 32, 32]         589,824
      BatchNorm2d-77          [-1, 256, 32, 32]             512
      BatchNorm2d-78          [-1, 256, 32, 32]             512
           Conv2d-79          [-1, 256, 32, 32]          32,768
           Conv2d-80          [-1, 256, 32, 32]          32,768
      BatchNorm2d-81          [-1, 256, 32, 32]             512
      BatchNorm2d-82          [-1, 256, 32, 32]             512
             ReLU-83          [-1, 256, 32, 32]               0
             ReLU-84          [-1, 256, 32, 32]               0
       BasicBlock-85          [-1, 256, 32, 32]               0
       BasicBlock-86          [-1, 256, 32, 32]               0
           Conv2d-87          [-1, 256, 32, 32]         589,824
           Conv2d-88          [-1, 256, 32, 32]         589,824
      BatchNorm2d-89          [-1, 256, 32, 32]             512
      BatchNorm2d-90          [-1, 256, 32, 32]             512
             ReLU-91          [-1, 256, 32, 32]               0
             ReLU-92          [-1, 256, 32, 32]               0
           Conv2d-93          [-1, 256, 32, 32]         589,824
           Conv2d-94          [-1, 256, 32, 32]         589,824
      BatchNorm2d-95          [-1, 256, 32, 32]             512
      BatchNorm2d-96          [-1, 256, 32, 32]             512
             ReLU-97          [-1, 256, 32, 32]               0
             ReLU-98          [-1, 256, 32, 32]               0
       BasicBlock-99          [-1, 256, 32, 32]               0
      BasicBlock-100          [-1, 256, 32, 32]               0
          Conv2d-101          [-1, 512, 16, 16]       1,179,648
          Conv2d-102          [-1, 512, 16, 16]       1,179,648
     BatchNorm2d-103          [-1, 512, 16, 16]           1,024
     BatchNorm2d-104          [-1, 512, 16, 16]           1,024
            ReLU-105          [-1, 512, 16, 16]               0
            ReLU-106          [-1, 512, 16, 16]               0
          Conv2d-107          [-1, 512, 16, 16]       2,359,296
          Conv2d-108          [-1, 512, 16, 16]       2,359,296
     BatchNorm2d-109          [-1, 512, 16, 16]           1,024
     BatchNorm2d-110          [-1, 512, 16, 16]           1,024
          Conv2d-111          [-1, 512, 16, 16]         131,072
          Conv2d-112          [-1, 512, 16, 16]         131,072
     BatchNorm2d-113          [-1, 512, 16, 16]           1,024
     BatchNorm2d-114          [-1, 512, 16, 16]           1,024
            ReLU-115          [-1, 512, 16, 16]               0
            ReLU-116          [-1, 512, 16, 16]               0
      BasicBlock-117          [-1, 512, 16, 16]               0
      BasicBlock-118          [-1, 512, 16, 16]               0
          Conv2d-119          [-1, 512, 16, 16]       2,359,296
          Conv2d-120          [-1, 512, 16, 16]       2,359,296
     BatchNorm2d-121          [-1, 512, 16, 16]           1,024
     BatchNorm2d-122          [-1, 512, 16, 16]           1,024
            ReLU-123          [-1, 512, 16, 16]               0
            ReLU-124          [-1, 512, 16, 16]               0
          Conv2d-125          [-1, 512, 16, 16]       2,359,296
          Conv2d-126          [-1, 512, 16, 16]       2,359,296
     BatchNorm2d-127          [-1, 512, 16, 16]           1,024
     BatchNorm2d-128          [-1, 512, 16, 16]           1,024
            ReLU-129          [-1, 512, 16, 16]               0
            ReLU-130          [-1, 512, 16, 16]               0
      BasicBlock-131          [-1, 512, 16, 16]               0
      BasicBlock-132          [-1, 512, 16, 16]               0
          Conv2d-133          [-1, 512, 16, 16]         262,144
     BatchNorm2d-134          [-1, 512, 16, 16]           1,024
            ReLU-135          [-1, 512, 16, 16]               0
        Upsample-136          [-1, 512, 32, 32]               0
          Conv2d-137          [-1, 256, 32, 32]       1,179,648
     BatchNorm2d-138          [-1, 256, 32, 32]             512
            ReLU-139          [-1, 256, 32, 32]               0
        Upsample-140          [-1, 256, 64, 64]               0
          Conv2d-141          [-1, 128, 64, 64]         294,912
     BatchNorm2d-142          [-1, 128, 64, 64]             256
            ReLU-143          [-1, 128, 64, 64]               0
        Upsample-144        [-1, 128, 128, 128]               0
          Conv2d-145         [-1, 64, 128, 128]          73,728
     BatchNorm2d-146         [-1, 64, 128, 128]             128
            ReLU-147         [-1, 64, 128, 128]               0
        Upsample-148         [-1, 64, 256, 256]               0
        Upsample-149         [-1, 64, 512, 512]               0
          Conv2d-150          [-1, 3, 512, 512]             195
================================================================
Total params: 24,165,571
Trainable params: 24,165,571
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 3.00
Forward/backward pass size (MB): 895.00
Params size (MB): 92.18
Estimated Total Size (MB): 990.18
----------------------------------------------------------------
```


```
model = MyModel(decoder_type='unet').to(device)
print(model)
```

```
MyModel(
  (encoder_base): ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (layer1): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer2): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer3): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    (fc): Linear(in_features=512, out_features=1000, bias=True)
  )
  (conv0): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (conv1): Sequential(
    (0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (1): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (conv2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (conv3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (conv4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (upsample): Upsample(scale_factor=2.0, mode=bilinear)
  (conv_last): Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))
  (conv4_1x1): Sequential(
    (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (deconv3): Sequential(
    (0): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (deconv2): Sequential(
    (0): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (deconv1): Sequential(
    (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
)
```

```
from torchsummary import summary

model = MyModel(decoder_type='unet').to(device)
summary(model, (3, 512, 512))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 256, 256]           9,408
            Conv2d-2         [-1, 64, 256, 256]           9,408
       BatchNorm2d-3         [-1, 64, 256, 256]             128
       BatchNorm2d-4         [-1, 64, 256, 256]             128
              ReLU-5         [-1, 64, 256, 256]               0
              ReLU-6         [-1, 64, 256, 256]               0
         MaxPool2d-7         [-1, 64, 128, 128]               0
         MaxPool2d-8         [-1, 64, 128, 128]               0
            Conv2d-9         [-1, 64, 128, 128]          36,864
           Conv2d-10         [-1, 64, 128, 128]          36,864
      BatchNorm2d-11         [-1, 64, 128, 128]             128
      BatchNorm2d-12         [-1, 64, 128, 128]             128
             ReLU-13         [-1, 64, 128, 128]               0
             ReLU-14         [-1, 64, 128, 128]               0
           Conv2d-15         [-1, 64, 128, 128]          36,864
           Conv2d-16         [-1, 64, 128, 128]          36,864
      BatchNorm2d-17         [-1, 64, 128, 128]             128
      BatchNorm2d-18         [-1, 64, 128, 128]             128
             ReLU-19         [-1, 64, 128, 128]               0
             ReLU-20         [-1, 64, 128, 128]               0
       BasicBlock-21         [-1, 64, 128, 128]               0
       BasicBlock-22         [-1, 64, 128, 128]               0
           Conv2d-23         [-1, 64, 128, 128]          36,864
           Conv2d-24         [-1, 64, 128, 128]          36,864
      BatchNorm2d-25         [-1, 64, 128, 128]             128
      BatchNorm2d-26         [-1, 64, 128, 128]             128
             ReLU-27         [-1, 64, 128, 128]               0
             ReLU-28         [-1, 64, 128, 128]               0
           Conv2d-29         [-1, 64, 128, 128]          36,864
           Conv2d-30         [-1, 64, 128, 128]          36,864
      BatchNorm2d-31         [-1, 64, 128, 128]             128
      BatchNorm2d-32         [-1, 64, 128, 128]             128
             ReLU-33         [-1, 64, 128, 128]               0
             ReLU-34         [-1, 64, 128, 128]               0
       BasicBlock-35         [-1, 64, 128, 128]               0
       BasicBlock-36         [-1, 64, 128, 128]               0
           Conv2d-37          [-1, 128, 64, 64]          73,728
           Conv2d-38          [-1, 128, 64, 64]          73,728
      BatchNorm2d-39          [-1, 128, 64, 64]             256
      BatchNorm2d-40          [-1, 128, 64, 64]             256
             ReLU-41          [-1, 128, 64, 64]               0
             ReLU-42          [-1, 128, 64, 64]               0
           Conv2d-43          [-1, 128, 64, 64]         147,456
           Conv2d-44          [-1, 128, 64, 64]         147,456
      BatchNorm2d-45          [-1, 128, 64, 64]             256
      BatchNorm2d-46          [-1, 128, 64, 64]             256
           Conv2d-47          [-1, 128, 64, 64]           8,192
           Conv2d-48          [-1, 128, 64, 64]           8,192
      BatchNorm2d-49          [-1, 128, 64, 64]             256
      BatchNorm2d-50          [-1, 128, 64, 64]             256
             ReLU-51          [-1, 128, 64, 64]               0
             ReLU-52          [-1, 128, 64, 64]               0
       BasicBlock-53          [-1, 128, 64, 64]               0
       BasicBlock-54          [-1, 128, 64, 64]               0
           Conv2d-55          [-1, 128, 64, 64]         147,456
           Conv2d-56          [-1, 128, 64, 64]         147,456
      BatchNorm2d-57          [-1, 128, 64, 64]             256
      BatchNorm2d-58          [-1, 128, 64, 64]             256
             ReLU-59          [-1, 128, 64, 64]               0
             ReLU-60          [-1, 128, 64, 64]               0
           Conv2d-61          [-1, 128, 64, 64]         147,456
           Conv2d-62          [-1, 128, 64, 64]         147,456
      BatchNorm2d-63          [-1, 128, 64, 64]             256
      BatchNorm2d-64          [-1, 128, 64, 64]             256
             ReLU-65          [-1, 128, 64, 64]               0
             ReLU-66          [-1, 128, 64, 64]               0
       BasicBlock-67          [-1, 128, 64, 64]               0
       BasicBlock-68          [-1, 128, 64, 64]               0
           Conv2d-69          [-1, 256, 32, 32]         294,912
           Conv2d-70          [-1, 256, 32, 32]         294,912
      BatchNorm2d-71          [-1, 256, 32, 32]             512
      BatchNorm2d-72          [-1, 256, 32, 32]             512
             ReLU-73          [-1, 256, 32, 32]               0
             ReLU-74          [-1, 256, 32, 32]               0
           Conv2d-75          [-1, 256, 32, 32]         589,824
           Conv2d-76          [-1, 256, 32, 32]         589,824
      BatchNorm2d-77          [-1, 256, 32, 32]             512
      BatchNorm2d-78          [-1, 256, 32, 32]             512
           Conv2d-79          [-1, 256, 32, 32]          32,768
           Conv2d-80          [-1, 256, 32, 32]          32,768
      BatchNorm2d-81          [-1, 256, 32, 32]             512
      BatchNorm2d-82          [-1, 256, 32, 32]             512
             ReLU-83          [-1, 256, 32, 32]               0
             ReLU-84          [-1, 256, 32, 32]               0
       BasicBlock-85          [-1, 256, 32, 32]               0
       BasicBlock-86          [-1, 256, 32, 32]               0
           Conv2d-87          [-1, 256, 32, 32]         589,824
           Conv2d-88          [-1, 256, 32, 32]         589,824
      BatchNorm2d-89          [-1, 256, 32, 32]             512
      BatchNorm2d-90          [-1, 256, 32, 32]             512
             ReLU-91          [-1, 256, 32, 32]               0
             ReLU-92          [-1, 256, 32, 32]               0
           Conv2d-93          [-1, 256, 32, 32]         589,824
           Conv2d-94          [-1, 256, 32, 32]         589,824
      BatchNorm2d-95          [-1, 256, 32, 32]             512
      BatchNorm2d-96          [-1, 256, 32, 32]             512
             ReLU-97          [-1, 256, 32, 32]               0
             ReLU-98          [-1, 256, 32, 32]               0
       BasicBlock-99          [-1, 256, 32, 32]               0
      BasicBlock-100          [-1, 256, 32, 32]               0
          Conv2d-101          [-1, 512, 16, 16]       1,179,648
          Conv2d-102          [-1, 512, 16, 16]       1,179,648
     BatchNorm2d-103          [-1, 512, 16, 16]           1,024
     BatchNorm2d-104          [-1, 512, 16, 16]           1,024
            ReLU-105          [-1, 512, 16, 16]               0
            ReLU-106          [-1, 512, 16, 16]               0
          Conv2d-107          [-1, 512, 16, 16]       2,359,296
          Conv2d-108          [-1, 512, 16, 16]       2,359,296
     BatchNorm2d-109          [-1, 512, 16, 16]           1,024
     BatchNorm2d-110          [-1, 512, 16, 16]           1,024
          Conv2d-111          [-1, 512, 16, 16]         131,072
          Conv2d-112          [-1, 512, 16, 16]         131,072
     BatchNorm2d-113          [-1, 512, 16, 16]           1,024
     BatchNorm2d-114          [-1, 512, 16, 16]           1,024
            ReLU-115          [-1, 512, 16, 16]               0
            ReLU-116          [-1, 512, 16, 16]               0
      BasicBlock-117          [-1, 512, 16, 16]               0
      BasicBlock-118          [-1, 512, 16, 16]               0
          Conv2d-119          [-1, 512, 16, 16]       2,359,296
          Conv2d-120          [-1, 512, 16, 16]       2,359,296
     BatchNorm2d-121          [-1, 512, 16, 16]           1,024
     BatchNorm2d-122          [-1, 512, 16, 16]           1,024
            ReLU-123          [-1, 512, 16, 16]               0
            ReLU-124          [-1, 512, 16, 16]               0
          Conv2d-125          [-1, 512, 16, 16]       2,359,296
          Conv2d-126          [-1, 512, 16, 16]       2,359,296
     BatchNorm2d-127          [-1, 512, 16, 16]           1,024
     BatchNorm2d-128          [-1, 512, 16, 16]           1,024
            ReLU-129          [-1, 512, 16, 16]               0
            ReLU-130          [-1, 512, 16, 16]               0
      BasicBlock-131          [-1, 512, 16, 16]               0
      BasicBlock-132          [-1, 512, 16, 16]               0
          Conv2d-133          [-1, 256, 16, 16]         131,072
     BatchNorm2d-134          [-1, 256, 16, 16]             512
            ReLU-135          [-1, 256, 16, 16]               0
        Upsample-136          [-1, 256, 32, 32]               0
          Conv2d-137          [-1, 128, 32, 32]         589,824
     BatchNorm2d-138          [-1, 128, 32, 32]             256
            ReLU-139          [-1, 128, 32, 32]               0
        Upsample-140          [-1, 128, 64, 64]               0
          Conv2d-141           [-1, 64, 64, 64]         147,456
     BatchNorm2d-142           [-1, 64, 64, 64]             128
            ReLU-143           [-1, 64, 64, 64]               0
        Upsample-144         [-1, 64, 128, 128]               0
          Conv2d-145         [-1, 64, 128, 128]          73,728
     BatchNorm2d-146         [-1, 64, 128, 128]             128
            ReLU-147         [-1, 64, 128, 128]               0
        Upsample-148         [-1, 64, 256, 256]               0
        Upsample-149         [-1, 64, 512, 512]               0
          Conv2d-150          [-1, 3, 512, 512]             195
================================================================
Total params: 23,296,323
Trainable params: 23,296,323
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 3.00
Forward/backward pass size (MB): 870.50
Params size (MB): 88.87
Estimated Total Size (MB): 962.37
----------------------------------------------------------------
```

(ablation table)

| model  | mean error [deg]   | median error [deg]  | acc @ 11.25 [%] | acc @ 22.5 [%] | acc @ 30 [%] | 
| ------ | ------------------ | ------------------- | --------------- | -------------- | ------------ | 
| none   |  |  |  |  |  |
| basic  |  |  |  |  |  |
| unet   |  |  |  |  |  |


## 4. **Visualize your prediction [5 pts]:**

## 5. **Secret test set [5 pts].** 