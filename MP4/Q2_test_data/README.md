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


## 3. **Increase your model output resolution [15 pts]:**

### Implementation

### Result 

(model structure)

```
model = MyModel(decoder_type='unet').to(device)
print(model)
```

```
```

(ablation table)

| model  | mean error [deg]   | median error [deg]  | acc @ 11.25 [%] | acc @ 22.5 [%] | acc @ 30 [%] | 
| ------ | ------------------ | ------------------- | --------------- | -------------- | ------------ | 
| none   |  |  |  |  |  |
| basic  |  |  |  |  |  |
| unet   |  |  |  |  |  |


## 4. **Visualize your prediction [5 pts]:**

## 5. **Secret test set [5 pts].** 