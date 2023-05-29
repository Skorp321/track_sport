import torchreid

# This code creates an ImageDataManager object that manages image data for training and testing.
datamanager = torchreid.data.ImageDataManager(
    root="data/processed", # The directory where the data is stored.
    sources="data/processed", # The source dataset to use.
    targets="data/processed", # The target dataset to use.
    height=256, # The height of the input image.
    width=128, # The width of the input image.
    batch_size_train=32, # Batch size for training images.
    batch_size_test=100, # Batch size for testing images.
    transforms=["random_flip", "random_crop"], # Data augmentation techniques to apply during training.
    norm_mean=[0.485, 0.456, 0.406], # Mean values used for normalization of input images
    norm_std=[0.229, 0.224, 0.225]   # Standard deviation values used for normalization of input images
)

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=datamanager.num_train_pids,
    loss="softmax",
    pretrained=True
)

model = model.cuda()

# This code builds an Adam optimizer with learning rate of 0.0003.
optimizer =  torchreid.optim.build_optimizer(
    model,
    optim="adam",
    lr=0.0003
)

# This code builds a single step learning rate scheduler with step size of 20 epochs.
scheduler =  torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler="single_step",
    stepsize=20
)

# This code creates an ImageSoftmaxEngine object that trains and tests the model using softmax loss function.
engine =  torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

engine.run(
    save_dir="../../log/resnet50-softmax-market1501",
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=False
)
   
