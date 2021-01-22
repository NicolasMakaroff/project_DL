import torchvision.transforms as transforms

data_transforms = transforms.Compose([
    transforms.Resize((299, 299))
    #transforms.RandomHorizontalFlip(),  # horizontaly flip the images with probability 0.5
    #transforms.RandomVerticalFlip(),
    #transforms.RandomRotation(45),
    #transforms.RandomCrop(224),
])