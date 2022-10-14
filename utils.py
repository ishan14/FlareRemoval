import torchvision
from torchvision import transforms

def get_transforms(height , width):

    transform = transforms.Compose(
        [   
            transforms.ToTensor(),
            transforms.Resize((height , width)),
            # transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))
        ]
    )
    return transform
