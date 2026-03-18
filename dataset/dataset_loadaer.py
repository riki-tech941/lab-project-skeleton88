import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def get_dataloaders(batch_size=64):
    # Inserisci qui le tue 'trasformazioni'
    trasformazioni = transforms.Compose([
    transforms.RandomResizedCrop((64,64)), #Ridimensiona l immaigne a un
    #quadrato 224*224 pixel
    transforms.ToTensor(),
    transforms.Normalize( mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    #Normalizza immagina
    ]) 
    
    train_dataset = torchvision.datasets.ImageFolder('/content/dataset/tiny-imagenet-200/train', transform=trasformazioni)
    test_dataset = torchvision.datasets.ImageFolder('/content/dataset/tiny-imagenet-200/val', transform=trasformazioni)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

