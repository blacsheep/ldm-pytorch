import torchvision.transforms as transforms
import webdataset as wds
from torch.utils.data import DataLoader


def preprocessing(sample):
    txt, img = sample
    preproc = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return txt, preproc(img)

def get_dataloader(file_path, batch_size):
    dirpath = file_path
    dataset = wds.WebDataset(dirpath).shuffle(1000).decode("pil").to_tuple("txt", "jpg").map(preprocessing)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader
