import glob
import numpy as np
from PIL import Image
import torch
from skimage.color import rgb2lab
from torchvision import transforms
import wandb
from constants import dataset_path


def get_data_paths(path):
    paths = glob.glob(path + "/*.jpg")  # Grabbing all the image file names
    np.random.seed(14)
    paths_subset = np.random.choice(paths, 10000, replace=False)  # choosing 10000 images randomly
    rand_idx = np.random.permutation(10000)
    train_idx = rand_idx[:32]  # choosing the first 8000 as training set
    val_idx = rand_idx[32:48]  # choosing last 2000 as validation set
    train_paths = paths_subset[train_idx]
    val_paths = paths_subset[val_idx]
    return train_paths, val_paths


class ColorizationDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.transform = transform
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        img = np.asarray(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        img_l = img_lab[0] / 50. - 1.  # Between -1 and 1
        img_ab = img_lab[1:] / 110.  # Between -1 and 1
        return [img_l.unsqueeze(0), img_ab]

    def __len__(self):
        return len(self.paths)


def get_dataloaders(path, batch_size=16, num_workers=0, pin_memory=True):
    train_transforms = transforms.Compose(
        [
         transforms.Resize((256, 256),  transforms.InterpolationMode.BICUBIC),
         transforms.RandomHorizontalFlip(),
        ])
    val_transforms = transforms.Compose(
        [
         transforms.Resize((256, 256),  transforms.InterpolationMode.BICUBIC),
        ])

    train_paths, val_paths = get_data_paths(path)
    train_dataset = ColorizationDataset(train_paths, train_transforms)
    val_dataset = ColorizationDataset(val_paths, val_transforms)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_dataloader, val_dataloader


def log_dataset(path=dataset_path, name=None, aliases=None):
    with wandb.init(project="Colorize_GAN", job_type="load-data", name=name) as run:
        raw_data = wandb.Artifact(
            "coco-10k", type="dataset",
            description="Coco dataset",
            metadata={"source": "fastai.data.external.untar_data(URLs.COCO_SAMPLE)",
                      "size": len(glob.glob(path + "/*.jpg"))})

        raw_data.add_dir(path)
        run.log_artifact(raw_data, aliases=aliases)
        raw_data.wait()
        print(raw_data.id, raw_data.name, raw_data.version)


# log_dataset("colorize_model/dataset", name="adding_new_data", aliases=["latest", "custom"])
