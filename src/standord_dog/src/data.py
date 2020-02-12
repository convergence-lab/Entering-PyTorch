import numpy as np
import scipy.io

from PIL import Image
from torch.utils.data import Dataset


class StanfordDocDataset(Dataset):
    def __init__(self, image_path, list_path, set="train"):
        self.image_path = image_path
        if set == "train":
            lst = scipy.io.loadmat(f"{list_path}/train_list.mat")
        elif set == "test":
            lst = scipy.io.loadmat(f"{list_path}/test_list.mat")

        self.records = []
        for file_name, label in zip(lst["file_list"], lst["labels"]):
            self.records += [
                {
                    "img": file_name[0][0],
                    "label": label[0]
                }
            ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        img = Image.open(f"{self.image_path}/{item['img']}")
        img = img.resize((255, 255))
        img = np.asarray(img)
        img = np.transpose(img, (2, 0, 1)).astype("float32")
        label = int(item["label"]) - 1
        return img, label


if __name__ == "__main__":
    train_set = StanfordDocDataset("data/Images", "data/Lists", "train")
