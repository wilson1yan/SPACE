import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class H5Py(Dataset):
    def __init__(self, root, mode):
        assert mode in ['train', 'test']
        self.root = root
        self.mode = mode

        self.data = h5py.File(root, 'r')
        self._images = self.data[f'{self.mode}_data']

    def __setstate__(self, state):
        self.__dict__ = state
        self.data = h5py.File(self.root, 'r')
        self._images = self.data[f'{self.mode}_data']

    def __getstate__(self):
        state = self.__dict__
        state['data'].close()
        state['data'] = None
        state['_images'] = None

        return state
    
    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = self._images[index]
        image = torch.from_numpy(image / 255).permute(2, 0, 1).float()
        image = F.interpolate(image, size=128)
        print(image.min(), image.max(), image.shape)
        return image
