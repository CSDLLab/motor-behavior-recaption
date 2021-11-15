import unittest

import __init_paths
from dataset.muscle_sequence import MuscleSequence
from utils.utils import save_to_video

from torch.utils.data import DataLoader


class TestCreateVideo(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = MuscleSequence('../notebooks/dataset_zjr.npz',
                                      time_steps=120,
                                      dilation=1)
        self.loader = DataLoader(self.dataset, batch_size=2,
                                 shuffle=True, num_workers=2)
        return super().setUp()
    
    def test_save_video(self):
        for step, (x0, x, u) in enumerate(self.loader):
            x = x * self.dataset.muscle_std[None, :, None, None] + self.dataset.muscle_mean[None, :, None, None]
            save_to_video(x.numpy(), f'../outputs/step_{step}')


if __name__ == '__main__':
    unittest.main()
