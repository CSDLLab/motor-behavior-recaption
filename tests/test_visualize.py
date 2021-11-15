import unittest
from torch.utils.data import DataLoader

import __init_paths
from dataset.muscle_sequence import MuscleSequence
from utils.visualize import VisualizeCurve


class TestCreateCurve(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = MuscleSequence('../data/ventral', reverse=True)
        self.loader = DataLoader(self.dataset, batch_size=2, 
                                 shuffle=True, num_workers=1)
        return super().setUp()
    
    def test_create_curve(self):
        for step, (x0, x, u) in enumerate(self.loader):
            vis = VisualizeCurve(x[0].numpy(), x[1].numpy())
            vis.show_curve()
