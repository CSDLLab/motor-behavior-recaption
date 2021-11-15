import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import __init_paths
from models.backbone import st_gcn
from models.stgcn_generator import STDiscriminator, STGenerator, STGCNGenerator


class TestSTGCNGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 4
        self.time_steps = 32
        self.channels = 5
        self.num_vertices = 34
        return super().setUp()
    
    def test_stgcn(self):
        # (spatial steps, temporal steps)
        kernel_size = (3, 5)
        st_net = nn.ModuleList([
            st_gcn(5, 16, kernel_size),
            st_gcn(16, 5, kernel_size)
        ])
        sample_inputs = torch.rand(self.batch_size, 
                                   self.channels, 
                                   self.time_steps, 
                                   self.num_vertices)
        adj = torch.rand(3, self.num_vertices, self.num_vertices)
        new_adj = adj
        x = sample_inputs
        for module in st_net:
            x, new_adj = module(x, new_adj)

        print('The produced convolution result shape {}, diff {}'.format(x.shape,
                                                                         torch.norm(x - sample_inputs)))
        print('The produced adjacent matrix differnce: ', torch.norm(new_adj - adj))
    
    def test_discrimniator(self):
        st_kernel_size = (3, 5)
        st_discriminator = STDiscriminator(np.random.rand(3, 34, 34).astype(np.float32),
                                           self.channels, 1, self.num_vertices, 64, st_kernel=st_kernel_size)
        sample_control = torch.rand(self.batch_size,
                                    1,
                                    self.time_steps,
                                    self.num_vertices)
        sample_real = torch.rand(self.batch_size,
                                 self.channels,
                                 self.time_steps,
                                 self.num_vertices)
        sample_fake = torch.rand(self.batch_size,
                                 self.channels,
                                 self.time_steps,
                                 self.num_vertices)
        seq_fake = st_discriminator(sample_fake, sample_control)
        seq_real = st_discriminator(sample_real, sample_control)
        print('The real fake distance is: {:.5f}'.format(torch.norm(seq_fake - seq_real)))
    
    def test_generator(self):
        st_kernel_size = (3, 5)
        st_generator = STGenerator(np.random.rand(3, 34, 34).astype(np.float32), 
                                   x_channels=5, z_channels=1, u_channels=1)
        sample_x0 = torch.rand(self.batch_size,
                               5,
                               self.num_vertices)
 
        sample_control = torch.rand(self.batch_size,
                                    1,
                                    self.time_steps,
                                    self.num_vertices)
        sample_noise = torch.rand(self.batch_size,
                                  1,
                                  self.time_steps,
                                  self.num_vertices)
        sample_fake = st_generator(sample_x0, sample_noise, sample_control)
        print('The generated sample shape: {}'.format(sample_fake.shape))
    
    def test_stgcn_generator_optim(self):
        st_generator = STGenerator(np.random.rand(3, 34, 34).astype(np.float32), 
                                   x_channels=5, z_channels=1, u_channels=1)
        st_discriminator = STDiscriminator(np.random.rand(3, 34, 34).astype(np.float32),
                                           x_channels=5, u_channels=1, num_nodes=self.num_vertices, hidden_size=256)
        optimizer_g = optim.RMSprop(st_generator.get_parameters(), lr=0.0001)
        optimizer_d = optim.RMSprop(st_discriminator.get_parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        for i in range(10):
            sample_x0 = torch.rand(self.batch_size,
                                   5,
                                   self.num_vertices)
            sample_control = torch.rand(self.batch_size,
                                        1,
                                        self.time_steps,
                                        self.num_vertices)
            sample_noise = torch.rand(self.batch_size,
                                      1,
                                      self.time_steps,
                                      self.num_vertices)
            sample_x = torch.rand(self.batch_size, 5, self.time_steps, self.num_vertices)
            sample_fake = st_generator(sample_x0, sample_noise, sample_control)
            seq_real = st_discriminator(sample_x, sample_control)
            seq_fake = st_discriminator(sample_fake, sample_control)
            loss = criterion(seq_fake, seq_real)
            loss.backward()
            optimizer_g.step()
            optimizer_d.step()
        print("Network optimize test successful")
    
    def test_minmax(self):
        sample_real = torch.rand(self.batch_size,
                                 self.channels,
                                 self.time_steps,
                                 self.num_vertices)
        sample_fake = torch.rand(self.batch_size,
                                 self.channels,
                                 self.time_steps,
                                 self.num_vertices)
        r_max, _ = sample_real.max(dim=2, keepdim=True)
        r_min, _ = sample_real.min(dim=2, keepdim=True)
        norm_real = (sample_real - r_min) / (r_max - r_min)
        print(norm_real.shape)


if __name__ == '__main__':
    unittest.main()
