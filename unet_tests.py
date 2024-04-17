import torch
import numpy as np

class TestDown:
    def __init__(self, down_module):
        self.down_module = down_module
    def test_shape_checker(self):
        down2 = self.down_module(2)
        assert down2.check_valid((8,8))
        assert not down2.check_valid((9,9))
        down3 = self.down_module(3)
        assert down3.check_valid((9,9))
        assert not down3.check_valid((8,8))
        
    
    def test_shape(self):
        tensor2 = torch.arange(16).reshape((1,4,4))
        down2 = self.down_module(2)
        expected = torch.Tensor([5,7,13,15]).reshape((1,2,2))
        assert torch.equal(expected, down2(tensor2))
    def run(self):
        self.test_shape_checker()
        self.test_shape()


class TestConvBlock:
    def __init__(self, conv_module):
        self.conv_module = conv_module
    def test_shape_valid(self):
        shape = [20, 30]
        channels = 4
        out_channels = 5
        kernel_size = 7
        
        tensor_in = torch.ones([channels] + shape)
        conv = self.conv_module(channels, out_channels, kernel_size, padding="valid")
        tensor_out = conv(tensor_in)
        
        shape_expected = np.array(shape) - 2 * (kernel_size - 1)
        shape_expected = [out_channels, ] + list(shape_expected)
        assert tensor_out.shape == torch.Size(shape_expected), "Shape for valid padding is incorrect"
    
    def test_shape_same(self):
        shape = [16, 39]
        channels = 4
        out_channels = 5
        kernel_size = 7
        
        tensor_in = torch.ones([channels] + shape)
        conv = self.conv_module(channels, out_channels, kernel_size, padding="same")
        tensor_out = conv(tensor_in)
        
        shape_expected  = [out_channels, ] + shape
        assert tensor_out.shape == torch.Size(shape_expected) 
        
    def run(self):
        self.test_shape_valid()
        self.test_shape_same()