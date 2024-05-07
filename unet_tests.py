import torch
import numpy as np

class TestDown:
    def __init__(self, down_module):
        self.down_module = down_module
    def test_shape_checker(self):
        down2 = self.down_module(2)
        msg = "Your `check_valid` function is not right yet."
        assert down2.check_valid((8,8)), msg
        assert not down2.check_valid((9,9)), msg
        down3 = self.down_module(3)
        assert down3.check_valid((9,9)), msg
        assert not down3.check_valid((8,8)), msg
        
    
    def test_shape(self):
        tensor2 = torch.arange(16).reshape((1,4,4))
        down2 = self.down_module(2)
        expected = torch.Tensor([5,7,13,15]).reshape((1,2,2))
        msg = "The output shape of your Downsample module is not correct."
        assert expected.shape == down2(tensor2).shape, msg
        msg = "The ouput shape of your Downsample module is correct, but the values are not."
        assert torch.equal(expected, down2(tensor2)), msg
    def run(self):
        self.test_shape_checker()
        self.test_shape()
        print("TESTS PASSED")


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
        msg = "Output shape for valid padding is incorrect."
        assert tensor_out.shape == torch.Size(shape_expected), msg
    
    def test_shape_same(self):
        shape = [16, 39]
        channels = 4
        out_channels = 5
        kernel_size = 7
        
        tensor_in = torch.ones([channels] + shape)
        conv = self.conv_module(channels, out_channels, kernel_size, padding="same")
        tensor_out = conv(tensor_in)
        
        shape_expected  = [out_channels, ] + shape
        msg = "Output shape for same padding is incorrect."
        assert tensor_out.shape == torch.Size(shape_expected), msg
    
    def test_relu(self):
        shape = [1, 100, 100]
        tensor_in = torch.randn(shape) * 2
        conv = self.conv_module(1, 50, 5, padding="same")
        tensor_out = conv(tensor_in)
        msg = "Your activation function is incorrect."
        assert torch.all(tensor_out >= 0), msg
        
    def run(self):
        self.test_shape_valid()
        self.test_shape_same()
        for i in range(5):
            self.test_relu()
        print("TESTS PASSED")
        
class TestCropAndConcat:
    def __init__(self, ccmodule):
        self.ccmodule = ccmodule
        
    def test_crop(self):
        big_tensor = torch.ones((12, 14, 40, 50))
        small_tensor = torch.zeros((12, 5, 13, 18))
        ccmod = self.ccmodule()
        out_tensor = ccmod(big_tensor, small_tensor)
        expected_tensor = torch.cat([torch.ones(12,14,13,18), torch.zeros(12,5, 13, 18)], dim=1)
        msg = "Your CropAndConcat node does not give the expected output"
        assert torch.equal(out_tensor, expected_tensor), msg
    
    def run(self):
        self.test_crop()
        print("TESTS PASSED")

class TestOutputConv:
    def __init__(self, outconvmodule):
        self.outconvmodule = outconvmodule
    def test_channels(self):
        outconv = self.outconvmodule(3, 30, activation="Softshrink")
        tensor = torch.ones((3,24, 17))
        tensor_out = outconv(tensor)
        msg = "The output shape of your output conv is not right."
        assert tensor_out.shape == torch.Size((30,24,17)), msg
    def run(self):
        self.test_channels()
        print("TESTS PASSED")
        
class TestUNet:
    def __init__(self, unetmodule):
        self.unetmodule = unetmodule
    def test_fmaps(self):
        unet = self.unetmodule(5, 1, 1, 
                               num_fmaps=17,
                               fmap_inc_factor=4)
        msg = "The computation of number of feature maps in the encoder is incorrect"
        assert unet.compute_fmaps_encoder(3) == (272, 1088), msg
        msg = "The computation of number of feature maps in the decoder is incorrect"
        assert unet.compute_fmaps_decoder(3) ==  (5440, 1088), msg
        msg = "The computation of number of feature maps in the encoder is incorrect for level 0"
        assert unet.compute_fmaps_encoder(0) == (1, 17), msg
        msg = "The computation of number of feature maps in the decoder is incorrect for level 0"
        assert unet.compute_fmaps_decoder(0) == (85, 17), msg
    def test_shape_valid(self):
        unetvalid = self.unetmodule(
            depth=4,
            in_channels=2,
            out_channels=7,
            num_fmaps=5,
            fmap_inc_factor=5,
            downsample_factor=3,
            kernel_size=5,
            padding="valid"
            )
        msg = "The output shape of your UNet is incorrect for valid padding."
        assert unetvalid(torch.ones((2,2,536,536))).shape == torch.Size((2,7,112,112)), msg
    def test_shape_same(self):
        unetsame = self.unetmodule(
            depth=4,
            in_channels=2,
            out_channels=7,
            num_fmaps=5,
            fmap_inc_factor=5,
            downsample_factor=3,
            kernel_size=5,
            padding="same"
            ) 
        msg = "The output shape of your Unet is incorrect for same padding."
        assert unetsame(torch.ones((2,2,243,243))).shape == torch.Size((2,7,243,243)), msg
        
    def run(self):
        self.test_fmaps()
        self.test_shape_valid()
        self.test_shape_same()
        print("TESTS PASSED")
        