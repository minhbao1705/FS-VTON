from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--warp_checkpoint', type=str, default='C:\\Users\\vomin\\Desktop\\ckp\\PFAFN_warp_epoch_101.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--gen_checkpoint', type=str, default='C:\\Users\\vomin\\Desktop\\ckp\\PFAFN_gen_epoch_101.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--link_cloth', type=str, default=None, help='path garment')
        self.parser.add_argument('--link_edge', type=str, default=None, help='path edge')
        self.parser.add_argument('--link_image', type=str, default=None, help='path image')
        self.isTrain = False
