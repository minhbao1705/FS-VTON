# import time
# from options.train_options import TrainOptions
# from models.networks import ResUnetGenerator, VGGLoss, save_checkpoint, load_checkpoint_parallel
# from models.afwm import TVLoss, AFWM, NetAFWMParallel
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# import numpy as np
import torch
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
# from tensorboardX import SummaryWriter
# import cv2
# import datetime
# import wandb

# from PIL import Image
# import os
# import torchvision.transforms as transforms
#python test.py 
model = torch.load("D:\FSVTONdata\pfafn\PFAFN_warp_epoch_174.pth",map_location=torch.device('cpu'))

for k, v in model['model_state_dict'].items():
    # if k == "param_groups":
    print(k)
# opt = TrainOptions().parse()
# net = AFWM(opt, 45)
# model = NetAFWMParallel(net,opt.local_rank)
# params_warp = [p for p in model.parameters()]
# optimizer_warp = torch.optim.Adam(params_warp, lr=opt.lr, betas=(opt.beta1, 0.999))

# for k, v in net.state_dict():
#     print(k)




# fine_height = 512
# fine_width = int(512 / 256 * 192)
# transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# densepose_name = "00000_00.jpg"
# densepose_map = Image.open(os.path.join("D:\\virtualdataset\\train", 'image-densepose', densepose_name))
# densepose_map = transforms.Resize(fine_width, interpolation=2)(densepose_map)
# densepose_map = transform(densepose_map)
# print(densepose_map.shape)


# labels = {
#             0: ['background', [0, 10]],
#             1: ['hair', [1, 2]],
#             2: ['face', [4, 13]],
#             3: ['upper', [5, 6, 7]],
#             4: ['bottom', [9, 12]],
#             5: ['left_arm', [14]],
#             6: ['right_arm', [15]],
#             7: ['left_leg', [16]],
#             8: ['right_leg', [17]],
#             9: ['left_shoe', [18]],
#             10: ['right_shoe', [19]],
#             11: ['socks', [8]],
#             12: ['noise', [3, 11]]
#         }

# parse_name = "00000_00.png"
# image_parse_agnostic = Image.open(os.path.join("D:\\virtualdataset\\train", 'image-parse-agnostic-v3.2', parse_name))

# image_parse_agnostic = transforms.Resize(fine_width, interpolation=0)(image_parse_agnostic)
# parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
# image_parse_agnostic = transform(image_parse_agnostic.convert('RGB'))


# parse_agnostic_map = torch.FloatTensor(20, fine_height, fine_width).zero_()
# parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
# new_parse_agnostic_map = torch.FloatTensor(13, fine_height,fine_width).zero_()
# for i in range(len(labels)):
#     for label in labels[i][1]:
#         new_parse_agnostic_map[i] += parse_agnostic_map[label]

# print(new_parse_agnostic_map.shape)

