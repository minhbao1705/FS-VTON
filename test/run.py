import time
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from torchvision import utils
# from util import flow_util
from data.base_dataset import get_params, get_transform
from PIL import Image
opt = TestOptions().parse()

#import ipdb; ipdb.set_trace()
warp_model = AFWM(opt, 3)
# print(warp_model)
warp_model.eval()
warp_model.cuda()
load_checkpoint(warp_model, opt.warp_checkpoint)

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
#print(gen_model)
gen_model.eval()
gen_model.cuda()
load_checkpoint(gen_model, opt.gen_checkpoint)


if not os.path.exists('D:/FSVTONdata/our_t_results'):
  os.mkdir('D:/FSVTONdata/our_t_results')


def Input_load(options, image_path, cloth_path, c_mask_path):
    I_path = os.path.join(image_path)
    I = Image.open(I_path).convert('RGB')
    params = get_params(options, I.size)
    transform = get_transform(options, params)
    transform_E = get_transform(options, params, method=Image.NEAREST, normalize=False)

    I_tensor = transform(I)
    C_path = os.path.join(cloth_path)

    C = Image.open(C_path).convert('RGB')
    C_tensor = transform(C)

    E_path = os.path.join(c_mask_path)
    E = Image.open(E_path).convert('L')
    E_tensor = transform_E(E)

    p_name = image_path.split('\\')[-1].split(".")[0] + "_" + cloth_path.split('\\')[-1].split(".")[0] + ".jpg"
    print(p_name)

    input_dict = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor, 'p_name': p_name}
    return input_dict

if opt.link_image == None or opt.link_cloth == None or opt.link_edge == None:
  raise Exception("Not enough provided links") 
else:
  image_path = opt.link_image
  cloth_path = opt.link_cloth
  c_mask_path = opt.link_edge
# image_path = "D:\\FSVTONdata\\VITON_test\\test_img\\000020_0.jpg"
#   cloth_path = "D:\\FSVTONdata\\VITON_test\\test_clothes\\000048_1.jpg"
#   c_mask_path = "D:\\FSVTONdata\\VITON_test\\test_edge\\000048_1.jpg"
data = Input_load(opt, image_path, cloth_path, c_mask_path)

real_image = data['image']
clothes = data['clothes']
##edge is extracted from the clothes image with the built-in function in python
edge = data['edge']

real_image = real_image[None, :, :, :] #[1, 3, 256, 192]
clothes = clothes[None, :, :, :]
edge = edge[None, :, :, :]

edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
clothes = clothes * edge

#import ipdb; ipdb.set_trace()

flow_out = warp_model(real_image.cuda(), clothes.cuda())
warped_cloth, last_flow, = flow_out
warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                    mode='bilinear', padding_mode='zeros',align_corners=True)

gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
gen_outputs = gen_model(gen_inputs)
p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
p_rendered = torch.tanh(p_rendered)
m_composite = torch.sigmoid(m_composite)
m_composite = m_composite * warped_edge
p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
# print(type(p_tryon))
path = 'results/' + opt.name
os.makedirs(path, exist_ok=True)
sub_path = path + '/PFAFN'
os.makedirs(sub_path,exist_ok=True)
print(data['p_name'])

print(p_tryon[0].permute((1, 2, 0)).numpy(force= True)[2])
# print(os.path.join('D:\FSVTONdata//our_t_results', data['p_name']))
# save try-on image only

utils.save_image(
    p_tryon,
    os.path.join('D:/FSVTONdata/our_t_results', data['p_name']),
    nrow=int(1),
    normalize=True,
    value_range=(-1,1),
)
