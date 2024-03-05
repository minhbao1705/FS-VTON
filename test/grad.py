import gradio as gr
import numpy as np
import torch
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import os
import torch.nn.functional as F
from torchvision import utils
from options.test_options import TestOptions
from PIL import Image
from data.base_dataset import get_params, get_transform
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
opt = TestOptions().parse()
import collections
try:
    from collections import abc
    collections.MutableMapping = abc.MutableMapping
except:
    pass
opt.resize_or_crop = "None"
opt.warp_checkpoint = "C:\\Users\\vomin\\Desktop\\ckp\\PFAFN_warp_epoch_101.pth"
opt.gen_checkpoint = "C:\\Users\\vomin\\Desktop\\ckp\\PFAFN_gen_epoch_101.pth"
opt.gpu_ids =-1

def Input_load(options, image, cloth, c_mask):
    print(image)
    # I = image.convert('RGB')
    # print(I.size)
    params = get_params(options, image.size)
    transform = get_transform(options, params)
    transform_E = get_transform(options, params, method=Image.NEAREST, normalize=False)

    I_tensor = transform(image)

    # C = cloth.convert('RGB')
    C_tensor = transform(cloth)

    
    E = c_mask.convert('L')
    E_tensor = transform_E(E)

    # p_name = image_path.split('\\')[-1].split(".")[0] + "_" + cloth_path.split('\\')[-1].split(".")[0] + ".jpg"
    # print(p_name)

    input_dict = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor}
    return input_dict
# def convert_to_tensor(img):
#     tensor = torch.from_numpy(img)
#     tensor = torch.permute(tensor, (2, 0, 1))
#     tensor = tensor[None, : , : , :]
#     return tensor

def sepia(img, cloth, cmask):
    warp_model = AFWM(opt, 3)
    print(warp_model)
    warp_model.eval()
    warp_model.cuda()
    load_checkpoint(warp_model, opt.warp_checkpoint)

    gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
    #print(gen_model)
    gen_model.eval()
    gen_model.cuda()
    load_checkpoint(gen_model, opt.gen_checkpoint)

    data = Input_load(opt, img, cloth, cmask)

    real_image = data['image']
    clothes = data['clothes']
    ##edge is extracted from the clothes image with the built-in function in python
    edge = data['edge']
    real_image = real_image[None, :, :, :] #[1, 3, 256, 192]
    clothes = clothes[None, :, :, :]
    edge = edge[None, :, :, :]

    edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
    clothes = clothes * edge

    # real_image = convert_to_tensor(img)
    # clothes = convert_to_tensor(cloth)
    # edge = convert_to_tensor(cmask)
    # edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
    # clothes = clothes * edge

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
    grid = utils.make_grid(p_tryon).cpu().detach()
    grid = grid / 2 + 0.5
    # imshow(grid.cpu().detach())
    # print(p_tryon.permute((1, 2, 0)).numpy(force= True).shape)
    pil_img = to_pil_image(grid, mode ="RGB")
    
    # print(np.array(pil_img))
    # path = 'results/' + opt.name
    # os.makedirs(path, exist_ok=True)
    # sub_path = path + '/PFAFN'
    # os.makedirs(sub_path,exist_ok=True)
    

    # # print(os.path.join('D:\FSVTONdata//our_t_results', data['p_name']))
    # # save try-on image only

    # utils.save_image(
    # pil_img,
    # os.path.join('D:/FSVTONdata/our_t_results', "fisttest"),
    # nrow=int(1),
    # normalize=True,
    # value_range=(-1,1),
    # )

    return pil_img

# css = ".output_image {height: 40rem !important; width: 100% !important;}"
demo = gr.Interface(fn = sepia, 
                    inputs = [gr.Image( height=256, width=192, type ="pil"), gr.Image( height=256, width=192, type ="pil"),gr.Image( height=256, width=192, type ="pil")], 
                    outputs= gr.components.Image(height=256, width=192, type="pil", label="image")
                    )
    
demo.launch(share=True)  # Share your demo with just 1 extra parameter 