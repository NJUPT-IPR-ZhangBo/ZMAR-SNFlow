import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
from PIL import Image

import utils.util as util
from models import create_model

import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from utils.util import opt_get
from data import create_dataloader
from models import create_model
from data.LoL_dataset import LoL_Dataset, LoL_Dataset_v2
from skimage.metrics import peak_signal_noise_ratio as Psnr
from skimage.metrics import structural_similarity as SSIM


def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    # opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", default="confs/LOLv2-pc.yml")
    args = parser.parse_args()
    conf_path = args.opt
    model, opt = load_model(conf_path)

    # # 使用 DataParallel 将模型分配到多个GPU上
    # device_ids = [0, 1]  # 根据需要修改使用的GPU ID
    # model.netG = torch.nn.DataParallel(model.netG, device_ids=device_ids)
    # model.netG = model.netG.cuda(device_ids[0])  #
    # model.netG.eval()
    # ##############################################################
    model.netG = model.netG.cuda()  # 原本测试只有这一行代码

    save_imgs = True
    save_folder = 'results/{}'.format(opt['name'])

    output_folder = osp.join(save_folder, 'images/output')

    util.mkdirs(save_folder)

    util.mkdirs(output_folder)


    print('mkdir finish')

    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    if opt['dataset'] == 'LoL':
        dataset_cls = LoL_Dataset
    elif opt['dataset'] == 'LoL_v2':
        dataset_cls = LoL_Dataset_v2

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = dataset_cls(opt=dataset_opt, train=False, all_opt=opt)
            val_loader = create_dataloader(False, val_set, dataset_opt, opt, None)

    psnr_total_avg = 0.
    ssim_total_avg = 0.
    idx = 0

    for val_data in val_loader:
        idx += 1
        img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
        model.feed_data(val_data)
        model.test()
        visuals = model.get_current_visuals()
        rlt_img = util.tensor2img(visuals['NORMAL'])  # uint8
        gt_img = util.tensor2img(visuals['GT'])  # uint8

        input_img = util.tensor2img(visuals['LQ'])
        if save_imgs:
            try:
                print(idx)
                cv2.imwrite(osp.join(output_folder, '{}.png'.format(img_name)), rlt_img)
                # cv2.imwrite(osp.join(GT_folder, '{}.png'.format(img_name)), gt_img)
                #
                # cv2.imwrite(osp.join(input_folder, '{}.png'.format(img_name)), input_img)

            except Exception as e:
                print(e)
                import ipdb
                ipdb.set_trace()

        # calculate PSNR
        # psnr = util.calculate_psnr(rlt_img, gt_img)
        psnr = Psnr(rlt_img, gt_img)
        psnr_total_avg = psnr_total_avg + psnr

        # ssim = util.ssim(visuals['NORMAL'].unsqueeze(0), visuals['GT'].unsqueeze(0))
        # ssim = util.calculate_ssim(rlt_img, gt_img)
        ssim = SSIM(rlt_img, gt_img, channel_axis=2)
        ssim_total_avg = ssim_total_avg + ssim

    print(psnr_total_avg / idx)
    print(ssim_total_avg / idx)


if __name__ == '__main__':

    time_start = time.time()  # 记录开始时间
    main()
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)
