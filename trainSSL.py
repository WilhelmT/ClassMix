import argparse
import os
import sys
import random
import timeit
import datetime

import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform

from model.deeplabv2 import Res_Deeplab

from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d

from utils import transformmasks
from utils import transformsgpu
from utils.helpers import colorize_mask
import utils.palette as palette

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from data.voc_dataset import VOCDataSet

from data import get_loader, get_data_path
from data.augmentations import *
from tqdm import tqdm

import PIL
from torchvision import transforms
import json
from torch.utils import tensorboard
from evaluateSSL import evaluate

import time

start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpus", type=int, default=1,
                        help="choose number of gpu devices to use (default: 1)")
    parser.add_argument("-c", "--config", type=str, default='config.json',
                        help='Path to the config file (default: config.json)')
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help='Path to the .pth file to resume from (default: None)')
    parser.add_argument("-n", "--name", type=str, default=None, required=True,
                        help='Name of the run (default: None)')
    parser.add_argument("--save-images", type=str, default=None,
                        help='Include to save images (default: None)')
    return parser.parse_args()



def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    if len(gpus) > 1:
        criterion = torch.nn.DataParallel(CrossEntropy2d(ignore_label=ignore_label), device_ids=gpus).cuda()  # Ignore label ??
    else:
        criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  # Ignore label ??

    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def sigmoid_ramp_up(iter, max_iter):
    if iter >= max_iter:
        return 1
    else:
        return np.exp(- 5 * (1 - iter / max_iter) ** 2)

def create_ema_model(model):
    ema_model = Res_Deeplab(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    if len(gpus)>1:
        if use_sync_batchnorm:
            ema_model = convert_model(ema_model)
            ema_model = DataParallelWithCallback(ema_model, device_ids=gpus)
        else:
            ema_model = torch.nn.DataParallel(ema_model, device_ids=gpus)
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus)>1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.mix(mask = parameters["Mix"], data = data, target = target)
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = None)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def getWeakInverseTransformParameters(parameters):
    return parameters

def getStrongInverseTransformParameters(parameters):
    return parameters

class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

class Learning_Rate_Object(object):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])

            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('../visualiseImages/', str(epoch)+ id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('../visualiseImages/', str(epoch)+ id + '.png'))

def _save_checkpoint(iteration, model, optimizer, config, ema_model, save_best=False, overwrite=True):
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    if len(gpus) > 1:
        checkpoint['model'] = model.module.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.module.state_dict()
    else:
        checkpoint['model'] = model.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.state_dict()

    if save_best:
        filename = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(checkpoint, filename)
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(checkpoint_dir, f'checkpoint-iter{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(checkpoint_dir, f'checkpoint-iter{iteration - save_checkpoint_every}.pth'))
            except:
                pass

def _resume_checkpoint(resume_path, model, optimizer, ema_model):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    iteration = checkpoint['iteration'] + 1
    print('Starting at iteration: ' + str(iteration))

    if len(gpus) > 1:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    if train_unlabeled:
        if len(gpus) > 1:
            ema_model.module.load_state_dict(checkpoint['ema_model'])
        else:
            ema_model.load_state_dict(checkpoint['ema_model'])

    return iteration, model, optimizer, ema_model

def main():
    print(config)

    best_mIoU = 0

    if consistency_loss == 'CE':
        if len(gpus) > 1:
            unlabeled_loss = torch.nn.DataParallel(CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label), device_ids=gpus).cuda()
        else:
            unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    elif consistency_loss == 'MSE':
        if len(gpus) > 1:
            unlabeled_loss =  torch.nn.DataParallel(MSELoss2d(), device_ids=gpus).cuda()
        else:
            unlabeled_loss =  MSELoss2d().cuda()

    cudnn.enabled = True

    # create network
    model = Res_Deeplab(num_classes=num_classes)

    # load pretrained parameters
    if restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(restore_from)
    else:
        saved_state_dict = torch.load(restore_from)

    # Copy loaded parameters to model
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)

    # Initiate ema-model
    if train_unlabeled:
        ema_model = create_ema_model(model)
        ema_model.train()
        ema_model = ema_model.cuda()
    else:
        ema_model = None

    if len(gpus)>1:
        if use_sync_batchnorm:
            model = convert_model(model)
            model = DataParallelWithCallback(model, device_ids=gpus)
        else:
            model = torch.nn.DataParallel(model, device_ids=gpus)
    model.train()
    model.cuda()

    cudnn.benchmark = True

    if dataset == 'pascal_voc':
        data_loader = get_loader(dataset)
        data_path = get_data_path(dataset)
        train_dataset = data_loader(data_path, crop_size=input_size, scale=random_scale, mirror=random_flip)

    elif dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        if random_crop:
            data_aug = Compose([RandomCrop_city(input_size)])
        else:
            data_aug = None

        train_dataset = data_loader(data_path, is_transform=True, augmentations=data_aug, img_size=input_size)

    train_dataset_size = len(train_dataset)
    print ('dataset size: ', train_dataset_size)

    partial_size = labeled_samples
    print('Training on number of samples:', partial_size)
    if split_id is not None:
        train_ids = pickle.load(open(split_id, 'rb'))
        print('loading train ids from {}'.format(split_id))
    else:
        np.random.seed(random_seed)
        train_ids = np.arange(train_dataset_size)
        np.random.shuffle(train_ids)

    train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
    trainloader = data.DataLoader(train_dataset,
                    batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    trainloader_iter = iter(trainloader)

    if train_unlabeled:
        train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
        trainloader_remain = data.DataLoader(train_dataset,
                    batch_size=batch_size, sampler=train_remain_sampler, num_workers=1, pin_memory=True)
        trainloader_remain_iter = iter(trainloader_remain)

    # Optimizer for segmentation network
    learning_rate_object = Learning_Rate_Object(config['training']['learning_rate'])

    if optimizer_type == 'SGD':
        if len(gpus) > 1:
            optimizer = optim.SGD(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)

    optimizer.zero_grad()

    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)

    start_iteration = 0

    if args.resume:
        start_iteration, model, optimizer, ema_model = _resume_checkpoint(args.resume, model, optimizer, ema_model)

    accumulated_loss_l = []
    if train_unlabeled:
        accumulated_loss_u = []

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/config.json', 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=False)
    pickle.dump(train_ids, open(os.path.join(checkpoint_dir, 'train_split.pkl'), 'wb'))

    epochs_since_start = 0
    for i_iter in range(start_iteration, num_iterations):
        model.train()

        loss_l_value = 0
        if train_unlabeled:
            loss_u_value = 0

        optimizer.zero_grad()

        if lr_schedule:
            adjust_learning_rate(optimizer, i_iter)

        # Training loss for labeled data only
        try:
            batch = next(trainloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(trainloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            print('Epochs since start: ',epochs_since_start)
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        weak_parameters={"flip": 0}

        images, labels, _, _, _ = batch
        images = images.cuda()
        labels = labels.cuda()

        images, labels = weakTransform(weak_parameters, data = images, target = labels)
        pred = interp(model(images))

        L_l = loss_calc(pred, labels)

        if train_unlabeled:
            try:
                batch_remain = next(trainloader_remain_iter)
                if batch_remain[0].shape[0] != batch_size:
                    batch_remain = next(trainloader_remain_iter)
            except:
                trainloader_remain_iter = iter(trainloader_remain)
                batch_remain = next(trainloader_remain_iter)

            images_remain, _, _, _, _ = batch_remain
            images_remain = images_remain.cuda()
            inputs_u_w, _ = weakTransform(weak_parameters, data = images_remain)
            logits_u_w = interp(ema_model(inputs_u_w))
            logits_u_w, _ = weakTransform(getWeakInverseTransformParameters(weak_parameters), data = logits_u_w.detach())

            softmax_u_w = torch.softmax(logits_u_w.detach(), dim=1)
            max_probs, argmax_u_w = torch.max(softmax_u_w, dim=1)

            if mix_mask == "class":

                for image_i in range(batch_size):
                    classes = torch.unique(argmax_u_w[image_i])
                    classes = classes[classes != ignore_label]
                    nclasses = classes.shape[0]
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()]).cuda()
                    if image_i == 0:
                        MixMask = transformmasks.generate_class_mask(argmax_u_w[image_i], classes).unsqueeze(0).cuda()
                    else:
                        MixMask = torch.cat((MixMask,transformmasks.generate_class_mask(argmax_u_w[image_i], classes).unsqueeze(0).cuda()))

            elif mix_mask == 'cut':
                img_size = inputs_u_w.shape[2:4]
                for image_i in range(batch_size):
                    if image_i == 0:
                        MixMask = torch.from_numpy(transformmasks.generate_cutout_mask(img_size)).unsqueeze(0).cuda().float()
                    else:
                        MixMask = torch.cat((MixMask, torch.from_numpy(transformmasks.generate_cutout_mask(img_size)).unsqueeze(0).cuda().float()))

            elif mix_mask == "cow":
                img_size = inputs_u_w.shape[2:4]
                sigma_min = 8
                sigma_max = 32
                p_min = 0.5
                p_max = 0.5
                for image_i in range(batch_size):
                    sigma = np.exp(np.random.uniform(np.log(sigma_min), np.log(sigma_max)))     # Random sigma
                    p = np.random.uniform(p_min, p_max)     # Random p
                    if image_i == 0:
                        MixMask = torch.from_numpy(transformmasks.generate_cow_mask(img_size, sigma, p, seed=None)).unsqueeze(0).cuda().float()
                    else:
                        MixMask = torch.cat((MixMask,torch.from_numpy(transformmasks.generate_cow_mask(img_size, sigma, p, seed=None)).unsqueeze(0).cuda().float()))

            elif mix_mask == None:
                MixMask = torch.ones((inputs_u_w.shape))

            strong_parameters = {"Mix": MixMask}
            if random_flip:
                strong_parameters["flip"] = random.randint(0, 1)
            else:
                strong_parameters["flip"] = 0
            if color_jitter:
                strong_parameters["ColorJitter"] = random.uniform(0, 1)
            else:
                strong_parameters["ColorJitter"] = 0
            if gaussian_blur:
                strong_parameters["GaussianBlur"] = random.uniform(0, 1)
            else:
                strong_parameters["GaussianBlur"] = 0

            inputs_u_s, _ = strongTransform(strong_parameters, data = images_remain)
            logits_u_s = interp(model(inputs_u_s))

            softmax_u_w_mixed, _ = strongTransform(strong_parameters, data = softmax_u_w)
            max_probs, pseudo_label = torch.max(softmax_u_w_mixed, dim=1)

            if pixel_weight == "threshold_uniform":
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
                pixelWiseWeight = unlabeled_weight * torch.ones(max_probs.shape).cuda()
            elif pixel_weight == "threshold":
                pixelWiseWeight = max_probs.ge(0.968).long().cuda()
            elif pixel_weight == 'sigmoid':
                max_iter = 10000
                pixelWiseWeight = sigmoid_ramp_up(i_iter, max_iter) * torch.ones(max_probs.shape).cuda()
            elif pixel_weight == False:
                pixelWiseWeight = torch.ones(max_probs.shape).cuda()

            if consistency_loss == 'CE':
                L_u = consistency_weight * unlabeled_loss(logits_u_s, pseudo_label, pixelWiseWeight)
            elif consistency_loss == 'MSE':
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
                #softmax_u_w_mixed = torch.cat((softmax_u_w_mixed[1].unsqueeze(0),softmax_u_w_mixed[0].unsqueeze(0)))
                L_u = consistency_weight * unlabeled_weight * unlabeled_loss(logits_u_s, softmax_u_w_mixed)

            loss = L_l + L_u

        else:
            loss = L_l

        if len(gpus) > 1:
            loss = loss.mean()
            loss_l_value += L_l.mean().item()
            if train_unlabeled:
                loss_u_value += L_u.mean().item()
        else:
            loss_l_value += L_l.item()
            if train_unlabeled:
                loss_u_value += L_u.item()

        loss.backward()
        optimizer.step()

        # update Mean teacher network
        if ema_model is not None:
            alpha_teacher = 0.99
            ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=alpha_teacher, iteration=i_iter)

        if train_unlabeled:
            print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}, loss_u = {3:.3f}'.format(i_iter, num_iterations, loss_l_value, loss_u_value))
        else:
            print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}'.format(i_iter, num_iterations, loss_l_value))

        if i_iter % save_checkpoint_every == 0 and i_iter != 0:
            _save_checkpoint(i_iter, model, optimizer, config, ema_model)

        if use_tensorboard:
            if 'tensorboard_writer' not in locals():
                tensorboard_writer = tensorboard.SummaryWriter(log_dir, flush_secs=30)

            accumulated_loss_l.append(loss_l_value)
            if train_unlabeled:
                accumulated_loss_u.append(loss_u_value)
            if i_iter % log_per_iter == 0 and i_iter != 0:

                tensorboard_writer.add_scalar('Training/Supervised loss', np.mean(accumulated_loss_l), i_iter)
                accumulated_loss_l = []

                if train_unlabeled:
                    tensorboard_writer.add_scalar('Training/Unsupervised loss', np.mean(accumulated_loss_u), i_iter)
                    accumulated_loss_u = []


        if i_iter % val_per_iter == 0 and i_iter != 0:
            model.eval()
            mIoU, eval_loss = evaluate(model, dataset, ignore_label=ignore_label, input_size=(512,1024), save_dir=checkpoint_dir)

            model.train()

            if mIoU > best_mIoU and save_best_model:
                best_mIoU = mIoU
                _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)

            if use_tensorboard:
                tensorboard_writer.add_scalar('Validation/mIoU', mIoU, i_iter)
                tensorboard_writer.add_scalar('Validation/Loss', eval_loss, i_iter)

        if save_unlabeled_images and train_unlabeled and i_iter % save_checkpoint_every == 0:
            # Saves two mixed images and the corresponding prediction
            save_image(inputs_u_s[0].cpu(),i_iter,'input1',palette.CityScpates_palette)
            save_image(inputs_u_s[1].cpu(),i_iter,'input2',palette.CityScpates_palette)
            _, pred_u_s = torch.max(logits_u_s, dim=1)
            save_image(pred_u_s[0].cpu(),i_iter,'pred1',palette.CityScpates_palette)
            save_image(pred_u_s[1].cpu(),i_iter,'pred2',palette.CityScpates_palette)

    _save_checkpoint(num_iterations, model, optimizer, config, ema_model)

    model.eval()
    mIoU, val_loss = evaluate(model, dataset, ignore_label=ignore_label, input_size=(512,1024), save_dir=checkpoint_dir)

    model.train()
    if mIoU > best_mIoU and save_best_model:
        best_mIoU = mIoU
        _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)

    if use_tensorboard:
        tensorboard_writer.add_scalar('Validation/mIoU', mIoU, i_iter)
        tensorboard_writer.add_scalar('Validation/Loss', val_loss, i_iter)

    end = timeit.default_timer()
    print('Total time: ' + str(end-start) + ' seconds')

if __name__ == '__main__':

    print('---------------------------------Starting---------------------------------')

    args = get_arguments()

    if args.resume:
        config = torch.load(args.resume)['config']
    else:
        config = json.load(open(args.config))

    model = config['model']
    dataset = config['dataset']

    if dataset == 'cityscapes':
        IMG_MEAN = np.flip(np.array([73.15835921, 82.90891754, 72.39239876]))
        num_classes = 19
        if config['training']['data']['split_id_list'] == 0:
            split_id = './splits/city/split_0.pkl'
        elif config['training']['data']['split_id_list'] == 1:
            split_id = './splits/city/split_1.pkl'
        elif config['training']['data']['split_id_list'] == 2:
            split_id = './splits/city/split_2.pkl'
        else:
            split_id = None

    elif dataset == 'pascal_voc':
        IMG_MEAN = np.flip(np.array([104.00698793,116.66876762,122.67891434]))
        num_classes = 21
        data_dir = './data/voc_dataset/'
        data_list_path = './data/voc_list/train_aug.txt'
        if config['training']['data']['split_id_list'] == 0:
            split_id = './splits/voc/split_0.pkl'
        else:
            split_id = None

    if config['pretrained'] == 'coco':
        restore_from = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'

    batch_size = config['training']['batch_size']
    num_iterations = config['training']['num_iterations']

    input_size_string = config['training']['data']['input_size']
    h, w = map(int, input_size_string.split(','))
    input_size = (h, w)

    ignore_label = config['ignore_label'] # 255 for PASCAL-VOC / 250 for Cityscapes

    learning_rate = config['training']['learning_rate']

    optimizer_type = config['training']['optimizer']
    lr_schedule = config['training']['lr_schedule']
    lr_power = config['training']['lr_schedule_power']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    num_workers = config['training']['num_workers']
    use_sync_batchnorm = config['training']['use_sync_batchnorm']
    random_seed = config['seed']

    labeled_samples = config['training']['data']['labeled_samples']

    #unlabeled CONFIGURATIONS
    train_unlabeled = config['training']['unlabeled']['train_unlabeled']
    mix_mask = config['training']['unlabeled']['mix_mask']
    pixel_weight = config['training']['unlabeled']['pixel_weight']
    consistency_loss = config['training']['unlabeled']['consistency_loss']
    consistency_weight = config['training']['unlabeled']['consistency_weight']
    random_flip = config['training']['unlabeled']['flip']
    color_jitter = config['training']['unlabeled']['color_jitter']
    gaussian_blur = config['training']['unlabeled']['blur']

    random_scale = config['training']['data']['scale']
    random_crop = config['training']['data']['crop']

    save_checkpoint_every = config['utils']['save_checkpoint_every']
    if args.resume:
        checkpoint_dir = os.path.join(*args.resume.split('/')[:-1]) + '_resume-' + start_writeable
    else:
        checkpoint_dir = os.path.join(config['utils']['checkpoint_dir'], start_writeable + '-' + args.name)
    log_dir = checkpoint_dir

    val_per_iter = config['utils']['val_per_iter']
    use_tensorboard = config['utils']['tensorboard']
    log_per_iter = config['utils']['log_per_iter']

    save_best_model = config['utils']['save_best_model']
    if args.save_images:
        print('Saving unlabeled images')
        save_unlabeled_images = True
    else:
        save_unlabeled_images = False

    gpus = (0,1,2,3)[:args.gpus]

    main()
