# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision import  transforms
import pickle
from time import time
import io 

import os.path
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from pathlib import Path
import json
import sys
import math
from scipy.spatial.transform import Rotation as R
import copy
import neptune
from tqdm import tqdm
from functools import partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl

import argparse
import src.resnet as resnet
# import src.models_no_dist as m
import src.models as m
from src.sr_capsnet import SelfRouting2d

def create_directory_if_not_exists(directory_path):
    """
    Creates a directory if it does not exist.

    Parameters:
    directory_path (str): The path of the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

parser = argparse.ArgumentParser()

parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--caps-type", type=str, choices=["SR","VC"], default="SR")
parser.add_argument("--caps-depth", type=int, default=1)
parser.add_argument("--temp_init", nargs="+", type=float, default=[0.10, 0.25])
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--deep-end", action="store_true",help="If used, uses a MLP instead of linear head")
parser.add_argument("--equi-dims",type=int,default=512,help="Number of equivariant dimensions (to evaluate). Put the full size to evaluate the whole representation.")
parser.add_argument("--inv-part",action="store_true",help="Whether or not to evaluate the invariant part")

parser.add_argument("--representation", action="store_true",help="Whether or not to evaluate the representations")

parser.add_argument("--equi", type=int, default=256)
parser.add_argument("--experience", type=str, choices=["SIENoVar","SIE","SIEOnlyEqui","VICReg","SimCLR","VICRegPartInv",
                                                        "SimCLROnlyEqui","SIERotColor","SimCLRAugSelf","SimCLRAugSelfRotColor",
                                                        "SimCLROnlyEquiRotColor","SimCLREquiModRotColor","SimCLREquiMod","VICRegEquiMod", "SIECaps", "SIECaps2", "SIECaps3"],
                                                        default="SIE")
parser.add_argument("--experience-old", type=str, choices=["quat","euler"],default="quat",help="Whether to use Euler angles or quaternions for the targets.")
parser.add_argument("--mlp", default="2048-2048-2048")
parser.add_argument("--predictor", default="")
parser.add_argument("--pred-size-in",type=int, default=10)
parser.add_argument("--predictor-relu",  action="store_true")
parser.add_argument("--hypernetwork", type=str, choices=["linear","deep"],default="linear")
parser.add_argument("--no-activation-checkpoint",  action="store_false")

# Predictor
parser.add_argument("--predictor-type",type=str,choices=["hypernetwork","mlp"],default="hypernetwork")
parser.add_argument("--bias-pred", action="store_true")
parser.add_argument("--bias-hypernet", action="store_true")
parser.add_argument("--simclr-temp",type=float,default=0.1)
parser.add_argument("--ec-weight",type=float,default=1)
parser.add_argument("--tf-num-layers",type=int,default=1)

# Experience loading
parser.add_argument("--weights-file", type=str, default="./resnet50.pth")
parser.add_argument("--supervised",action="store_true")

# Optim
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0)

# Data
parser.add_argument("--dataset-root", type=Path, default="DATA_FOLDER", required=True)
# parser.add_argument("--images-file", type=Path, default="./data/train_images.npy", required=True)
# parser.add_argument("--labels-file", type=Path, default="./data/val_images.npy", required=True)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--preload", action="store_true")
parser.add_argument("--no-pickle", action="store_false")

# Checkpoints
parser.add_argument("--exp-dir", type=Path, default="")
parser.add_argument("--root-log-dir", type=Path,default="EXP_DIR/logs/")
parser.add_argument("--log-freq-time", type=int, default=10)
parser.add_argument("--logger_project", type=str, default='ABDN-DL/3DIECAPS')
parser.add_argument("--logger_name", type=str, default='InitialTesting')
parser.add_argument("--logger_mode", type=str, default='async')
parser.add_argument("--finetune", action="store_true", default=False)
parser.add_argument("--neptune-cp", type=str, default=None)
# Running
parser.add_argument("--num-workers", type=int, default=1)

args = parser.parse_args()

if args.neptune_cp != None:
    run = neptune.init_run(project=args.logger_project, with_id=args.neptune_cp, mode="read-only")
    create_directory_if_not_exists(f"checkpoints/{args.neptune_cp}")
    run["final_checkpoint"].download(destination=f"checkpoints/{args.neptune_cp}")
    args.weights_file = f"checkpoints/{args.neptune_cp}/final_checkpoint.pth"
    run.stop()

neptune_logger = neptune.init_run(api_token=None, project=args.logger_project, name=args.logger_name, mode=args.logger_mode)
neptune_logger["parameters"] = args
neptune_logger["sys/description"] = f"Eval angle prediction of weights: {str(args.weights_file)} repr: {args.representation}"
neptune_logger["eval_task"] = True
neptune_logger["specific_task"] = "Angle"

url = neptune_logger.get_url()
run_id = url.split("/")[-1]

print("\n"+("="*60))
print(f"run_id: {run_id}")
print(("="*60)+"\n")

# args.exp_dir = Path(os.path.join(args.exp_dir, run_id))

class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = m.__dict__[args.experience](args).cuda()

        self.equi_dims = args.equi_dims
        self.out_dim = 3
        self.inv = args.inv_part
        if args.experience_old == "quat":
            self.out_dim = 4
            
        self.num_caps = int(args.mlp.split("-")[-1])

        print(f"Making model")

        self.representation = args.representation

        if self.representation:
            self.in_dims = self.model.repr_size
        else:
            if not self.inv:
                self.in_dims = self.num_caps * 16
            else:
                self.in_dims = self.num_caps


        weights_file = Path(args.weights_file)

        print(f"Args weights file: {weights_file}")

        if (weights_file).is_file():
            ckpt = torch.load(weights_file, map_location="cpu")
            # print(ckpt.keys())
            try:
                new_ckpt = {k.replace('module.',''):v for k,v in ckpt["model"].items()}
            except KeyError:
                new_ckpt = {k.replace('module.',''):v for k,v in ckpt.items()}
            self.model.load_state_dict(new_ckpt)

        print(f"MAKING LINEAR WITH {self.in_dims*2}")

        if args.deep_end:
            self.head = nn.Sequential(
                nn.Linear(self.in_dims*2,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, self.out_dim),
            )
        else:
            self.head = nn.Linear(self.in_dims*2, self.out_dim)

        if args.finetune == True:
            self.finetune = True
        else:
            self.finetune = False

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, y):

        b, c, h, w = x.size()

        if self.finetune == True:
            # with torch.no_grad():
                x_repr = self.model.backbone(x)
                # x_repr_pool = self.avgpool(x_repr).reshape(x_repr.size(0), -1)
                x_inv, x_equi = self.model.projector.forward_other(x_repr)
                x_equi = self.avgpool(x_equi)
                x_equi = x_equi.view(b, -1)


                y_repr = self.model.backbone(y)
                # x_repr_pool = self.avgpool(x_repr).reshape(x_repr.size(0), -1)
                y_inv, y_equi = self.model.projector.forward_other(y_repr)
                y_equi = self.avgpool(y_equi)
                y_equi = y_equi.view(b, -1)

        else:
            with torch.no_grad():
                x_repr = self.model.backbone(x)
                # x_repr_pool = self.avgpool(x_repr).reshape(x_repr.size(0), -1)
                x_inv, x_equi = self.model.projector.forward_other(x_repr)
                x_equi = self.avgpool(x_equi)
                x_equi = x_equi.view(b, -1)


                y_repr = self.model.backbone(y)
                # x_repr_pool = self.avgpool(x_repr).reshape(x_repr.size(0), -1)
                y_inv, y_equi = self.model.projector.forward_other(y_repr)
                y_equi = self.avgpool(y_equi)
                y_equi = y_equi.view(b, -1)

        # print(y_equi.size())

        if not self.inv:
            # print("not self inv")
            concat = torch.cat([x_equi,y_equi],axis=1)
            # print(concat.size())
            out = self.head(concat)
        else:
            # print("yes self inv")
            concat = torch.cat([x_inv,y_inv],axis=1)
            out = self.head(concat)

        return out

class Dataset3DIEBench(Dataset):
    def __init__(self,dataset_root, img_file,experience, size_dataset=-1, transform=None):
        self.dataset_root = dataset_root
        self.samples = np.load(img_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.rng = np.random.RandomState()    
        self.experience = experience    

    def get_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img) 
        return img

    def __getitem__(self, i):
        # Latent vector creation
        views = self.rng.choice(50,2, replace=False)

        # print(str(self.dataset_root) + self.samples[i]+ f"/image_{views[0]}.jpg")

        img_1 = self.get_img(str(self.dataset_root) + self.samples[i]+ f"/image_{views[0]}.jpg")
        img_2 = self.get_img(str(self.dataset_root) + self.samples[i]+ f"/image_{views[1]}.jpg")         
    
        angles_1 =np.load(str(self.dataset_root) + self.samples[i]+ f"/latent_{views[0]}.npy")[:3].astype(np.float32)
        angles_2 =np.load(str(self.dataset_root) + self.samples[i]+ f"/latent_{views[1]}.npy")[:3].astype(np.float32)
        rot_1 = R.from_euler("xyz",angles_1)
        rot_2 = R.from_euler("xyz",angles_2)
        rot_1_to_2 = rot_1.inv()*rot_2
        if self.experience == "quat":
            angles = rot_1_to_2.as_quat().astype(np.float32)
        else:
            angles = rot_1_to_2.as_euler("xyz").astype(np.float32)

        return img_1, img_2, torch.FloatTensor(angles)

    def __len__(self):
        return len(self.samples)
 

normalize = transforms.Normalize(
       mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]
    )

class PickleDataset3DIEBench(Dataset):
    def __init__(self, dataset_root, img_file,labels_file,experience="quat", size_dataset=-1, transform=None, preload=False, val=False):
        self.dataset_root = dataset_root
        self.samples = np.load(img_file)
        self.labels = np.load(labels_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.labels = self.labels[:size_dataset]
        assert len(self.samples) == len(self.labels)
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.experience = "quat"

        print(f"MAKING DATASET Pickle Pickle")

        print(f"TIME TO LOAD THE PICKLE:")

        start_t = time()

        if val:
            pickle_file = "3die_val.pkl"
        else:
            pickle_file = "3die.pkl"

        file = open(pickle_file,'rb')
        self.lookup_dict = pickle.load(file)

        total_mmap_t = time() - start_t
        print(f'\nPickle loaded, took (s): {total_mmap_t} \n')

        # print(self.lookup_dict.keys())

    def get_img(self, path, idx):
        img = Image.open(io.BytesIO(self.lookup_dict[path][idx]["img"]))
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img) 
        return img
    
    def get_latent(self, path, idx):
        return self.lookup_dict[path][idx]["latent"]

    def __getitem__(self, i):

        # print(f"GETTING ITEM")

        label = self.labels[i]

        views = np.random.choice(50, 2, replace=False)

        # print(self.samples[i][1:][:-1])

        # keys in dict will be in format e.g. 04401088/9fac50c7b7c72dc694f8f49303e93f14
        img_1 = self.get_img(self.samples[i][1:][:-1], views[0])
        img_2 = self.get_img(self.samples[i][1:][:-1], views[1])         
    
        angles_1 = self.get_latent(self.samples[i][1:][:-1], views[0])
        angles_2 = self.get_latent(self.samples[i][1:][:-1], views[1])

        rot_1 = R.from_euler("xyz",angles_1)
        rot_2 = R.from_euler("xyz",angles_2)
        rot_1_to_2 = rot_1.inv()*rot_2
        if self.experience == "quat":
            angles = rot_1_to_2.as_quat().astype(np.float32)
        else:
            angles = rot_1_to_2.as_euler("xyz").astype(np.float32)

        return img_1, img_2, torch.FloatTensor(angles)

    def __len__(self):
        return len(self.samples)

def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def exclude_bias_and_norm(p):
    return p.ndim == 1

def load_from_state_dict(model, state_dict, prefix, new_suffix):
        state_dict = copy.deepcopy(state_dict)
        state_dict = {
            k.replace(prefix, new_suffix): v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                print(
                    'key "{}" could not be found in provided state dict'.format(k)
                )
            elif state_dict[k].shape != v.shape:
                print(
                    'key "{}" is of different shape in model and provided state dict {} vs {}'.format(
                        k, v.shape, state_dict[k].shape
                    )
                )
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        print("Load pretrained model with msg: {}".format(msg))

### INIT STUFF

args.exp_dir.mkdir(parents=True, exist_ok=True)
args.root_log_dir.mkdir(parents=True, exist_ok=True)
print(" ".join(sys.argv))
with open(args.exp_dir / "params.json", 'w') as fp:
    pass

dict_args = deepcopy(vars(args))
for key,value in dict_args.items():
    if isinstance(value,Path):
        dict_args[key] = str(value)
with open(args.exp_dir / "params.json", 'w') as f:
    json.dump(dict_args, f)

if str(args.exp_dir)[-1] == "/":
    exp_name = str(args.exp_dir)[:-1].split("/")[-1]	
else:	
    exp_name = str(args.exp_dir).split("/")[-1]	
logdir = args.root_log_dir / exp_name
writer = SummaryWriter(log_dir=logdir)


### DATA
if args.no_pickle:
    ds_train = PickleDataset3DIEBench(args.dataset_root,"./data/train_images.npy", "./data/train_labels.npy",transform=transforms.Compose([ transforms.Resize((args.resolution,args.resolution)),transforms.ToTensor(),normalize]), preload=args.preload)
    ds_val = PickleDataset3DIEBench(args.dataset_root,"./data/val_images.npy", "./data/val_labels.npy",transform=transforms.Compose([ transforms.Resize((args.resolution,args.resolution)),transforms.ToTensor(),normalize]), preload=args.preload, val=True)

else:
    ds_train = Dataset3DIEBench(args.dataset_root,
                                "./data/train_images.npy",args.experience,
                                transform=transforms.Compose([transforms.Resize((args.resolution,args.resolution)),transforms.ToTensor(),normalize]))
    ds_val = Dataset3DIEBench(args.dataset_root,
                                "./data/val_images.npy",args.experience,
                                transform=transforms.Compose([transforms.Resize((args.resolution,args.resolution)),transforms.ToTensor(),normalize]))

train_loader = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=True,)
val_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=True,)


## MODEL AND OPTIM

net = Model(args)
# Change number of output dimensions to match our problem
net = net.to(args.device)

# Activation checkpointing for SelfRouting2d layers.
if args.no_activation_checkpoint:
    print("ACTIVATION CHECKPOINTING CAPSULE LAYERS")
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, SelfRouting2d)

    apply_activation_checkpointing(net, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

epochs = args.epochs

# # Load and freeze the model
# ckpt  = torch.load(args.weights_file, map_location="cpu")
# if args.supervised:
#      load_from_state_dict(net.backbone,ckpt,prefix="backbone.", new_suffix="")
#      for param in net.backbone.parameters():
#         param.requires_grad = False
# else:
#     if "final_weights" in args.weights_file:
#         load_from_state_dict(net.backbone,ckpt,prefix="", new_suffix="")
#     else:
#         load_from_state_dict(net.backbone,ckpt["model"],prefix="module.backbone.", new_suffix="")
#     for param in net.backbone.parameters():
#         param.requires_grad = False

start_epoch = 0
## LOOP

# for epoch in range(start_epoch,epochs):
#     net.train()
#     for step, (inputs_1,inputs_2, latents) in enumerate(train_loader,start=epoch * len(train_loader)):
#         inputs_1 = inputs_1.to(args.device)
#         inputs_2 = inputs_2.to(args.device)
#         latents = latents.to(args.device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         outputs = net(inputs_1,inputs_2)
#         loss = F.mse_loss(outputs, latents)
#         r2 = r2_score(outputs,latents)
#         if step%args.log_freq_time == 0:
#             writer.add_scalar('Loss/loss', loss.item(), step)
#             writer.add_scalar('Metrics/train_R2', r2.item(), step)
#             writer.add_scalar('General/lr', args.lr, step)
#             writer.flush()

#         loss.backward()
#         optimizer.step()
#         if step%50 == 0 :
#             print(f"[Epoch {epoch}, step : {step}]: Loss: {loss.item():.6f}, R2 score: {r2.item():.3f}")
#     net.eval()
#     with torch.no_grad():
#         avg_mse = 0
#         len_ds = len(ds_val)
#         for i, (inputs_1,inputs_2, latents) in enumerate(val_loader):
#             inputs_1 = inputs_1.to(args.device)
#             inputs_2 = inputs_2.to(args.device)
#             latents = latents.to(args.device)
            
#             outputs = net(inputs_1,inputs_2)
#             mse = F.mse_loss(outputs,latents)
#             avg_mse += inputs_1.shape[0]*mse.item()/len_ds
#             if i == 0:
#                 total_latents = latents.cpu()
#                 total_preds = outputs.cpu()
#             else:
#                 total_latents = torch.cat((total_latents,latents.cpu()),axis=0)
#                 total_preds = torch.cat((total_preds,outputs.cpu()),axis=0)
#         r2 = r2_score(total_preds,total_latents)
#         writer.add_scalar('Metrics/val_MSE', avg_mse, step)
#         writer.add_scalar('Metrics/val_R2', r2.item(), step)
#         writer.flush()
#         print(f"[Epoch {epoch}, validation]: MSE: {avg_mse:.6f}, R2 score: {r2.item():.3f}")
train_loss = AverageMeter()
train_r2 = AverageMeter()
val_loss = AverageMeter()
val_r2 = AverageMeter()

for epoch in range(start_epoch,epochs):

    neptune_logger["train/epoch"].append((epoch))

    net.train()
    for step, (inputs_1,inputs_2, latents) in enumerate(tqdm(train_loader),start=epoch * len(train_loader)):
        inputs_1 = inputs_1.to(args.device)
        inputs_2 = inputs_2.to(args.device)
        latents = latents.to(args.device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(inputs_1,inputs_2)
        loss = F.mse_loss(outputs, latents)
        r2 = r2_score(outputs,latents)

        train_loss.update(loss, inputs_1.size()[0])
        train_r2.update(r2, inputs_1.size()[0])

        if step%args.log_freq_time == 0:
            writer.add_scalar('Loss/loss', loss.item(), step)
            writer.add_scalar('Metrics/train_R2', r2.item(), step)
            writer.add_scalar('General/lr', args.lr, step)
            writer.flush()
            neptune_logger["train/step_loss"].append(loss.item())
            neptune_logger["train/step_r2"].append(r2.item())

        loss.backward()
        optimizer.step()
        # if step%50 == 0 :
            # print(f"[Epoch {epoch}, step : {step}]: Loss: {loss.item():.6f}, R2 score: {r2.item():.3f}")
    
    print(f"[Epoch {epoch}, step : {step}]: Loss: {train_loss.avg:.6f}, R2 score: {train_r2.avg:.3f}")
    neptune_logger["train/epoch_loss"].append(train_loss.avg)
    neptune_logger["train/epoch_r2"].append(train_r2.avg)

    net.eval()
    with torch.no_grad():
        avg_mse = 0
        len_ds = len(ds_val)
        for i, (inputs_1,inputs_2, latents) in enumerate(val_loader):
            inputs_1 = inputs_1.to(args.device)
            inputs_2 = inputs_2.to(args.device)
            latents = latents.to(args.device)
            
            outputs = net(inputs_1,inputs_2)
            mse = F.mse_loss(outputs,latents)
            r2 = r2_score(outputs,latents)

            val_loss.update(mse, inputs_1.size()[0])
            val_r2.update(r2, inputs_1.size()[0])

            avg_mse += inputs_1.shape[0]*mse.item()/len_ds
            if i == 0:
                total_latents = latents.cpu()
                total_preds = outputs.cpu()
            else:
                total_latents = torch.cat((total_latents,latents.cpu()),axis=0)
                total_preds = torch.cat((total_preds,outputs.cpu()),axis=0)
        r2 = r2_score(total_preds,total_latents)
        writer.add_scalar('Metrics/val_MSE', avg_mse, step)
        writer.add_scalar('Metrics/val_R2', r2.item(), step)
        writer.flush()
        print(f"[Epoch {epoch}, validation]: MSE: {avg_mse:.6f}, R2 score: {r2.item():.3f}")
        print(f"[Epoch {epoch}, DOOP]: MSE: {val_loss.avg:.6f}, R2 score: {val_r2.avg:.3f}")
    
        neptune_logger["val/epoch_loss"].append(val_loss.avg)
        neptune_logger["val/epoch_r2"].append(val_r2.avg)

    train_loss.reset()
    train_r2.reset()
    val_loss.reset()
    val_r2.reset()

    ## CHECKPOINT
    state = dict(
                epoch=epoch + 1,
                model=net.state_dict(),
                optimizer=optimizer.state_dict(),
            )
    torch.save(state, args.exp_dir / "model.pth")

torch.save(net.state_dict(), args.exp_dir / "final_eval_weights.pth")

neptune_logger["final_checkpoint"].upload(str(args.exp_dir) +  "/final_eval_weights.pth")

neptune_logger.stop()

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass
