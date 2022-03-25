from types import MethodType
import numpy as np
import torch
from torch import nn


def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    empty = []
    for i in range(8):
        command = 'nvidia-smi -i '+str(i)+' | grep "No running" | wc -l'
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        #print('nvidia-smi -i '+str(i)+' | grep "No running" | wc -l > empty_gpu_check')
        if int(output)==1:
            empty.append(i)
    if len(empty)<ngpus:
        print ('avaliable gpus are less than required')
        exit(-1)
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
    return cmd

def initialize_model(model, device, load_save_file=False, gpu=True):
    if load_save_file:
        if gpu:
            model.load_state_dict(torch.load(load_save_file)) 
        else:
            model.load_state_dict(torch.load(load_save_file, map_location=torch.device('cpu')))  
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                #nn.init.normal(param, 0.0, 0.15)
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)
    model.to(device)
    return model

def parse_to_dict(args):
    args_dict = {}
    for arg in dir(args):
        if not arg.startswith('__') and not isinstance(getattr(args,arg), MethodType):
            if getattr(args , arg) is not None:
                args_dict[arg] = getattr(args,arg)

    return args_dict

def collate_fn_padd(batch):
    """
    Padds batch of variable length

    Note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    """
    ## Get sequence lengths
    lengths = [t[1].shape[0] for t in batch]
    max_length = max(lengths)
    if max_length == 0:
        max_length += 1
    batch_size = len(lengths)

    # 5000 is tf-idf dimension, change it when calling any args
    user = np.zeros((batch_size, 20))
    tweet = np.zeros((batch_size, max_length, 5000))
    adj = np.zeros((batch_size, max_length, max_length))
    up = np.zeros((batch_size, max_length))
    label = np.zeros((batch_size, 1))
    tlabel = np.zeros((batch_size, max_length, 1))

    # Padding
    for i, (us, t, a, u, lab, tlab) in enumerate(batch):
        l = lengths[i]
        user[i] = us
        if l != 0:
            tweet[i, :l] = t
            adj[i, :l, :l] = a
            up[i, :l] = u
        label[i] = lab
        tlabel[i, :l] = tlab[:, np.newaxis]
    
    user = torch.from_numpy(user).float()
    tweet = torch.from_numpy(tweet).float()
    adj = torch.from_numpy(adj).float()
    up = torch.from_numpy(up).float()
    label = torch.from_numpy(label).float()
    tlabel = torch.from_numpy(tlabel).float()

    return user, tweet, adj, up, label, tlabel

def tweet_loss_fn(input, output, up_masking):
    losses = list()
    for i in range(len(input)):
        mask = torch.where(up_masking[i] == 1.0, True, False)
        logits = input[i][mask]
        targets = output[i][mask]
        loss = torch.nn.functional.binary_cross_entropy(logits, targets)
        losses.append(loss)
    losses = torch.mean(torch.Tensor(losses))
    return losses
