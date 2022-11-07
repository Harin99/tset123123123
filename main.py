from ast import arg, parse
import sys
import os
import cv2
from unittest import result
from xml.dom import ValidationErr
import time
import numpy as np
import matplotlib.pyplot as plt
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"]="1"
from loader.loader_mvsec_flow import *
from loader.loader_dsec import *
from loader.loader_indoor_flying import *
from utils.logger import *
import utils.helper_functions as helper
import json
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data.dataloader import DataLoader
from utils import visualization as visu
import argparse
from test import *
import git
import datetime
import wandb
from utils import image_utils
import torch.nn
import shutil
sys.path.append('model')
from model import eraft

from torch.utils.tensorboard import SummaryWriter 

try:
    ## AMP : 신경망 수렴 및 GPU 메모리 소모, 연산 속도를 증가시켜주는 library 가 속한 pkg
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

# exclude extremly large displacements ( For DSEC Dataset )
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

def save_ckp(state, checkpoint_dir):
    # Checkpoint save ( model + optimizer )
    torch.save(state, checkpoint_dir)

def initialize_tester(config):
    # Warm Start
    if config['subtype'].lower() == 'warm_start':
        return TestRaftEventsWarm
    # Classic
    else:
        return TestRaftEvents

def get_visualizer(args): 
    # DSEC dataset
    if args.dataset.lower() == 'dsec':
        return visualization.DsecFlowVisualizer
    # MVSEC dataset
    else:
        return visualization.FlowVisualizerEvents

def sequence_loss(flow_preds, flow_gt, valid, gamma = 0.8, max_flow = MAX_FLOW): 
    """ Loss function defined over sequence of flow predictions """
    # exclude invalid pixels and extremely large displacements ( Only in dsec dataset )
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    inner = mag < max_flow
    for i in range(inner.shape[0]): #batch만큼
        inner_mask = inner[i]
        inner_mask = torch.stack([inner_mask, inner_mask], dim=0)
        valid[i] = (valid[i] == True) & (inner_mask == True)
    
    ## Loss 
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i)
        criterion = nn.L1Loss()
        # Compute all batch once  
        i_flow_preds = flow_preds[i].view(-1)[valid.view(-1)]
        i_flow_gt = flow_gt.view(-1)[valid.view(-1)]
        i_loss = criterion(i_flow_preds, i_flow_gt)
        flow_loss += i_weight * i_loss
    
    # Average by batch_size
    flow_loss = flow_loss / (flow_preds[0].size(0))
    
    ## metric using last flow
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    # masking ( valid[:, 0, :, :] == valid[:, 1, :, :])
    valid_mask = valid[:, 0, :, :].cpu()
    epe = epe.view(-1)[valid_mask.view(-1)]
    
    # N-pe error 
    count_1px = torch.sum(( epe < 1 ).float())
    count_2px = torch.sum(( epe < 2 ).float())
    count_3px = torch.sum(( epe < 3 ).float())
    ratio_1px = ((len(epe) - count_1px) / len(epe)) * 100
    ratio_2px = ((len(epe) - count_2px) / len(epe)) * 100 
    ratio_3px = ((len(epe) - count_3px) / len(epe)) * 100
    
    metrics = {
        'epe': epe.mean().item(),
        '1pe': ratio_1px, 
        '2pe': ratio_2px,
        '3pe': ratio_3px,
    }

    return flow_loss, metrics

def count_parameters(model): 
    # When starting the train process, print the total # of parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model, batch_size, train_set_loader): # Train Code
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    # pct_start : lr 을 언제까지 증가시킬지 epoch에 대한 비율로 나타냄 _ default : 0.3 ( e.g 100 epochs 중 30epoch까지 증가 )
    # total_steps : ( # of epoch * steps_per_epoch ) 
    # steps_per_epoch = len(train_set_loader) [전체 데이터셋 개수 / batch size]
    if isinstance(train_set_loader, list) == True:
        length = len(train_set_loader[0]) + len(train_set_loader[1])
        # length = len(train_set_loader[0])
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps * int(length) , pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps * int(len(train_set_loader)) , pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

class Logger_train: 
    
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None 

    def _print_training_status(self):
        metrics_data = [ self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys()) ]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data) 

        # print the training status 
        print(training_str + metrics_str)

        if self.writer is None: 
            self.writer = SummaryWriter()

        for k in self.running_loss:
            # add_scalar(tag, scalar_value, step)
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ , self.total_steps)
            self.running_loss[k] = 0.0
    
    def push(self, metrics):
        self.total_steps += 1 

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            
            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {} # 100번 돌면 초기화 
    
    def write_dict(self, results):
        if self.writer is None: 
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def test(args): 
    ## Choose correct config file 
    # Choose correct Dataset(dsec vs. MVSEC) and initialization method (warm-start vs. standard)
    if args.dataset.lower()=='dsec':
        if args.type.lower()=='warm_start':
            config_path = 'config/dsec_warm_start.json'
        elif args.type.lower()=='standard':
            config_path = 'config/dsec_standard.json'
        else:
            raise Exception('Please provide a valid argument for --type. [warm_start/standard]')

    elif args.dataset.lower()=='mvsec':
        if args.frequency==20:
            #config_path = 'config/mvsec_20.json'
            config_path = 'config/indoor_flying2.json'
        elif args.frequency==45:
            config_path = 'config/mvsec_45.json'
        else:
            raise Exception('Please provide a valid argument for --frequency. [20/45]')
        if args.type=='standard':
            raise NotImplementedError('Sorry, this is not implemented yet, please choose --type warm_start')
    
    elif args.dataset.lower() == 'indoor_flying':
        config_path = 'config/indoor_flying_test.json'
    
    else:
        raise Exception('Please provide a valid argument for --dataset. [dsec/mvsec]')


    ## Load config file
    config = json.load(open(config_path))
    ## Create Save Folder (./saved/mvsec_20hz)
    save_path = helper.create_save_path(config['save_dir'].lower(), config['name'].lower())
    print('Storing output in folder {}'.format(save_path))

    # Copy config file to save dir
    # json.dump : save the python object(config) to json file
    json.dump(config, open(os.path.join(save_path, 'config.json'), 'w'),
              indent=4, sort_keys=False)
    logger = Logger(save_path)
    logger.initialize_file("test") # model : test or train 

    ## Data Load
    additional_loader_returns = None
    if args.dataset.lower() == 'dsec':
        loader = DatasetProvider(
            dataset_path=Path(args.path_test), 
            representation_type=RepresentationType.VOXEL, 
            delta_t_ms=100,
            config=config,
            type=config['subtype'].lower(),
            visualize=args.visualize)
        loader.summary(logger)
        test_set = loader.get_test_dataset()
        # get_name_mapping_test() : return self.name_mapper_test [test dataset의 name 을 list에 append 해둔 list]
        additional_loader_returns = {'name_mapping_test': loader.get_name_mapping_test()}
    elif args.dataset.lower() == 'mvsec':
        if config['subtype'].lower() == 'standard':
            test_set = MvsecFlow(
                args = config["data_loader"]["test"]["args"],
                type='test',
                path=args.path_test
            )
        elif config['subtype'].lower() == 'warm_start':
            test_set = MvsecFlowRecurrent(
                args = config["data_loader"]["test"]["args"],
                type='test',
                path=args.path_test, 
                aug = False
            )
        else:
            raise NotImplementedError 
        # summary : loader type, sequence length, step_size, framerate -> logger.write_line() // Just Write Log 
        test_set.summary(logger)
    elif args.dataset.lower() == 'indoor_flying':
        if config['subtype'].lower() == 'standard':
            test_set = IndoorFlow(
                args = config["data_loader"]["test"]["args"],
                type='test',
                path=args.path_test
            )
        elif config['subtype'].lower() == 'warm_start':
            test_set = IndoorFlowRecurrent(
                args = config["data_loader"]["test"]["args"],
                type='test',
                path=args.path_test, 
                aug = False
            )

    # Instantiate Dataloader
    test_set_loader = DataLoader(test_set,
                                 batch_size=config['data_loader']['test']['args']['batch_size'],
                                 shuffle=config['data_loader']['test']['args']['shuffle'],
                                 num_workers=args.num_workers,
                                 drop_last=True)

    # Load Model
    model = eraft.ERAFT(
        config=config, 
        n_first_channels=config['data_loader']['test']['args']['num_voxel_bins'] 
    )
    
    # Load Checkpoint
    checkpoint = torch.load(config['test']['checkpoint'])
    checkpoint_name = config['test']['checkpoint']
    file_extension = os.path.splitext(checkpoint_name)[1]
    if file_extension == '.pth':
        ## For checkpoint (.pth)
        model.load_state_dict(checkpoint['model'])
    elif file_extension == '.tar':
        ## For checkpoint (.tar)
        model.load_state_dict(checkpoint['model'])
    else:
        assert (file_extension == '.pth' or file_extension == '.tar'), "Wrong Checkpoint!"

    # Get Visualizer
    visualizer = get_visualizer(args)

    # Initialize Tester (e.g TestRaftEventsWarm )
    test = initialize_tester(config)

    test = test(
        model=model,
        config=config,
        data_loader=test_set_loader,
        test_logger=logger,
        save_path=save_path,
        visualizer=visualizer,
        additional_args=additional_loader_returns
    )
    
    # summary : logger.write_line() 
    test.summary()
    
    # .test() : return 'log' that contains information about validation 
    file_path = config['test']['checkpoint']
    split_file_path = file_path.split('/')
    if args.dataset == 'dsec':
         ## DSEC Test (dsec test는 코드에서 GT와 비교하지않음. 저자의 benchmark page에 업로드하면 수치도출되는 형태)
        test._test_dsec()
    elif (args.dataset == 'mvsec') or (args.dataset == 'indoor_flying'):
        ## MVSEC Test (GT를 가지고있기때문에 직접 Metric을 추출하는 형태)
        file_name = split_file_path[1]
        test._test_mvsec(file_name)


def train(args):    
    
    ## For resume 
    checkpoint = None
    # optionally resume from a checkpoint
    if args.resume:  
        args_new = args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint ‘{}’ ... ".format(args.resume), end='')
            checkpoint = torch.load(args.resume, map_location='cuda')
            # args.start_epoch = 120
    
    # choose correct config file & Load json file 
    if args.dataset.lower() == 'mvsec':
        if args.frequency == 20:
            config_path_1 = 'config/mvsec_train_20_1.json' # sequence length : 2
            # config_path_2 = 'config/mvsec_train_20_2.json' # sequence length : 5
            # json load 
            config_1 = json.load(open(config_path_1))
            # config_2 = json.load(open(config_path_2))
            
        elif args.frequency == 45:
            raise NotImplementedError('Sorry, this is not implemented yet, please choose 20Hz!')
        if args.type == 'standard':
            raise NotImplementedError('Sorry, this is not implemented yet, please choose 20Hz!')
    
    elif args.dataset.lower() == 'dsec':
        if args.type.lower() == 'warm_start':
            config = 'config/dsec_warm_start_train.json'
            config = json.load(open(config))
        elif args.type.lower() == 'standard':
            config = 'config/dsec_standard_train.json'
            config = json.load(open(config))
        else:
            raise Exception('Please provide a valid argument for --type. [warm_start/standard]')
    
    elif args.dataset.lower() == 'indoor_flying':
        config_path_1 = 'config/indoor_flying1.json'
        config_path_2 = 'config/indoor_flying2.json'
        config_path_3 = 'config/indoor_flying3.json'
        #json load 
        config_1 = json.load(open(config_path_1))
        config_2 = json.load(open(config_path_2))
        config_3 = json.load(open(config_path_3))
    
    else:
        raise Exception('Please provide a valid argument for --dataset. [dsec/mvsec]')
    
    # # Choose validation config file 
    # config_path_valid = 'config/indoor_flying1.json'
    # config_valid = json.load(open(config_path_valid))

    # Data Load
    if args.dataset.lower() == 'dsec':
        if (config['subtype'].lower() == 'standard'):
            loader = DatasetProvider(
                dataset_path=Path(args.path_train), 
                representation_type=RepresentationType.VOXEL, 
                delta_t_ms=100,
                num_bins = config['data_loader']['train']['args']['num_voxel_bins'],
                config=config,
                type=config['subtype'].lower(),
                visualize=args.visualize, 
                mode = config['data_loader']['train']['args']['type'], 
                transforms = config['data_loader']['train']['args']['transforms'])
            
            train_set = loader.get_train_dataset()
            additional_loader_returns = {'name_mapping_train': loader.get_name_mapping_train()}
            train_set_loader = DataLoader(train_set,
                                 batch_size=config['data_loader']['train']['args']['batch_size'],
                                 shuffle=config['data_loader']['train']['args']['shuffle'],
                                 num_workers=args.num_workers,
                                 drop_last=True, 
                                 )
        elif (config['subtype'].lower() == 'warm_start'):
            loader = DatasetProvider(
                dataset_path=Path(args.path_train), 
                representation_type=RepresentationType.VOXEL, 
                delta_t_ms=100,
                num_bins = config['data_loader']['train']['args']['num_voxel_bins'],
                config=config,
                type=config['subtype'].lower(),
                visualize=args.visualize, 
                mode = config['data_loader']['train']['args']['type'], 
                transforms = config['data_loader']['train']['args']['transforms'])
            
            train_set = loader.get_train_dataset()
            additional_loader_returns = {'name_mapping_train': loader.get_name_mapping_train()}
            train_set_loader = DataLoader(train_set,
                                 batch_size=config['data_loader']['train']['args']['batch_size'],
                                 shuffle=config['data_loader']['train']['args']['shuffle'],
                                 num_workers=args.num_workers,
                                 drop_last=True
                                 )
    # case : mvsec dataset
    elif args.dataset.lower() == 'mvsec':
        if (config_1['subtype'].lower() == 'standard'):
            raise NotImplementedError('Sorry, this is not implemented yet, please choose 20Hz!')
        elif config_1['subtype'].lower() == 'warm_start':
            # Sequence Length : 2 
            train_set_1 = MvsecFlowRecurrent(
                args = config_1['data_loader']['train']['args'], 
                type = 'train', 
                path = args.path_train, 
                aug = True
            )
            # # Sequence Length : 5
            # train_set_2 = MvsecFlowRecurrent(
            #     args = config_2['data_loader']['train']['args'], 
            #     type = 'train', 
            #     path = args.path_train,
            #     aug = True
            # )
            train_set_loader_1 = DataLoader(train_set_1, 
                                    batch_size = config_1['data_loader']['train']['args']['batch_size'], 
                                    shuffle = config_1['data_loader']['train']['args']['shuffle'], 
                                    num_workers = args.num_workers, 
                                    drop_last = True
            )
            # train_set_loader_2 = DataLoader(train_set_2, 
            #                                 batch_size = config_2['data_loader']['train']['args']['batch_size'], 
            #                                 shuffle = config_2['data_loader']['train']['args']['shuffle'], 
            #                                 num_workers = args.num_workers, 
            #                                 drop_last = True
            # )
        else:
            raise NotImplementedError
    
    elif args.dataset.lower() == 'indoor_flying':
        if (config_3['subtype'].lower() == 'standard'):
            raise NotImplementedError('Sorry, this is not implemented yet')
        elif config_3['subtype'].lower() == 'warm_start':
            sub_path1 = 'indoor_flying1_Original/'
            sub_path2 = 'indoor_flying2_Original/'
            sub_path3 = 'indoor_flying3_Original/'
            
            # train_set_1 = IndoorFlowRecurrent(
            #     args = config_1['data_loader']['train']['args'], 
            #     type = 'train', 
            #     path = args.path_train + sub_path1, 
            #     aug = False
            # )
            train_set_2 = IndoorFlowRecurrent(
                args = config_2['data_loader']['train']['args'], 
                type = 'train', 
                path = args.path_train + sub_path2, 
                aug = True
            )
            train_set_3 = IndoorFlowRecurrent(
                args = config_3['data_loader']['train']['args'], 
                type = 'train', 
                path = args.path_train + sub_path3, 
                aug = True
            )
            # Train_set_loader Append 
            train_set_loader_list = list()
            # train_set_loader_1 = DataLoader(train_set_1, 
            #                         batch_size = config_1['data_loader']['train']['args']['batch_size'], 
            #                         shuffle = config_1['data_loader']['train']['args']['shuffle'], 
            #                         num_workers = args.num_workers, 
            #                         drop_last = True
            # )
            # train_set_loader_list.append(train_set_loader_1)
            train_set_loader_2 = DataLoader(train_set_2, 
                                            batch_size = config_2['data_loader']['train']['args']['batch_size'], 
                                            shuffle = config_2['data_loader']['train']['args']['shuffle'], 
                                            num_workers = args.num_workers, 
                                            drop_last = True
            )
            train_set_loader_list.append(train_set_loader_2)
            train_set_loader_3 = DataLoader(train_set_3, 
                                            batch_size = config_3['data_loader']['train']['args']['batch_size'], 
                                            shuffle = config_3['data_loader']['train']['args']['shuffle'], 
                                            num_workers = args.num_workers, 
                                            drop_last = True
            )
            train_set_loader_list.append(train_set_loader_3)
            
        else:
            raise NotImplementedError        
    
    # # Validation Data Load 
    # if (config_valid['subtype'].lower() == 'standard'):
    #     raise NotImplementedError('Sorry, this is not implemented yet')
    # elif config_valid['subtype'].lower() == 'warm_start':
    #     valid_set = MvsecFlowRecurrent(
    #         args = config_valid['data_loader']['test']['args'], 
    #         type = 'test', 
    #         path = args.path_valid
    #         )
    #     valid_set_loader = DataLoader(valid_set, 
    #                                 batch_size = config_valid['data_loader']['test']['args']['batch_size'], 
    #                                 shuffle = config_valid['data_loader']['test']['args']['shuffle'], 
    #                                 num_workers = args.num_workers, 
    #                                 drop_last = True
    #         )
    # else:
    #     raise NotImplementedError
    
    
    if args.dataset == 'mvsec':
        model = eraft.ERAFT(
            config = config_1, 
            n_first_channels = config_1['data_loader']['train']['args']['num_voxel_bins']
        )
        optimizer, scheduler = fetch_optimizer(args, model, config_1['data_loader']['train']['args']['batch_size'], train_set_loader_1)
    elif args.dataset == 'dsec':
        model = eraft.ERAFT(
            config = config, 
            n_first_channels= config['data_loader']['train']['args']['num_voxel_bins']
        )
        optimizer, scheduler = fetch_optimizer(args, model, config['data_loader']['train']['args']['batch_size'], train_set_loader)
    elif args.dataset == 'indoor_flying':
        model = eraft.ERAFT(
            config = config_3, 
            n_first_channels=config_3['data_loader']['train']['args']['num_voxel_bins']
        )
        optimizer, scheduler = fetch_optimizer(args, model, config_3['data_loader']['train']['args']['batch_size'], train_set_loader_list)
    
    # For resume
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        print("=> checkpoint state loaded.")
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> checkpoint optimizer state loaded.')
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        # scheduler = checkpoint['scheduler']
        # print('=> checkpoint scehduler state loaded.')
        # total_steps = checkpoint['epoch']
    else:
        total_steps = 0
    
    wandb.watch(model)
    print("Parameter Count: %d" % count_parameters(model))
    model.cuda()
    model.train()
    
    # AMP : loss scale 을 위한 GradScaler 생성 
    scaler = GradScaler(enabled = args.mixed_precision)
    logger = Logger_train(model, scheduler)
    should_keep_training = True
    
    # ## Validation 
    # valid_save_path = helper.create_save_path(config_valid['save_dir'].lower(), config_valid['name'].lower())
    # print('Storing output in folder {}'.format(valid_save_path))
    # # Get Visualizer
    # visualizer = get_visualizer(args)
    best_epe = 100
    while should_keep_training:
        if args.dataset == 'mvsec':
            # 0 ~ 10 epoch : sequence_length = 2 (mvsec 20hz)
            if total_steps <= args.num_steps:
                for i_batch, batch in enumerate(train_set_loader_1):
                    optimizer.zero_grad()
                    sequence_check = total_loss = total_epe = total_1pe = total_2pe = total_3pe = 0
                    ## sequence_length
                    for x in batch:
                        ## Original mvsec outdoor_day2 20Hz 
                        flow = x['flow'].cuda()
                        gt_valid_mask = x['gt_valid_mask'].cuda()
                        event_volume_new = x['event_volume_new'].cuda()
                        event_volume_old = x['event_volume_old'].cuda()
                        
                        if sequence_check == 0:
                            last_flow, flow_predictions = model(event_volume_old,event_volume_new, iters = args.iters, flow_init = None)
                        else:
                            # For warm-start 
                            last_flow, flow_predictions = model(event_volume_old, event_volume_new, iters = args.iters, flow_init = last_flow)
                            last_flow = image_utils.forward_interpolate_pytorch(last_flow)
                        
                        loss, metrics = sequence_loss(flow_predictions, flow, gt_valid_mask, args.gamma)
                        total_loss += loss
                        total_epe += metrics['epe']
                        total_1pe += metrics['1pe']
                        total_2pe += metrics['2pe']
                        total_3pe += metrics['3pe']
                        sequence_check += 1
                 
                    # scaled loss를 이용해 backward 진행 ( gradient 모두 같은 scale factor로 scale됨)
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    # weight update 
                    scaler.step(optimizer)
                    # scaler factor update 
                    scaler.update()
                    logger.push(metrics)
                    # gpu memory 
                    torch.cuda.empty_cache()
                        
                    # Check epe (sequence_length의 마지막 metric result)
                    # train_epe = metrics['epe']
                    # train_1pe = metrics['1pe']
                    # train_2pe = metrics['2pe']
                    # train_3pe = metrics['3pe']
                    ## Check metrics ( sequence length 전체에 대한 Average value )
                    train_epe = total_epe / sequence_check
                    train_1pe = total_1pe / sequence_check
                    train_2pe = total_2pe / sequence_check
                    train_3pe = total_3pe / sequence_check
                    print("Epoch : %f, Loss : %f, Training EPE: %f, 1pe: %f, 2pe: %f, 3pe: %f" % (total_steps+1, total_loss, train_epe, train_1pe, train_2pe, train_3pe))
                    
                    #wandb
                    wandb.log({"train_loss": total_loss})  
                    wandb.log({"train_epe": train_epe})
                    wandb.log({"train_1pe": train_1pe})
                    wandb.log({"train_2pe": train_2pe})     
                    wandb.log({"train_3pe": train_3pe})                   
            
            # 10 epoch ~ : sequence_length = 5 (mvsec 20hz) 
            # elif total_steps > 10:
            #     for i_batch, batch in enumerate(train_set_loader_2):
            #         optimizer.zero_grad()
            #         sequence_check = total_loss = total_epe = total_1pe = total_2pe = total_3pe = 0
            #         ## sequence_length
            #         for x in batch:
            #             ## Original mvsec outdoor_day2 20Hz 
            #             flow = x['flow'].cuda()
            #             ## Only in 정인 Dataset
            #             #flow = -x['flow'].cuda()
            #             gt_valid_mask = x['gt_valid_mask'].cuda() 
            #             event_volume_new = x['event_volume_new'].cuda()
            #             event_volume_old = x['event_volume_old'].cuda()
                        
            #             if sequence_check == 0:
            #                 last_flow, flow_predictions = model(event_volume_old,event_volume_new, iters = args.iters, flow_init = None)
            #             else:
            #                 # For warm-start 
            #                 last_flow, flow_predictions = model(event_volume_old, event_volume_new, iters = args.iters, flow_init = last_flow)
            #                 last_flow = image_utils.forward_interpolate_pytorch(last_flow)
                        
            #             loss, metrics = sequence_loss(flow_predictions, flow, gt_valid_mask, args.gamma)
                        
            #             total_loss += loss
            #             total_epe += metrics['epe']
            #             total_1pe += metrics['1pe']
            #             total_2pe += metrics['2pe']
            #             total_3pe += metrics['3pe']
            #             sequence_check += 1
                    
            #         scaler.scale(total_loss).backward()
            #         scaler.unscale_(optimizer)
            #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            #         scaler.step(optimizer)
            #         scaler.update()
            #         logger.push(metrics)
            #         # gpu memory 
            #         torch.cuda.empty_cache()
                        
            #         ## Check metrics ( sequence length 전체에 대한 Average value )
            #         train_epe = total_epe / sequence_check
            #         train_1pe = total_1pe / sequence_check
            #         train_2pe = total_2pe / sequence_check
            #         train_3pe = total_3pe / sequence_check
            #         print("Epoch : %f, Loss : %f, Training EPE: %f, 1pe: %f, 2pe: %f, 3pe: %f" % (total_steps+1, total_loss, train_epe, train_1pe, train_2pe, train_3pe))

            #         #wandb
            #         wandb.log({"train_loss": total_loss})  
            #         wandb.log({"train_epe": train_epe})
            #         wandb.log({"train_1pe": train_1pe})
            #         wandb.log({"train_2pe": train_2pe})     
            #         wandb.log({"train_3pe": train_3pe})     
        elif (args.dataset == 'dsec'):
            for i_batch, batch in enumerate(train_set_loader):
                optimizer.zero_grad()
                sequence_check = total_loss = total_epe = total_1pe = total_2pe = total_3pe = 0
                ## for sequence length 
                for x in batch:
                    flow = x['flow'].cuda() 
                    gt_valid_mask = x['gt_valid_mask'].cuda() 
                    event_volume_new = x['event_volume_new'].cuda()
                    event_volume_old = x['event_volume_old'].cuda()
                    
                    if sequence_check == 0:
                            last_flow, flow_predictions = model(event_volume_old,event_volume_new, iters = args.iters, flow_init = None)
                            
                    else:
                        # For warm-start 
                        last_flow, flow_predictions = model(event_volume_old, event_volume_new, iters = args.iters, flow_init = last_flow)
                        last_flow = image_utils.forward_interpolate_pytorch(last_flow)
                    
                    loss, metrics = sequence_loss(flow_predictions, flow, gt_valid_mask, args.gamma)
                    total_loss += loss
                    total_epe += metrics['epe']
                    total_1pe += metrics['1pe']
                    total_2pe += metrics['2pe']
                    total_3pe += metrics['3pe']
                    sequence_check += 1
                
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
                logger.push(metrics)
                    
                ## Check metrics ( sequence length 전체에 대한 Average value )
                train_epe = total_epe / sequence_check
                train_1pe = total_1pe / sequence_check
                train_2pe = total_2pe / sequence_check
                train_3pe = total_3pe / sequence_check
                print("Epoch : %f, Loss : %f, Training EPE: %f, 1pe: %f, 2pe: %f, 3pe: %f" % (total_steps+1, total_loss, train_epe, train_1pe, train_2pe, train_3pe))

                #wandb
                wandb.log({"train_loss": total_loss})  
                wandb.log({"train_epe": train_epe})
                wandb.log({"train_1pe": train_1pe})
                wandb.log({"train_2pe": train_2pe})     
                wandb.log({"train_3pe": train_3pe})    
        elif (args.dataset == 'indoor_flying'):
            for train_set_loader in train_set_loader_list:
                for i_batch, batch in enumerate(train_set_loader):
                    optimizer.zero_grad()
                    sequence_check = total_loss = total_epe = total_1pe = total_2pe = total_3pe = 0
                    ## sequence_length
                    for x in batch:
                        flow = x['flow'].cuda()
                        gt_valid_mask = x['gt_valid_mask'].cuda()
                        event_volume_new = x['event_volume_new'].cuda()
                        event_volume_old = x['event_volume_old'].cuda()
                        
                        if sequence_check == 0:
                            last_flow, flow_predictions = model(event_volume_old,event_volume_new, iters = args.iters, flow_init = None)
                            last_flow = image_utils.forward_interpolate_pytorch(last_flow)
                        else:
                            # For warm-start 
                            last_flow, flow_predictions = model(event_volume_old, event_volume_new, iters = args.iters, flow_init = last_flow)
                            last_flow = image_utils.forward_interpolate_pytorch(last_flow)
                        
                        loss, metrics = sequence_loss(flow_predictions, flow, gt_valid_mask, args.gamma)
                        total_loss += loss
                        total_epe += metrics['epe']
                        total_1pe += metrics['1pe']
                        total_2pe += metrics['2pe']
                        total_3pe += metrics['3pe']
                        sequence_check += 1
                
                    # scaled loss를 이용해 backward 진행 ( gradient 모두 같은 scale factor로 scale됨)
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    # weight update 
                    scaler.step(optimizer)
                    # scaler factor update 
                    scaler.update()
                    logger.push(metrics)
                    # gpu memory 
                    torch.cuda.empty_cache()

                    ## Check metrics ( sequence length 전체에 대한 Average value )
                    train_epe = total_epe / sequence_check
                    train_1pe = total_1pe / sequence_check
                    train_2pe = total_2pe / sequence_check
                    train_3pe = total_3pe / sequence_check
                    print("Epoch : %f, Loss : %f, Training EPE: %f, 1pe: %f, 2pe: %f, 3pe: %f" % (total_steps+1, total_loss, train_epe, train_1pe, train_2pe, train_3pe))
                    
                    #wandb
                    wandb.log({"train_loss": total_loss})  
                    wandb.log({"train_epe": train_epe})
                    wandb.log({"train_1pe": train_1pe})
                    wandb.log({"train_2pe": train_2pe})     
                    wandb.log({"train_3pe": train_3pe})    
        
        # Every one epoch -> lr scheduler Update & Epoch update 
        scheduler.step()
        # Epoch 별 lr 값 확인용 
        # print("lr: ", optimizer.param_groups[0]['lr'])
        total_steps += 1
        
        # EPE Training value 를 기준으로 checkpoint 추출  
        if train_epe < best_epe:
            best_epe = train_epe
            PATH = 'checkpoints/EPE_%s_%s.pth' %(args.name, wandb.run.name)
            torch.save({
                'epoch' : total_steps,
                'model' : model.state_dict(), 
                'optimizer' : optimizer.state_dict(), 
                'scheduler' : scheduler, 
            }, PATH)
        
        # 20 epoch 마다 checkpoint 저장 (For test)
        # if (total_steps+1) % 20 == 0:
        #     PATH2 = 'checkpoints/indoor_%s_%s_epoch.pth' %(str(total_steps), wandb.run.name)
        #     torch.save({
        #         'epoch' : total_steps,
        #         'model' : model.state_dict(), 
        #         'optimizer' : optimizer.state_dict(), 
        #         'scheduler' : scheduler, 
        #     }, PATH2)
        #     txt = './saved/indoor_%s_%s_epoch.txt' %(str(total_steps), wandb.run.name)
        #     f = open(txt, 'w')
        #     f.write("average_loss : %.7f\n" %(total_loss))
        #     f.write("last_epe_avg : %.7f\n" %train_epe)
        #     f.write("last_1pe_avg : %.7f\n" %train_1pe)
        #     f.write("last_2pe_avg : %.7f\n" %train_2pe)
        #     f.write("last_3pe_avg : %.7f\n" %train_3pe)
        #     f.close()
        
        # Finish Training 
        if (total_steps+1) >= args.num_steps:
            should_keep_training = False
            last_txt = './saved/%s_train.txt' %(wandb.run.name)
            f = open(last_txt, 'w')
            f.write("average_loss : %.7f\n" %(total_loss))
            f.write("last_epe_avg : %.7f\n" %train_epe)
            f.write("last_1pe_avg : %.7f\n" %train_1pe)
            f.write("last_2pe_avg : %.7f\n" %train_2pe)
            f.write("last_3pe_avg : %.7f\n" %train_3pe)
            f.close()
            break

    logger.close()
    
    return PATH


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_test', default='train', type=str, help='Choose the type of train or test')
    parser.add_argument('--dataset', default='indoor_flying', type=str, help='which dataset to use: ([dsec]/mvsec/indoor_flying)')
    parser.add_argument('--frequency', default=20, type=int, help="Evaluation frequency of MVSEC dataset ([20]/45) Hz")
    parser.add_argument('--type', default='warm_start', type=str, help="Evaluation type ([warm_start]/standard)")
    parser.add_argument('--visualize', action='store_true', help='Provide this argument s.t. DSEC results are visualized. MVSEC experiments are always visualized.')
    parser.add_argument('--num_workers', default=4, type=int, help='How many sub-processes to use for data loading')
    '''
    Test 
    - mvsec 20hz (Original) = '/share/data/mvsec_outdoor_day_1_20Hz'
    - indoor_flying1 (Jeongin) = '/share/data/indoor_flying2'
    - dsec = '/share/data/test'
    '''
    parser.add_argument('--path_test', default = '/share/data/indoor_flying_test/indoor_flying1_Original_total/indoor_flying1_Original', type = str, help='Dataset Path for Testing')
    '''
    Training 
    - MVSEC 20hz (Original) = '/share/data/mvsec_outdoor_day_2_20Hz'
    - MVSEC (Jeongin) = '/share/data/mvsec_outdoor_day_2'
    - Indoor (Original) = '/share/data/indoor_flying_train/'
    - DSEC 20hz (Harin) = '/share/data/train'
    '''
    parser.add_argument('--path_train', default = '/share/data/indoor_flying_train/', type = str, help='Dataset Path for Training ')
    parser.add_argument('--path_valid', default = '/share/data/indoor_flying1/indoor_flying1_Original', type = str, help='Dataset Path for Validation')
    parser.add_argument('--name', default='eraft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_steps', type=int, default=500) # epoch
    # parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    # parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12) # GRU Iteration 
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    '''
    For resume the training 
    - Yes = './checkpoints/eraft_09-27-06:10.pth' ( e.g )
    - No = None
    '''
    parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint(default:None)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (userful on restarts)')
    args = parser.parse_args()
    
    # wandb
    wandb.init(project="eraft", entity="harinpk", reinit=True)
    wandb.run.name = datetime.datetime.now().strftime("%m-%d-%H:%M")
    wandb.config.update(args)
    wandb.run.save()
    
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    if args.train_test == 'test':
        # Run Test Script
        test(args)
    elif args.train_test == 'train':
        # Run Train Script 
        torch.manual_seed(1234)
        np.random.seed(1234)
        train(args)
