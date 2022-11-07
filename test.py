from cgi import test
import numpy as np
import torch as th
from torchvision import utils
from utils.helper_functions import *
import utils.visualization as visualization
import utils.filename_templates as TEMPLATES
import utils.helper_functions as helper
import utils.logger as logger
from utils import image_utils
import torch.nn as nn
import wandb

MAX_FLOW = 400
SUM_FREQ = 100

class Test(object):
    """
    Test class

    """

    def __init__(self, model, config,
                 data_loader, visualizer, test_logger=None, save_path=None, additional_args=None):
        self.downsample = False # Downsampling for Rebuttal
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.additional_args = additional_args
        if config['cuda'] and not torch.cuda.is_available():
            print('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        else:
            self.gpu = torch.device('cuda:' + str(config['gpu']))
            self.model = self.model.to(self.gpu)
        if save_path is None:
            self.save_path = helper.create_save_path(config['save_dir'].lower(),
                                           config['name'].lower())
        else:
            self.save_path=save_path
        if logger is None:
            self.logger = logger.Logger(self.save_path)
        else:
            self.logger = test_logger
        if isinstance(self.additional_args, dict) and 'name_mapping_test' in self.additional_args.keys():
            visu_add_args = {'name_mapping' : self.additional_args['name_mapping_test']}
        else:
            visu_add_args = None
        self.visualizer = visualizer(data_loader, self.save_path, additional_args=visu_add_args)

    def summary(self):
        self.logger.write_line("====================================== TEST SUMMARY ======================================", True)
        self.logger.write_line("Model:\t\t\t" + self.model.__class__.__name__, True)
        self.logger.write_line("Tester:\t\t" + self.__class__.__name__, True)
        self.logger.write_line("Test Set:\t" + self.data_loader.dataset.__class__.__name__, True)
        self.logger.write_line("\t-Dataset length:\t"+str(len(self.data_loader)), True)
        self.logger.write_line("\t-Batch size:\t\t" + str(self.data_loader.batch_size), True)
        self.logger.write_line("==========================================================================================", True)

    def run_network(self, epoch):
        raise NotImplementedError

    def move_batch_to_cuda(self, batch):
        raise NotImplementedError

    def visualize_sample(self, batch):
        self.visualizer(batch)

    def visualize_sample_dsec(self, batch, batch_idx):
        self.visualizer(batch, batch_idx, None)

    def get_estimation_and_target(self, batch):
        # Returns the estimation and target of the current batch
        raise NotImplementedError
    
    def sequence_loss(self, flow_preds, flow_gt, valid, gamma = 0.8, max_flow = MAX_FLOW): 
        """ Loss function defined over sequence of flow predictions """
        # # exclude invalid pixels and extremely large displacements ( Only in dsec dataset )
        # mag = torch.sum(flow_gt**2, dim=1).sqrt() # (1, 256, 256)
        # inner = mag < max_flow # (1, 256, 256)
        # # 1바퀴만 돈다. 
        # for i in range(inner.shape[0]): 
        #     # inner=(1, 256, 256)
        #     # inner_mask = (256, 256)
        #     inner_mask = inner[i]
        #     # inner_mask = (2, 256, 256)
        #     inner_mask = torch.stack([inner_mask, inner_mask], dim=0)
        #     valid[i] = (valid[i] == True) & (inner_mask == True)
        
        ## Loss 
        # n_predictions = len(flow_preds)
        
        valid = torch.Tensor.bool(valid)
        n_predictions = 1
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i)
            criterion = nn.L1Loss()
            # Compute all batch once  
            i_flow_preds = flow_preds.view(-1)[valid.view(-1)]
            i_flow_gt = flow_gt.view(-1)[valid.view(-1)]
            i_loss = criterion(i_flow_preds, i_flow_gt)
            flow_loss += i_weight * i_loss
        
        # # Average by batch_size
        # flow_loss = flow_loss / (flow_preds.size(0))
        
        ## metric using last flow
        epe = torch.sum((flow_preds - flow_gt)**2, dim=1).sqrt()
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

    def _test_dsec(self):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                # Move Data to GPU
                if next(self.model.parameters()).is_cuda:
                    batch = self.move_batch_to_cuda(batch)
                # Network Forward Pass
                self.run_network(batch)
                print("Sample {}/{}".format(batch_idx + 1, len(self.data_loader)))

                # Visualize
                if hasattr(batch, 'keys') and 'loader_idx' in batch.keys() \
                        or (isinstance(batch,list) and hasattr(batch[0], 'keys') and 'loader_idx' in batch[0].keys()):
                    self.visualize_sample(batch)
                else:
                    # DSEC Special Snowflake
                    self.visualize_sample_dsec(batch, batch_idx)
                    #print('Not Visualizing')

        # Log Generation
        log = {}

        return log
    
    def _test_mvsec(self, run_name):
        self.model.eval()
        with torch.no_grad():
            test_result = './saved/%s_test_log.txt' %(run_name)
            f = open(test_result, 'w')
            total_epe = 0
            total_1pe = 0
            total_2pe = 0
            total_3pe = 0
            count = 0
            for batch_idx, batch in enumerate(self.data_loader):
                # Move Data to GPU
                if next(self.model.parameters()).is_cuda:
                    batch = self.move_batch_to_cuda(batch)
                # Network Forward Pass
                self.run_network(batch)
                print("Sample {}/{}".format(batch_idx + 1, len(self.data_loader)))
                
                ## Only Indoor_flying (정인)
                #batch[0]['flow'] = -batch[0]['flow']
                
                ## For compute metric 
                flow = batch[0]['flow']
                gt_valid_mask = batch[0]['gt_valid_mask']
                pred_flow = batch[0]['flow_est']
                # flow = batch['flow']
                # gt_valid_mask = batch['gt_valid_mask']
                # pred_flow = batch['flow_est']
                loss, metrics = self.sequence_loss(pred_flow, flow, gt_valid_mask)
                test_epe = metrics['epe']
                test_1pe = metrics['1pe']
                test_2pe = metrics['2pe']
                test_3pe = metrics['3pe']
                
                print("Sample {}/{}, loss : {}, epe : {}, 1pe : {}, 2pe : {}, 3pe : {}"
                      .format(batch_idx + 1, len(self.data_loader), loss, test_epe, test_1pe, test_2pe, test_3pe))
                
                f.write('Sample {%f}/{%f} \n' %(batch_idx + 1, len(self.data_loader)))
                f.write("epe : %.7f\n" %(test_epe))
                f.write("1pe : %.7f\n" %(test_1pe))
                f.write("2pe : %.7f\n" %(test_2pe))
                f.write("3pe : %.7f\n" %(test_3pe))
                
                # For computing average value 
                total_epe += test_epe 
                total_1pe += test_1pe
                total_2pe += test_2pe
                total_3pe += test_3pe
                count += 1
                
                # Visualize
                if hasattr(batch, 'keys') and 'loader_idx' in batch.keys() \
                        or (isinstance(batch,list) and hasattr(batch[0], 'keys') and 'loader_idx' in batch[0].keys()):
                    self.visualize_sample(batch)
                else:
                    # DSEC Special Snowflake
                    self.visualize_sample_dsec(batch, batch_idx)
                    #print('Not Visualizing')
            
            avg_epe = total_epe / count 
            avg_1pe = total_1pe / count 
            avg_2pe = total_2pe / count 
            avg_3pe = total_3pe / count
            f.write('=================================')
            f.write('Test EPE Average : %.7f, Test 1PE Average : %.7f, Test 2PE Average : %.7f, Test 3PE Average : %.7f' %(avg_epe, avg_1pe, avg_2pe, avg_3pe))
            f.close()
        # Log Generation
        log = {}

        return log
    
    def _valid(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_epe = 0
            total_1pe = 0
            total_2pe = 0
            total_3pe = 0
            count = 0
            for batch_idx, batch in enumerate(self.data_loader):
                # Move Data to GPU
                if next(self.model.parameters()).is_cuda:
                    batch = self.move_batch_to_cuda(batch)
                # Network Forward Pass
                self.run_network(batch)
                print("Validation Sample {}/{}".format(batch_idx + 1, len(self.data_loader)))
                
                ## For compute metric 
                flow = batch[0]['flow']
                gt_valid_mask = batch[0]['gt_valid_mask']
                pred_flow = batch[0]['flow_est']
                loss, metrics = self.sequence_loss(pred_flow, flow, gt_valid_mask)
                test_epe = metrics['epe']
                test_1pe = metrics['1px']
                test_2pe = metrics['2px']
                test_3pe = metrics['3px']
                
                print("Validation Sample {}/{}, loss : {}, epe : {}, 1pe : {}, 2pe : {}, 3pe : {}"
                      .format(batch_idx + 1, len(self.data_loader), loss, test_epe, test_1pe, test_2pe, test_3pe))
                
                # For computing average value 
                total_loss += loss
                total_epe += test_epe 
                total_1pe += test_1pe
                total_2pe += test_2pe
                total_3pe += test_3pe
                count += 1
                
                #wandb
                wandb.log({"valid_loss": loss})  
                wandb.log({"valid_epe": test_epe})
                wandb.log({"valid_1pe": test_1pe})
                wandb.log({"valid_2pe": test_2pe})     
                wandb.log({"valid_3pe": test_3pe})
            
            avg_epe = total_epe / count 
            avg_1pe = total_1pe / count 
            avg_2pe = total_2pe / count 
            avg_3pe = total_3pe / count
            result_dict = {'avg_epe' : avg_epe, 'avg_1pe' : avg_1pe, 'avg_2pe' : avg_2pe, 'avg_3pe' : avg_3pe}

        return result_dict

class TestRaftEvents(Test):
    def move_batch_to_cuda(self, batch):
        return move_dict_to_cuda(batch, self.gpu)    
    
    def get_estimation_and_target(self, batch):
        if not self.downsample:
            if 'gt_valid_mask' in batch.keys():
                return batch['flow_est'].cpu().data, (batch['flow'].cpu().data, batch['gt_valid_mask'].cpu().data)
            return batch['flow_est'].cpu().data, batch['flow'].cpu().data
        else:
            f_est = batch['flow_est'].cpu().data
            f_gt = torch.nn.functional.interpolate(batch['flow'].cpu().data, scale_factor=0.5)
            if 'gt_valid_mask' in batch.keys():
                f_mask = torch.nn.functional.interpolate(batch['gt_valid_mask'].cpu().data, scale_factor=0.5)
                return f_est, (f_gt, f_mask)
            return f_est, f_gt

    def run_network(self, batch):
        # RAFT just expects two images as input. cleanest. code. ever.
        if not self.downsample:
                im1 = batch['event_volume_old']
                im2 = batch['event_volume_new']
        else:
            im1 = torch.nn.functional.interpolate(batch['event_volume_old'], scale_factor=0.5)
            im2 = torch.nn.functional.interpolate(batch['event_volume_new'], scale_factor=0.5)
        _, batch['flow_list'] = self.model(image1=im1,
                                           image2=im2)
        batch['flow_est'] = batch['flow_list'][-1]

class TestRaftEventsWarm(Test):
    def __init__(self, model, config,
                 data_loader, visualizer, test_logger=None, save_path=None, additional_args=None):
        super(TestRaftEventsWarm, self).__init__(model, config,
                                                 data_loader, visualizer, test_logger, save_path,
                                                 additional_args=additional_args)
        self.subtype = config['subtype'].lower()
        print('Tester Subtype: {}'.format(self.subtype))
        self.net_init = None # Hidden state of the refinement GRU
        self.flow_init = None
        self.idx_prev = None
        self.init_print=False
        assert self.data_loader.batch_size == 1, 'Batch size for recurrent testing must be 1'

    def move_batch_to_cuda(self, batch):
        return move_list_to_cuda(batch, self.gpu)

    def get_estimation_and_target(self, batch):
        if not self.downsample:
            if 'gt_valid_mask' in batch[-1].keys():
                return batch[-1]['flow_est'].cpu().data, (batch[-1]['flow'].cpu().data, batch[-1]['gt_valid_mask'].cpu().data)
            return batch[-1]['flow_est'].cpu().data, batch[-1]['flow'].cpu().data
        else:
            f_est = batch[-1]['flow_est'].cpu().data
            f_gt = torch.nn.functional.interpolate(batch[-1]['flow'].cpu().data, scale_factor=0.5)
            if 'gt_valid_mask' in batch[-1].keys():
                f_mask = torch.nn.functional.interpolate(batch[-1]['gt_valid_mask'].cpu().data, scale_factor=0.5)
                return f_est, (f_gt, f_mask)
            return f_est, f_gt

    def visualize_sample(self, batch):
        self.visualizer(batch[-1])

    def visualize_sample_dsec(self, batch, batch_idx):
        self.visualizer(batch[-1], batch_idx, None)

    def check_states(self, batch):
        # 0th case: there is a flag in the batch that tells us to reset the state (DSEC)
        if 'new_sequence' in batch[0].keys():
            if batch[0]['new_sequence'].item() == 1:
                self.flow_init = None
                self.net_init = None
                self.logger.write_line("Resetting States!", True)
        else:
            # During Validation, reset state if a new scene starts (index jump)
            if self.idx_prev is not None and batch[0]['idx'].item() - self.idx_prev != 1:
                self.flow_init = None
                self.net_init = None
                self.logger.write_line("Resetting States!", True)
            self.idx_prev = batch[0]['idx'].item()

    def run_network(self, batch):
        self.check_states(batch)
        for l in range(len(batch)):
            # Run Recurrent Network for this sample

            if not self.downsample:
                im1 = batch[l]['event_volume_old']
                im2 = batch[l]['event_volume_new']
            else:
                im1 = torch.nn.functional.interpolate(batch[l]['event_volume_old'], scale_factor=0.5)
                im2 = torch.nn.functional.interpolate(batch[l]['event_volume_new'], scale_factor=0.5)
            flow_low_res, batch[l]['flow_list'] = self.model(image1=im1,
                                                                image2=im2,
                                                                flow_init=self.flow_init)

        batch[l]['flow_est'] = batch[l]['flow_list'][-1] # flow_list[-1] : last prediction flow ( 12개 중 )
        
        self.flow_init = image_utils.forward_interpolate_pytorch(flow_low_res)
        batch[l]['flow_init'] = self.flow_init
