import h5py
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import os
import numpy
import sys
from torchvision import transforms
sys.path.append('utils')
from utils import filename_templates as TEMPLATES
from loader.utils import *
from utils.transformers import *
from utils import mvsec_utils

class MvsecFlow(Dataset):
    def __init__(self, args, type, path):
        super(MvsecFlow, self).__init__()
        #self.data_files = self.get_files(args['path'], args['datasets'])
        self.path_dataset = path
        self.timestamp_files = {}
        self.timestamp_files_flow = {}
        # If we load the image timestamps, we consider the framerate to be 45Hz.
        # Else if we load the depth/flow timestamps, the framerate is 20Hz.
        # The update rate gets set to 20 or 40 in the "get indices" method
        self.update_rate = None
        # get_indices : {dataset_name, subset_num, index, timestep} return 
        self.dataset = self.get_indices(path, args['datasets'], args['filter'], args['align_to'], args['type'])
        self.input_type = 'events'
        self.type = type # Train/Val/Test

        # Evaluation Type.  Dense  -> Valid where GT exists
        #                   Sparse -> Valid where GT & Events exist
        self.evaluation_type = 'sparse'

        self.image_width = 346
        self.image_height = 260

        self.voxel = EventSequenceToVoxelGrid_Pytorch(
            num_bins=args['num_voxel_bins'], 
            normalize=True, 
            gpu=True
        )
        self.cropper = transforms.CenterCrop((256,256))
        self.subset = self.dataset[0]['subset_number']

    def summary(self, logger):
        logger.write_line("================================== Dataloader Summary ====================================", True)
        logger.write_line("Loader Type:\t\t" + self.__class__.__name__ + " for {}".format(self.type), True)
        logger.write_line("Framerate:\t\t{}".format(self.update_rate), True)
        logger.write_line("Evaluation Type:\t{}".format(self.evaluation_type), True)

    def get_indices(self, path, dataset, filter, align_to, type):
        # Returns a list of dicts. Each dict contains the following items:
        #   ['dataset_name']    (e.g. outdoor_day)
        #   ['subset_number']   (e.g. 1)
        #   ['index']           (e.g. 1), Frame Index in the dataset
        #   ['timestamp']       Timestamp of the frame with index i
        samples = []
        for dataset_name in dataset:
            self.timestamp_files[dataset_name] = {}
            self.timestamp_files_flow[dataset_name] = {}
            for subset in dataset[dataset_name]:
                dataset_path = TEMPLATES.MVSEC_DATASET_FOLDER.format(dataset_name, subset)
                # Aligning (Image or Depth or Flow timestamp)
                if align_to.lower() == 'images' or align_to.lower() == 'image':
                    print("Aligning everything to the image timestamps!")
                    ts_path = TEMPLATES.MVSEC_TIMESTAMPS_PATH_IMAGES
                    if self.update_rate is not None and self.update_rate != 45:
                        raise Exception('Something wrong with the update rate!')
                    self.update_rate = 45
                    '''self.timestamp_files_flow[dataset_name][subset] = numpy.loadtxt(os.path.join(path,
                                                                                                 dataset_path,
                                                                                                 TEMPLATES.MVSEC_TIMESTAMPS_PATH_FLOW))'''
                elif align_to.lower() == 'depth':
                    print("Aligning everything to the depth timestamps!")
                    ts_path = TEMPLATES.MVSEC_TIMESTAMPS_PATH_DEPTH
                    if self.update_rate is not None and self.update_rate != 20:
                        raise Exception('Something wrong with the update rate!')
                    self.update_rate = 20
                elif align_to.lower() == 'flow':
                    print("Aligning everything to the flow timestamps!")
                    ts_path = TEMPLATES.MVSEC_TIMESTAMPS_PATH_FLOW
                    if self.update_rate is not None and self.update_rate != 20:
                        raise Exception('Something wrong with the update rate!')
                    self.update_rate = 20
                else:
                    raise ValueError("Please define the variable 'align_to' in the dataset [image/depth/flow]")
                ts = numpy.loadtxt(os.path.join(path, dataset_path, ts_path))
                self.timestamp_files[dataset_name][subset] = ts
                for idx in eval(filter[dataset_name][str(subset)]):
                    sample = {}
                    sample['dataset_name'] = dataset_name
                    sample['subset_number'] = subset
                    sample['index'] = idx
                    sample['timestamp'] = ts[idx]
                    sample['type'] = type
                    samples.append(sample)

        return samples

    def get_data_sample(self, loader_idx):
        # ================================= Get Data Sample =============================== #
        # Returns dict with the following content:                                          #
        #   - Event Sequence New (if type == 'events')                                      #
        #   - Event Sequence Old (if type == 'events')                                      #
        #   - Optical Flow (forward) between two timesteps as defined below                 #
        #   - Timestamp and some other params                                               #
        #                                                                                   #
        # Nomenclature Definition                                                           #
        #                                                                                   #
        # NOTE THAT THIS IS A DIFFERENT NAMING SCHEME THAN IN THE OTHER DATASETS!           #
        #                                                                                   #
        # Flow[i-1]   Flow[i]     Flow[i+1]   Flow[i+2]                                     #
        # Depth[i-1]  Depth[i]    Depth[i+1]  Depth[i+2]
        # |       .   |  .        |   .     . |                                             #
        # |  .     .  |      .    |  .   .    |                                             #
        # |    .      | ..     .  |  ...      |                                             #
        # | .         |     .     |  .    .   |                                             #
        # Events[i]   Events[i+1] Events[i+2]                                               #
        #
        # Flow[i] tells us the flow between Depth[i] and Depth[i+1]                         #
        # This can be seen because the pixels of flow[i] are the same as depth[i]           #
        # We are for now using the events aligned to the depth-timestamps.                  #
        # This means, to get the flow between Depth[i] and Depth[i+1], we need to load      #
        #   - Flow[i]
        #   - Events[i+1]
        #   - Events[i] (if using volumetric cost volumes)
        #   - Timestamps (from depth) [i]
        #   - Timestamps (from depth) [i+1]
        train_val_test_type = self.dataset[0]['type']
        set = self.dataset[loader_idx]['dataset_name']
        subset = self.dataset[loader_idx]['subset_number']
        path_subset = TEMPLATES.MVSEC_DATASET_FOLDER.format(set, subset)
        path_dataset = os.path.join(self.path_dataset, path_subset)
        # params = self.config[self.dataset[loader_idx]['dataset_name']][self.dataset[loader_idx]['subset_number']]
        idx = self.dataset[loader_idx]['index']
        type = self.input_type

        # If the update rate is 20 Hz (i.e. we're aligned to the depth/flow maps), we can directly take the flow gt
        # timestamp_files 내에 있는 timestamp(flow_ts) 를 ts_old / ts_new 에 넣음 
        # idx : filter 범위에 있는 num (e.g 4356)
        ts_old = self.timestamp_files[set][subset][idx]
        ts_new = self.timestamp_files[set][subset][idx + 1]

        if self.update_rate == 20:
            # Test
            if self.dataset[0]['type'] == 'test':
                ## "optical_flow/{:06d}.npy"
                flow = get_flow_npy(os.path.join(path_dataset,TEMPLATES.MVSEC_FLOW_GT_FILE.format(idx+1)), train_val_test_type)
            # Train & Valid
            elif (self.dataset[0]['type'] == 'train') | (self.dataset[0]['type'] == 'valid'):
                flow_path = os.path.join(path_dataset, 'optical_flow')
                flow_list = os.listdir(flow_path)
                flow_int_list = []
                for i in range(len(flow_list)):
                    flo = flow_list[i]
                    flow_int_list.append(int(flo[:-4]))
                flow_int_list = sorted(flow_int_list)
                # event (2) + flow (1-later flow 사용)
                file_name = str(flow_int_list[idx+1]) + '.npy'
                flow = get_flow_npy(os.path.join(path_dataset,'optical_flow/', file_name ), train_val_test_type)
                
        # Else, we need to interpolate the flow
        elif self.update_rate == 45:
            flow = self.estimate_gt_flow(loader_idx, ts_old, ts_new)
        else:
            raise NotImplementedError


        # Either flow_x or flow_y has to be != 0 s.t. the flow is valid
        flow_valid = (flow[0]!=0) | (flow[1] != 0)
        # Additionally, the car hood (that goes from row 193..260 is not included in the GT. so this is invalid too.
        # 아래부분 car hood로 인해 False 처리 
        flow_valid[193:,:]=False

        return_dict = {'idx': idx, 
                       'loader_idx': loader_idx, 
                       'flow': torch.from_numpy(flow),
                       'gt_valid_mask': torch.from_numpy(numpy.stack([flow_valid]*2, axis=0)),
                       "param_evc": {'height': self.image_height,
                                     'width': self.image_width}
                       }

        # Load Events
        if type == 'events':
            # Test 
            if self.dataset[0]['type'] == 'test':
                event_path_old = os.path.join(path_dataset, TEMPLATES.MVSEC_EVENTS_FILE.format('left', idx))
                event_path_new = os.path.join(path_dataset, TEMPLATES.MVSEC_EVENTS_FILE.format('left', idx+1))
            # Train & Valid
            elif (self.dataset[0]['type'] == 'train') | (self.dataset[0]['type'] == 'valid'):
                event_path = os.path.join(path_dataset, 'davis/left/events')
                event_list = os.listdir(event_path)
                event_int_list = []
                for i in range(len(event_list)):
                    event = event_list[i]
                    event_int_list.append(int(event[:-4]))
                event_int_list = sorted(event_int_list)
                file_name_1 = str(event_int_list[idx]) + '.npy'
                file_name_2 = str(event_int_list[idx+1]) + '.npy'
                event_path_old = os.path.join(path_dataset,'davis/left/events', file_name_1)
                event_path_new = os.path.join(path_dataset,'davis/left/events', file_name_2)
                
            params = {'height': self.image_height, 'width': self.image_width}

            events_old = get_events(event_path_old, train_val_test_type)
            events_new = get_events(event_path_new, train_val_test_type)

            # Timestamp multiplier of 1e6 because the timestamps are saved as seconds and we're used to microseconds
            # This can be relevant for the voxel grid!
            ev_seq_old = EventSequence(events_old, params, timestamp_multiplier=1e6, convert_to_relative=True)
            ev_seq_new = EventSequence(events_new, params, timestamp_multiplier=1e6, convert_to_relative=True)
            return_dict['event_volume_new'] = self.voxel(ev_seq_new)
            return_dict['event_volume_old'] = self.voxel(ev_seq_old)
            if self.evaluation_type == 'sparse':
                seq = ev_seq_new.get_sequence_only()
                h = self.image_height
                w = self.image_width
                hist, _, _ = numpy.histogram2d(x=seq[:,1], y=seq[:,2],
                                         bins=(w,h),
                                         range=[[0,w], [0,h]])
                hist = hist.transpose()
                ev_mask = hist > 0
                # flow_valid & ev_mask : 모두 True 인 것들에 대해서 동일한 matrix 2개 stacking 
                # ( 2, 260, 346 )
                return_dict['gt_valid_mask'] = torch.from_numpy(numpy.stack([flow_valid & ev_mask]*2, axis=0))
        elif type == 'frames':
            raise NotImplementedError
        else:
            raise Exception("Input Type not defined properly! Check config file.")

        # Check Timestamps
        ev = get_events(event_path_new, train_val_test_type).to_numpy()
        ts_ev_min = numpy.min(ev[:,0])
        ts_ev_max = numpy.max(ev[:,0])
        assert(ts_ev_min >= ts_old and ts_ev_max <= ts_new)

        # plot images
        '''
        from utils import visualization as visu
        from matplotlib import pyplot as plt
        
        # Justifying my choice of alignment:
        # 1) Flow[i] corresponds to Depth[i]
        depth_i = torch.tensor(numpy.load(os.path.join(path_dataset, TEMPLATES.MVSEC_DEPTH_GT_FILE.format(idx))))
        plt.figure("depth i")
        plt.imshow(depth_i.numpy())
               
        flow_visu = visu.visualize_optical_flow(flow, return_image=True)[0]
        plt.figure('Flow i')
        plt.imshow(flow_visu)
        
        # 2) The events are aligned to the depth 
        #       -> events[i] correspond to all events BEFORE depth i
        #       -> events[i+1] correspond to all events AFTER depth i
        # This can be proven by the timestamps timestamp[i] corresponding to depth[i]
        ts_old = self.timestamp_files[set][subset][idx]
        ts_new = self.timestamp_files[set][subset][idx + 1]
      
        
        ev = get_events(event_path_new).to_numpy()
        ts_ev_min = numpy.min(ev[:,0])
        ts_ev_max = numpy.max(ev[:,0])
        assert(ts_ev_min > ts_old and ts_ev_max <= ts_new)
        
        #       -> Additionally, we can show this, if we plot the events of the first 5ms of events before the depth map
        #          Remember: events[i] are all the events BEFORE the depth[i] 
        
        event_path_i = os.path.join(path_dataset, TEMPLATES.MVSEC_EVENTS_FILE.format('left', idx))
        ev_i = get_events(event_path_i).to_numpy()
        ts_i = self.timestamp_files[set][subset][idx]
        ev_inst_idx = ev_i[:,0] > ts_i - 0.005
        ev_inst = ev_i[ev_inst_idx]
        evv = visu.events_to_event_image(ev_inst, self.image_height, self.image_width)
        plt.figure("events_instantaneous")
        plt.imshow(evv.numpy().transpose(1,2,0))
        # This should now match the depth_i
        
        # Hence, all misalignments are coming from the ground-truth itself.
        '''
        #  < return_dict > 
        #   - Event Sequence New (if type == 'events')
        #   - Event Sequence Old (if type == 'events')                                      
        #   - Optical Flow (forward) between two timesteps as defined below                 
        #   - Timestamp and some other params(w, h)  
        return return_dict

    def estimate_gt_flow(self, loader_idx, ts_old, ts_new):
        # We need to estimate the flow between two timestamps.when the frame rate is 45Hz only.

        # First, get the dataset & subset
        train_val_test_type = self.dataset[0]['type']
        set = self.dataset[loader_idx]['dataset_name']
        subset = self.dataset[loader_idx]['subset_number']
        path_flow = os.path.join(self.path_dataset,
                                 TEMPLATES.MVSEC_DATASET_FOLDER.format(set, subset))

        assert ts_old >= self.timestamp_files_flow[set][subset].min(), \
            'Timestamp is smaller than the first flow timestamp'

        # Now, estimate the corresponding GT
        flow = mvsec_utils.estimate_corresponding_gt_flow(path_flow=path_flow,
                                                          gt_timestamps=self.timestamp_files_flow[set][subset],
                                                          start_time=ts_old,
                                                          end_time=ts_new)
        # flow is a tuple of [H,W]. Stack it
        return numpy.stack(flow)

    @staticmethod
    def mvsec_time_conversion(timestamps):
        raise NotImplementedError

    def get_ts(self, path, i):
        try:
            f = open(path, "r")
            return float(f.readlines()[i])
        except OSError:
            raise

    def get_image_width_height(self, type='event_camera'):
        if hasattr(self, 'cropper'):
            h = self.cropper.size[0]
            w = self.cropper.size[1]
            return h, w
        return self.image_height, self.image_width

    def get_events(self, loader_idx, train_val_test_type):
        # Get Events For Visualization Only!!!
        path_dataset = os.path.join(self.path_dataset,self.dataset[loader_idx]['dataset_name'] + "_" + str(self.dataset[loader_idx]['subset_number']))
        params = {'height': self.image_height, 'width': self.image_width}
        i = self.dataset[loader_idx]['index']
        path = os.path.join(path_dataset, TEMPLATES.MVSEC_EVENTS_FILE.format('left', i+1))
        events = EventSequence(get_events(path, train_val_test_type), params).get_sequence_only()
        return events

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx, force_crop_window=None, force_flipping=None):
        if idx >= len(self):
            raise IndexError
        sample = self.get_data_sample(idx)
        # Center Crop Everything
        # sample['flow'] = self.cropper(sample['flow'])
        # sample['gt_valid_mask'] = self.cropper(sample['gt_valid_mask'])
        # sample['event_volume_new'] = self.cropper(sample['event_volume_new'])
        # sample['event_volume_old'] = self.cropper(sample['event_volume_old'])

        return sample

class MvsecFlowRecurrent(Dataset):
    def __init__(self, args, type, path, aug):
        super(MvsecFlowRecurrent, self).__init__()
        if type.lower() != 'test':
            self.sequence_length = args['sequence_length']
        else:
            self.sequence_length = 1
        # step_size=1 로인해 sequence_length=2로 설정하더라도 (0,1) -> (1, 2) -> (2, 3)과 같이 data가 load된다. 
        self.step_size = 1
        self.dataset = MvsecFlow(args, type, path=path)  
        self.aug = aug      

    def __len__(self):
        return (len(self.dataset) - self.sequence_length) // self.step_size + 1

    def __getitem__(self, idx):
        # ----------------------------------------------------------------------------- #
        # Returns a list, containing of event/frame/flow                                #
        # [ e_(i-sequence_length), ..., e_(i) ]                                         #
        # ----------------------------------------------------------------------------- #
        assert(idx >= 0)
        assert(idx < len(self))
        sequence = []
        j = idx * self.step_size

        flip = None
        crop_window = None

        for i in range(self.sequence_length):
            sequence.append(self.dataset.__getitem__(j + i, force_crop_window=crop_window, force_flipping=flip))
            
        ## Augmentation 
        if self.aug == True:
            # ----------------------------------------------------------------------------- #
            # Augmentation Part ( Random Cropping + Flipping (Horizontal & Vertical))       #
            # ----------------------------------------------------------------------------- #
            ## Random 변수 설정 
            crop_size_height = 256
            crop_size_width = 256
            # random starting point (x, y)
            y0 = np.random.randint(0, sequence[0]['flow'].shape[1] - crop_size_height) # 260 
            x0 = np.random.randint(0, sequence[0]['flow'].shape[2] - crop_size_width) # 346
            ## Random Flipping 변수 설정 
            rand_h = np.random.rand()
            rand_v = np.random.rand()
            # RandomFlip (Horizontal prob : 0.5, Vertical prob : 0.1)
            h_flip_prob = 0.5
            v_flip_prob = 0.1
            
            for i in range(self.sequence_length):
                # Cropping 
                event_new = sequence[i]['event_volume_new'].detach().cpu().numpy()
                event_old = sequence[i]['event_volume_old'].detach().cpu().numpy()
                flow = sequence[i]['flow'].detach().cpu().numpy()
                flow_gt_mask = sequence[i]['gt_valid_mask'].detach().cpu().numpy()
                event_new = event_new[:,y0:y0+crop_size_height, x0:x0 + crop_size_width]
                event_old = event_old[:,y0:y0+crop_size_height, x0:x0 + crop_size_width]
                flow = flow[:,y0:y0+crop_size_height, x0:x0 + crop_size_width]
                flow_gt_mask = flow_gt_mask[:,y0:y0+crop_size_height, x0:x0 + crop_size_width]
                
                # Random Flipping 
                if rand_h < h_flip_prob:
                    # event voxel 
                    event_new = event_new[:, :, ::-1]
                    event_old = event_old[:, :, ::-1]
                    # flow 
                    flow = flow[:, :, ::-1]
                    flow[0] = flow[0] * -1.0 
                    flow[1] = flow[1] * 1.0
                    # gt mask 
                    flow_gt_mask = flow_gt_mask[:, :, ::-1]
                
                if rand_v < v_flip_prob:
                    # event voxel 
                    event_new = event_new[:, ::-1]
                    event_old = event_old[:, ::-1]
                    # flow 
                    flow = flow[:, ::-1]
                    flow[0] = flow[0] * 1.0
                    flow[1] = flow[1] * -1.0
                    # gt mask 
                    flow_gt_mask = flow_gt_mask[:, ::-1]
                
                # To Tensor
                sequence[i]['event_volume_new'] = torch.from_numpy(event_new.copy())
                sequence[i]['event_volume_old'] = torch.from_numpy(event_old.copy())
                sequence[i]['flow'] = torch.from_numpy(flow.copy())
                sequence[i]['gt_valid_mask'] = torch.from_numpy(flow_gt_mask.copy())
            
        # Just Making Sure
        assert sequence[-1]['idx']-sequence[0]['idx'] == self.sequence_length-1
        return sequence

    def summary(self, logger):
        logger.write_line("================================== Dataloader Summary ====================================", True)
        logger.write_line("Loader Type:\t\t" + self.__class__.__name__ + " for {}".format(self.dataset.type), True)
        logger.write_line("Sequence Length:\t{}".format(self.sequence_length), True)
        logger.write_line("Step Size:\t\t{}".format(self.step_size), True)
        logger.write_line("Framerate:\t\t{}".format(self.dataset.update_rate), True)

    def get_image_width_height(self, type='event_camera'):
        return self.dataset.get_image_width_height(type)

    def get_events(self, loader_idx, train_val_test_type):
        return self.dataset.get_events(loader_idx, train_val_test_type)