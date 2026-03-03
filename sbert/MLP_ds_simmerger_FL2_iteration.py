# update on 26th July
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
# from model import CE_build3  # the mmodel
import time
import os

os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_cholec'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval'  # Change this to your target mode

print("Current working directory:", os.getcwd())
import shutil
# from train_display import *
# the model
# import arg_parse
import cv2
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import eval_slots
from model import  model_experiement, model_infer_slot_att
from working_dir_root import Output_root,Linux_computer
from dataset.dataset import myDataloader
from display import Display
import torch.nn.parallel
import torch.distributed as dist
import scheduler


from working_dir_root import GPU_mode ,Continue_flag ,Visdom_flag ,Display_flag ,loadmodel_index  ,img_size,Load_flow,Load_feature
from working_dir_root import Max_lr, learningR,learningR_res,Save_feature_OLG,sam_feature_OLG_dir, Evaluation,Save_sam_mask,output_folder_sam_masks
from working_dir_root import Enable_student,Batch_size,selected_data,Display_down_sample, Data_percentage,Gpu_selection,Evaluation_slots,Max_epoch
from dataset import io
import pathlib
import argparse

# GPU_mode= True
# Continue_flag = True
# Visdom_flag = False
# Display_flag = False
# loadmodel_index = '3.pth'

# Add these lines at the top of the original script
CLIENT_ID = 2
SHARED_ROOT = Output_root + "shared_models/"
STATUS_DIR = os.path.join(SHARED_ROOT, "status")
os.makedirs(SHARED_ROOT, exist_ok=True)
os.makedirs(STATUS_DIR, exist_ok=True)
# Modified model loading function

def load_global_model():
    # components = [
    #     "initializer", "encoder", "processor",
    #     "decoder", "temporal_binder", "presence_nn"
    # ]
    components = [
        "initializer", "encoder", "processor",
        "decoder", 
    ]
    
    global_folder = os.path.join(SHARED_ROOT, "global")
    os.makedirs(global_folder, exist_ok=True)

    for comp in components:
        path = os.path.join(global_folder, f"{comp}.pth")
        if os.path.exists(path):
            # Load to CPU first
            state_dict = torch.load(path, map_location='cpu')
            # Move to current device
            device_state_dict = {k: v.to(device) for k, v in state_dict.items()}
            getattr(Model_infer.model, comp).load_state_dict(device_state_dict)

# Modified client saving function
def save_client_model():
    client_folder = os.path.join(SHARED_ROOT, f"client_{CLIENT_ID}")
    os.makedirs(client_folder, exist_ok=True)
    
    components = [
        "initializer", "encoder", "processor",
        "decoder", "temporal_binder",  "presence_nn"
    ]
    
    for comp in components:
        # Move to CPU before saving
        state_dict = getattr(Model_infer.model, comp).state_dict()
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        torch.save(cpu_state_dict, os.path.join(client_folder, f"{comp}.pth"))

# Modified federated averaging function
def federated_average():
    # Check if both clients are ready
    if not all([os.path.exists(os.path.join(STATUS_DIR, f"client_{cid}.ready")) for cid in [1, 2]]):
        return False

    # Create lock file
    lock_path = os.path.join(STATUS_DIR, "fed.lock")
    if os.path.exists(lock_path):
        return False

    try:
        with open(lock_path, "w") as f: pass
        
        components = [
            "initializer", "encoder", "processor",
            "decoder", "temporal_binder", "presence_nn"
        ]
        client_weights = {1: {}, 2: {}}
        
        # Load weights with existence checks
        for cid in [1, 2]:
            for comp in components:
                path = os.path.join(SHARED_ROOT, f"client_{cid}", f"{comp}.pth")
                client_weights[cid][comp] = torch.load(path, map_location='cpu')  # Force CPU loading

        # Average weights
        # Average weights
        avg_weights = {}
        for comp in components:
            avg_weights[comp] = {}
            for key in client_weights[1][comp]:
                avg = (client_weights[1][comp][key] + client_weights[2][comp][key]) / 2
                avg_weights[comp][key] = avg

        # Save global model
        global_folder = os.path.join(SHARED_ROOT, "global")
        os.makedirs(global_folder, exist_ok=True)
        for comp in components:
            torch.save(avg_weights[comp], os.path.join(global_folder, f"{comp}.pth"))

        # Safe cleanup with existence checks
        for cid in [1, 2]:
            for comp in components:
                path = os.path.join(SHARED_ROOT, f"client_{cid}", f"{comp}.pth")
                if os.path.exists(path):
                    os.remove(path)
                else:
                    print(f"Warning: {path} already removed")
        print("averaged successfully")
        
        return True
        
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)

Output_root = Output_root+ "MLP_dynamic_slots_simmerger_FL" + selected_data + str(Data_percentage) + "/"
io.self_check_path_create(Output_root)
RESULT_FINISHED = 0
RESULT_TIMEOUT = 1

CHECKPOINT_SUBDIR = "checkpoints"
TENSORBOARD_SUBDIR = "tb"
METRICS_SUBDIR = "metrics"

# Define default configuration values
# DEFAULT_CONFIG = "videosaur_m/configs/videosaur/ytvis3_MLP_dinov2.yml"
DEFAULT_CONFIG = "videosaur_m/configs/videosaur/ytvis3_MLP_ds_simmerger7.yml"
# DEFAULT_CONFIG = "videosaur_m/configs/videosaur/ytvis3_MLP_cholec.yml"

# DEFAULT_CONFIG = "videosaur_m/configs/videosaur/ytvis3.yml"


DEFAULT_LOG_DIR = "./logs"

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true", help="Be verbose")
group.add_argument("-q", "--quiet", action="store_true", help="Suppress outputs")
parser.add_argument("-n", "--dry", action="store_true", help="Dry run (no logfiles)")
parser.add_argument(
    "--no-interactive", action="store_true", help="If running in non-interactive environment"
)
parser.add_argument("--no-tensorboard", action="store_true", help="Do not write tensorboard logs")
parser.add_argument(
    "--check-validation", action="store_true", help="Run correctness checks on data used during eval"
)
parser.add_argument(
    "--run-eval-after-training", action="store_true", help="Evaluate after training has stopped"
)
parser.add_argument(
    "--use-optimizations", action="store_true", help="Enable Pytorch performance optimizations"
)
parser.add_argument("--timeout", help="Stop training after this time (format: DD:HH:MM:SS)")
parser.add_argument("--data-dir", help="Path to data directory")
parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Path to log directory")
parser.add_argument(
    "--no-sub-logdirs", action="store_true", help="Directly use log dir to store logs"
)
parser.add_argument(
    "--continue",
    dest="continue_from",
    type=pathlib.Path,
    help="Continue training from this log folder or checkpoint path",
)
parser.add_argument("--config_overrides_file", help="Configuration to override")
parser.add_argument(
    "config", nargs="?", default=DEFAULT_CONFIG, help="Configuration to run"
)
parser.add_argument("config_overrides", nargs="*", help="Additional arguments")

import pickle

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
   
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs available:", num_gpus)
if GPU_mode ==True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# dataroot = "../dataset/CostMatrix/"
# torch.set_num_threads(8)
 # create the model

if Visdom_flag == True:
    from visual import VisdomLinePlotter

    plotter = VisdomLinePlotter(env_name='path finding training Plots')

def is_external_drive(drive_path):
    # Check if the drive is a removable drive (usually external)
    return os.path.ismount(drive_path) and shutil.disk_usage(drive_path).total > 0

def find_external_drives():
    # List all drives on the system
    drives = [d for d in os.listdir('/') if os.path.isdir(os.path.join('/', d))]

    # Filter out external drives and exclude certain paths
    external_drives = [drive for drive in drives if is_external_drive(os.path.join('/', drive))
                       and not drive.startswith(('media', 'run', 'dev'))]

    return external_drives
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict
def add_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = 'module.' + key
        new_state_dict[new_key] = value
    return new_state_dict
# weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
     
############ for the linux to find the extenral drive
external_drives = find_external_drives()

if external_drives:
    print("External drives found:")
    for drive in external_drives:
        print(drive)
else:
    print("No external drives found.")
############ for the linux to find the extenral drive

Model_infer = model_infer_slot_att._Model_infer(parser.parse_args(),GPU_mode,num_gpus,Using_contrast=False,Using_SP_regu = False,Using_SP = True,Using_slot_bert=False,slot_ini= "rnn",cTemp=1.1,gpu_selection=Gpu_selection,pooling="max",TPC=True)
device = Model_infer.device

# if GPU_mode == True:
#     if num_gpus > 1:
#         Model_infer.VideoNets = torch.nn.DataParallel(Model_infer.VideoNets)
#     Model_infer.VideoNets.to(device)

# Model.cuda()
dataLoader = myDataloader(img_size = img_size,Display_loading_video = False,Read_from_pkl= True,Save_pkl = False,Load_flow=Load_flow, Load_feature=Load_feature,Train_list='else',Device=device)

if Continue_flag == False:
    pass
    # Model_infer.VideoNets.apply(weights_init)
else:
    # torch.save(Model_infer.model.initializer.state_dict(), Output_root + "initializer" + str(saver_id) + ".pth")
    #     torch.save(Model_infer.model.encoder.state_dict(), Output_root + "encoder" + str(saver_id) + ".pth")
    #     torch.save(Model_infer.model.processor.state_dict(), Output_root + "processor" + str(saver_id) + ".pth")
    #     torch.save(Model_infer.model.decoder.state_dict(), Output_root + "decoder" + str(saver_id) + ".pth")
    # Model_infer.model.load_state_dict(torch.load(Output_root + 'model' + loadmodel_index ))
    
    Model_infer.model.initializer.load_state_dict(torch.load(Output_root + 'initializer' + loadmodel_index,map_location='cuda:0'))
    Model_infer.model.encoder.load_state_dict(torch.load(Output_root + 'encoder' + loadmodel_index,map_location='cuda:0' ))
    Model_infer.model.processor.load_state_dict(torch.load(Output_root + 'processor' + loadmodel_index,map_location='cuda:0' ))
    Model_infer.model.presence_nn.load_state_dict(torch.load(Output_root + 'presence_nn' + loadmodel_index,map_location='cuda:0' ))
    Model_infer.model.decoder.load_state_dict(torch.load(Output_root + 'decoder' + loadmodel_index,map_location='cuda:0' ))
    Model_infer.model.temporal_binder.load_state_dict(torch.load(Output_root + 'temporal_binder' + loadmodel_index,map_location='cuda:0' ))

        # Load the entire state dictionary for the encoder
    # Load the entire state dictionary for the encoder
    # state_dict = torch.load(Output_root + 'encoder' + loadmodel_index, map_location='cuda:0')

    # # # Filter the state dictionary to only include keys related to the backbone
    # backbone_state_dict = {k[len("module.backbone."):]: v for k, v in state_dict.items() if k.startswith("module.backbone.")}

    # # # Load the filtered state dictionary into the module.backbone
    # Model_infer.model.encoder.module.backbone.load_state_dict(backbone_state_dict)
read_id = 0
# print(Model_infer.resnet)
# print(Model_infer.VideoNets)

epoch = 0
# transform = BaseTransform(  Resample_size,(104/256.0, 117/256.0, 123/256.0))
# transform = BaseTransform(  Resample_size,[104])  #gray scale data
iteration_num = 0
#################
#############training
saver_id =0
displayer = Display(GPU_mode)
epoch =0
features =None
visdom_id=0

# Modified training loop with federated learning integration
round_number = 0
while (1):
    # Load global model at start of each round
    load_global_model()
    
    # Original training iteration
    start_time = time.time()
    start_time = time.time()
    input_videos, labels= dataLoader.read_a_batch(this_epoch= epoch)
    input_videos_GPU = torch.from_numpy(np.float32(input_videos))
    labels_GPU = torch.from_numpy(np.float32(labels))
    input_videos_GPU = input_videos_GPU.to (device)
    labels_GPU = labels_GPU.to (device)
    input_flows = dataLoader.input_flows*1.0/ 255.0
    input_flows_GPU = torch.from_numpy(np.float32(input_flows))  
    input_flows_GPU = input_flows_GPU.to (device)
    if Load_feature ==True:
        features = dataLoader.features.to (device)
    Model_infer.forward(input_videos_GPU,input_flows_GPU,features,Enable_student)
    lr=scheduler.cyclic_learning_rate(current_epoch=epoch,max_lr=Max_lr,min_lr=learningR,cycle_length=4)
    print("learning rate is :" + str(lr))
    if Evaluation == False and Evaluation_slots==False:
        Model_infer.optimization(labels_GPU,Enable_student) 

    if  Save_feature_OLG== True:
        this_features= Model_infer.f[Batch_size-1].permute(1,0,2,3).half()
        sam_pkl_file_name = dataLoader.this_file_name
        sam_pkl_file_path = os.path.join(sam_feature_OLG_dir, sam_pkl_file_name)

        with open(sam_pkl_file_path, 'wb') as file:
            pickle.dump(this_features, file)
            print("sam Pkl file created:" +sam_pkl_file_name)
    if Save_sam_mask == True:
         
        this_mask= Model_infer.sam_mask.half()
        mask_pkl_file_name = dataLoader.this_file_name
        mask_pkl_file_path = os.path.join(output_folder_sam_masks, mask_pkl_file_name)

        with open(mask_pkl_file_path, 'wb') as file:
            pickle.dump(this_mask, file)
            print("sam Pkl file created:" +mask_pkl_file_name)


    if Display_flag == True and read_id%Display_down_sample == 0:
        dataLoader.labels  = dataLoader.labels * 0 +1
        displayer.train_display(Model_infer,dataLoader,read_id,Output_root)
        print(" epoch" + str (epoch) )
        

    
        

        # break
    

    if Evaluation == False and Evaluation_slots == False:
        
        if read_id % 50== 0 and Visdom_flag == True  :
            
            plotter.plot('l0', 'l0', 'l0', visdom_id, Model_infer.lossDisplay.cpu().detach().numpy())
            # if Enable_student:
            plotter.plot('1ls', '1ls', 'l1s', visdom_id, Model_infer.lossDisplay_s.cpu().detach().numpy())
            plotter.plot('1lp', '1lp', 'l1p', visdom_id, Model_infer.lossDisplay_p.cpu().detach().numpy())

        if read_id % 1== 0   :
            print(" epoch" + str (epoch) )
            print(" loss" + str (Model_infer.lossDisplay.cpu().detach().numpy()) )
            if Enable_student:
                print(" loss_SS" + str (Model_infer.lossDisplay_s.cpu().detach().numpy()) )

  
    # Modified model saving with presence_nn
    if (read_id % 50) == 0  and  Evaluation == False and Evaluation_slots==False:
        components = [
            "initializer", "encoder", "processor",
              "decoder", "temporal_binder","presence_nn"
        ]
        for comp in components:
            torch.save(
                getattr(Model_infer.model, comp).state_dict(),
                Output_root + f"{comp}{saver_id}.pth"
            )
        save_client_model()
        with open(os.path.join(STATUS_DIR, f"client_{CLIENT_ID}.ready"), "w") as f:
            f.write("ready")
        
        print(f"Client {CLIENT_ID} waiting for federation...")
        
        # Simplified waiting logic
        while True:
            # Check partner status
            partner_id = 2 if CLIENT_ID == 1 else 1
            partner_ready = os.path.exists(os.path.join(STATUS_DIR, f"client_{partner_id}.ready"))
            my_ready = os.path.exists(os.path.join(STATUS_DIR, f"client_{CLIENT_ID}.ready"))
            
            if partner_ready and my_ready:
                # Client 1 handles the averaging
                if CLIENT_ID == 1:
                    if federated_average():
                        print("Federation successful!")
                        # Cleanup both ready files
                        for cid in [1, 2]:
                            path = os.path.join(STATUS_DIR, f"client_{cid}.ready")
                            if os.path.exists(path):
                                os.remove(path)
                        break
                # Client 2 waits for cleanup
                else:
                    time.sleep(5)
                    # Check if cleanup happened
                    if not os.path.exists(os.path.join(STATUS_DIR, f"client_{CLIENT_ID}.ready")):
                        break
            else:
                pass
            if not os.path.exists(os.path.join(STATUS_DIR, f"client_{CLIENT_ID}.ready")):
                        break
        # Load new global model
        load_global_model()
        print(f"Starting new round {round_number+1}")
        round_number += 1
        saver_id = 0 if saver_id > 1 else saver_id + 1
    read_id+=1
    visdom_id+=1
    # Federated averaging logic at epoch end
    if dataLoader.all_read_flag == 1:
         
        
        Save_feature_OLG = False
        #remove this for none converting mode
        epoch +=1

        print("finished epoch" + str (epoch) )
        dataLoader.all_read_flag = 0
        read_id=0

        if Evaluation:
            break
        if Save_feature_OLG: 
            break
        if epoch > Max_epoch  :
            output_file = eval_slots.process_metrics_from_excel(Output_root + "/metrics_video.xlsx", Output_root)

            break
    # ... [Rest of original training loop] ...

 
    # print(labels)

    # pass





















