import torch
import torch.nn as nn
import argparse
import logging
import warnings
from typing import Any, Dict, Optional
import os
import time

# import pytorch_lightning as pl
import torch.nn.utils as nn_utils

import torch.nn.functional as F
import pathlib
# from pytorch_lightning.utilities import rank_zero_info as log_info

# import torchvision.models as models
import cv2
from torch.optim import lr_scheduler

from model.model_3dcnn_linear_TC import _VideoCNN
from model.model_3dcnn_linear_ST import _VideoCNN_S
from working_dir_root import learningR,learningR_res,SAM_pretrain_root,Load_feature,Weight_decay,Evaluation,Display_student,Display_final_SAM
# from working_dir_root import Enable_teacher
from dataset.dataset import class_weights
import numpy as np
from image_operator import basic_operator   
# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import unary_from_softmax
from SAM.segment_anything import  SamPredictor, sam_model_registry
from working_dir_root import Random_mask_temporal_feature,Random_mask_patch_feature,Display_fuse_TC_ST
from working_dir_root import Use_max_error_rejection,Evaluation_slots,Display_embedding,min_lr
from model import model_operator
# from MobileSAM.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from dataset.dataset import label_mask,Mask_out_partial_label,Obj_num
from videosaur_m. videosaur import configuration, data, metrics, models, utils
from working_dir_root import Display_flag,video_saur_pretrain,Evaluation_slots
import random
import model.model_operator_slots as slots_op
import model.display. model_vis as modelVis
if Evaluation == True:
    learningR=0
    Weight_decay=0
# learningR = 0.0001
def select_gpus(gpu_selection):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("Number of GPUs available:", num_gpus)
        if gpu_selection == "all":
            device = torch.device("cuda" if num_gpus > 0 else "cpu")
            # if num_gpus > 1:
            #     device = torch.device("cuda:0," + ",".join([str(i) for i in range(1, num_gpus)]))
        elif gpu_selection.isdigit():
            gpu_index = int(gpu_selection)
            device = torch.device("cuda:" + gpu_selection if gpu_index < num_gpus else "cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
class _Model_infer(object):
    def __init__(self,args,args_s, GPU_mode =True,num_gpus=1,Using_contrast=False,Using_SP_regu = False, Using_SP = False,
                 Using_slot_bert=False,slot_ini = "rnn",Sim_threshold=0.9,Foundation_M="CLIP",foundation_list  = ["DINO","SAM","MAE"],fusion_method='moe_layer',Mask_feat=False,cTemp=1.0,img_sim=False,
                 gpu_selection = "all",alpha=0.001, Context_len = 5,TPC=True):
        config_overrides=None
        if GPU_mode ==True:
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = select_gpus(gpu_selection)
        else:
            device = torch.device("cpu")
        self.device = device
        self.use_contrast = Using_contrast
        self.alpha = alpha
        self.Using_SP_regu =  Using_SP_regu
        self.Using_SP  = Using_SP
        self.use_bert = Using_slot_bert
        self.slot_ini = slot_ini
        self.Mask_feat= Mask_feat
        self.cTemp = cTemp
        self.img_sim = img_sim
        self.schedulers = None
        if Evaluation_slots:
            self.Mask_feat = False
        rank_zero = utils.get_rank() == 0
        if config_overrides is None:
            config_overrides = args.config_overrides
            config_overrides_s = args_s.config_overrides
        config = configuration.load_config(args.config, config_overrides)
        config_s = configuration.load_config(args_s.config, config_overrides_s)
        if args.config_overrides_file is not None:
            config = configuration.override_config(
                config,
                override_config_path=args.config_overrides_file,
                additional_overrides=config_overrides,
            )
        if args_s.config_overrides_file is not None:
            config_s = configuration.override_config(
                config_s,
                override_config_path=args_s.config_overrides_file,
                additional_overrides=config_overrides_s,
            )

        

        # if config.train_metrics is not None:
        #     train_metrics = {
        #         name: metrics.build(config) for name, config in config.train_metrics.items()
        #     }
        # else:
        #     train_metrics = None

        # if config.val_metrics is not None:
        #     val_metrics = {name: metrics.build(config) for name, config in config.val_metrics.items()}
        # else:
        #     val_metrics = None
        
        self.model = models.build(config.model, config.optimizer,Using_SP=self.Using_SP,Sim_threshold=Sim_threshold,Foundation_M=Foundation_M,foundation_list = foundation_list,fusion_method = fusion_method,Context_len= Context_len)
        self.model_s = models.build(config_s.model, config_s.optimizer,Using_SP=self.Using_SP,Sim_threshold=Sim_threshold,Foundation_M=Foundation_M,foundation_list = foundation_list,fusion_method = fusion_method,Context_len= Context_len)
                # After building models:
        if self.model.adapter is not None:
            self.model.adapter.apply(initialize_weights)
            self.model_s.adapter.apply(initialize_weights)
        # self.model.adapter.apply(initialize_weights)
        # self.model_s.adapter.apply(initialize_weights)
        # self.model.encoder.module.output_transform.apply(initialize_weights)
        self.model.initializer.apply(initialize_weights)
        self.model_s.initializer.apply(initialize_weights)
        self.model.processor.apply(initialize_weights)
        self.model_s.processor.apply(initialize_weights)
        self.model.decoder.apply(initialize_weights)
        self.model_s.decoder.apply(initialize_weights)
        # checkpoint = torch.load(video_saur_pretrain, map_location=torch.device('cpu'))
         
        # print(checkpoint['state_dict'].keys())
        # full_state_dict = checkpoint['state_dict']
        # initializer_state_dict = {k.replace('initializer.', ''): v for k, v in full_state_dict.items() if k.startswith('initializer.')}
        # self.model.initializer.load_state_dict(initializer_state_dict)

        # processor_state_dict = {k.replace('processor.', ''): v for k, v in full_state_dict.items() if k.startswith('processor.')}
        # self.model.processor.load_state_dict(processor_state_dict)

        

        # decoder_state_dict = {k.replace('decoder.', ''): v for k, v in full_state_dict.items() if k.startswith('decoder.')}
        # self.model.decoder.load_state_dict(decoder_state_dict)

        # encoder_state_dict = {k.replace('encoder.', ''): v for k, v in full_state_dict.items() if k.startswith('encoder.')}
        # self.model.encoder.load_state_dict(encoder_state_dict)
        # self.model.load_state_dict(checkpoint['state_dict'])
        # encoder_state_dict = {k.replace('encoder.module.output_transform.', ''): v for k, v in full_state_dict.items() if k.startswith('encoder.module.output_transform.')}
        # self.model.encoder.load_state_dict(encoder_state_dict)
        # self.model.load_state_dict(checkpoint['state_dict'])
        
        # if Using_contrast:
            # Temperature constrained between 0.01 and 10.0
        self.log_cTemp = nn.Parameter(torch.log(torch.tensor(cTemp)))
        self.temp_min = 0.01
        self.temp_max = 10.0
        dummy_input = torch.rand(4, 14, 3, 224, 224)
        # Initialize the tensor with zeros

        # Directly create the dictionary with the tensor
        dummy_batch = {"video": dummy_input}
        optimizers = self.model .configure_optimizers()
        # If `configure_optimizers` returns a dictionary
        if isinstance(optimizers, dict):
            optimizer = optimizers['optimizer']
        elif isinstance(optimizers, (list, tuple)):
            optimizer = optimizers[0]  # assuming there's at least one optimizer
        else:
            optimizer = optimizers
        # self.optimizer = optimizer
        self.optimizer_s=None
        if self.Using_SP == True :
            self.optimizer = torch.optim.AdamW ([ 
            # {'params': self.model.initializer.parameters(),'lr': learningR},
            {'params': self.model.encoder.module.output_transform.parameters(),'lr': learningR},
            # {'params': self.model.processor.parameters(),'lr': learningR},
            {'params': self.model.presence_nn.parameters(),'lr': learningR},
            {'params': self.model.decoder.parameters(),'lr': learningR},
            {'params': self.model.temporal_binder.parameters(),'lr': learningR},
            {'params': self.model.future_state_prdt.parameters(),'lr': learningR}
            ], weight_decay=Weight_decay)
        elif Foundation_M == "ensemble":
            self.optimizer = torch.optim.AdamW ([ 
            {'params': self.model.initializer.parameters(),'lr': learningR},
            # # {'params': self.model.encoder.parameters(),'lr': 0.0001*learningR},
            {'params': self.model.adapter.parameters(),'lr': learningR},

            {'params': self.model.processor.parameters(),'lr': learningR},
            # {'params': self.model.presence_nn.parameters(),'lr': learningR},
            {'params': self.model.decoder.parameters(),'lr': learningR},
            # {'params': self.model.temporal_binder.parameters(),'lr': learningR}
            ], weight_decay=0)

            self.optimizer_s = torch.optim.AdamW ([ 
            {'params': self.model_s.initializer.parameters(),'lr': learningR},
            # {'params': self.model.encoder.module.gate.parameters(),'lr': learningR},
            {'params': self.model_s.adapter.parameters(),'lr': 0.1*learningR},

            {'params': self.model_s.processor.parameters(),'lr': learningR},
            # {'params': self.model.presence_nn.parameters(),'lr': learningR},
            {'params': self.model_s.decoder.parameters(),'lr': learningR},
            # {'params': self.model.temporal_binder.parameters(),'lr': learningR}
            ], weight_decay=Weight_decay)

        else:
            self.optimizer = torch.optim.AdamW ([ 
            {'params': self.model.initializer.parameters(),'lr': learningR},
            # # {'params': self.model.encoder.module.gate.parameters(),'lr': learningR},
            # {'params': self.model.encoder.module.output_transform.parameters(),'lr': learningR},

            # {'params': self.model.adapter.parameters(),'lr': learningR},
            {'params': self.model.processor.parameters(),'lr': learningR},
            # {'params': self.model.presence_nn.parameters(),'lr': learningR},
            # {'params': self.model.decoder.parameters(),'lr': learningR},
            {'params': self.model.temporal_binder.parameters(),'lr': learningR},
            # {'params': self.log_cTemp, 'lr': learningR * 0.1}
            ], weight_decay=Weight_decay)
       
           
        self.input_size = 224
        self.inter_bz = 1
        # self.model.train(True)

        # if GPU_mode == True:
        #     if num_gpus > 1 and gpu_selection == "all":
        #         # self.VideoNets.classifier = torch.nn.DataParallel(self.VideoNets.classifier)
        #         # self.VideoNets.blocks = torch.nn.DataParallel(self.VideoNets.blocks)
        #         self.model = torch.nn.DataParallel(self.model)
        self.model.to (device)
        self.model_s.to (device)

        # self.model.train(True)
        if Evaluation_slots == False:
            self.model.train(True)
            self.model_s.train(True)
        else:
            self.model.train(False)
            self.model_s.train(False)
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 20, eta_min=min_lr, last_epoch=-1)  # Optional parameters explained below
        # if  self.enab:
        if self.optimizer_s is not None:
            self.schedulers = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer_s, 20, eta_min=min_lr, last_epoch=-1)  # Optional parameters explained below

        # self.model.initializer.n_slots = config.n_slots

    #     output_dummy = model(dummy_batch)
    # # Create the dictionary and assign the tensor to the "video" key
    #     loss = model.compute_loss(output_dummy)
    #     loss[0].backward()
    #     optimizer.step()
    def get_temperature(self):
        """Get constrained temperature value"""
        if self.use_contrast and self.log_cTemp is not None:
            raw_temp = torch.exp(self.log_cTemp)
            return torch.clamp(raw_temp, self.temp_min, self.temp_max)
        return self.cTemp
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def convert_mask(self, input, outputs, softmasks=True):

        bz, D, ch, H, W = input["video"].size()

        b, f, n_slots, hw = outputs["decoder"]["masks"].shape
        h = int(np.sqrt(hw))
        w = h
        masks_video = outputs["decoder"]["masks"].reshape(b, f, n_slots, h, w)
        masks_video = masks_video.permute(0,2,1,3,4)
        masks_video = F.interpolate(masks_video,  size=(D, H, W), 
                                      mode='trilinear', align_corners=False)
        if not softmasks:
            ind = torch.argmax(masks_video, dim=1, keepdim=True)
            masks_video = torch.zeros_like(masks_video)
            masks_video.scatter_(1, ind, 1)
        
        return masks_video
    def orthogonality_loss(self,batch_frames_matrix):
    # Reshape the matrix from (Batch, Frames, M, N) to (Batch*Frames, M, N)
        batch_size, num_frames, M, N = batch_frames_matrix.size()
        reshaped_matrix = batch_frames_matrix.reshape(batch_size * num_frames, M, N)
        
        # Normalize each row (vector) in the matrix
        normalized_matrix = reshaped_matrix / reshaped_matrix.norm(dim=2, keepdim=True)
        
        # Compute the cosine similarity matrix for each (M, N) matrix
        # (Batch*Frames, M, M): Cosine similarity between M vectors in each matrix
        cosine_sim_matrix = torch.bmm(normalized_matrix, normalized_matrix.transpose(1, 2))
        
        # Create an identity matrix to exclude diagonal elements (self-similarity)
        identity_matrix = torch.eye(M).to(batch_frames_matrix.device)
        
        # Penalize the off-diagonal elements (cosine similarity between different vectors)
        # We subtract the identity matrix to zero-out diagonal elements, and take absolute value of off-diagonals
        off_diagonal_loss = cosine_sim_matrix - identity_matrix[None, :, :]  # Broadcast identity over batch
        
        # Sum of absolute values of off-diagonal elements
        loss = off_diagonal_loss.abs().sum()
        
        return loss
    def slot_contrastive_loss2(self, batch_frames_matrix, tau=0.1):
        batch_size, num_frames, K, d_slot = batch_frames_matrix.size()
        reshaped = batch_frames_matrix.view(batch_size * num_frames, K, d_slot)
        normalized = F.normalize(reshaped, dim=-1)
        sim_matrix = torch.bmm(normalized, normalized.transpose(1, 2))  # [B*F, K, K]
        
        # Mask diagonal and scale
        off_diag = sim_matrix - torch.eye(K, device=sim_matrix.device).unsqueeze(0)
        loss = torch.logsumexp(off_diag / tau, dim=(1, 2)) - np.log(K*(K-1))
        return loss.mean()
    def slot_contrastive_loss3(self, batch_frames_matrix, target_sim=0.0, tau=0.1, neg_weight=2.0, eps=1e-8):
        """
        Contrastive variant that pushes off-diagonal similarities toward target_sim
        with optional negative-sim penalties.
        """
        B, T, K, D = batch_frames_matrix.size()
        x = batch_frames_matrix.view(B * T, K, D)
        x = x / (x.norm(dim=-1, keepdim=True) + eps)

        sim = torch.bmm(x, x.transpose(1, 2))
        I = torch.eye(K, device=sim.device).unsqueeze(0)
        off = sim * (1 - I)

        # Loss to target similarity
        base = (off - target_sim) ** 2

        # Extra negative penalty
        neg = F.relu(-off)
        neg_penalty = neg_weight * (neg ** 2)

        # logsumexp emphasizes the strongest violators
        per_frame = torch.logsumexp((base + neg_penalty).view(B * T, -1) / tau, dim=1)

        return per_frame.mean()
    def slot_contrastive_loss(self,batch_frames_matrix, tau=0.1):
        # tau=10.0

        batch_size, num_frames, K, d_slot = batch_frames_matrix.size()
        reshaped_matrix = batch_frames_matrix.view(batch_size * num_frames, K, d_slot)

        # Normalize slots
        normalized_matrix = F.normalize(reshaped_matrix, dim=-1)  # (B*F, K, d_slot)

        # Cosine similarity
        sim_matrix = torch.bmm(normalized_matrix, normalized_matrix.transpose(1, 2))  # (B*F, K, K)

        # Mask to exclude diagonal
        mask = ~torch.eye(K, dtype=torch.bool, device=sim_matrix.device).unsqueeze(0)  # (1, K, K)

        # Apply temperature and compute exp
        sim_matrix = sim_matrix / tau
        exp_sim = torch.exp(sim_matrix)

        # Compute normalized similarities (softmax-like)
        exp_sim = exp_sim * mask  # zero diagonal
        denom = exp_sim.sum(dim=-1, keepdim=True) + 1e-8

        # Numerator is each exp(sim_ij), i ≠ j
        log_prob = sim_matrix - torch.log(denom)  # (B*F, K, K)

        # Final loss: average over all non-diagonal elements
        loss = -log_prob * mask
        contrastive_loss = loss.sum() / mask.sum()/ (batch_size * num_frames)

        return contrastive_loss
    def orthogonality_loss2(self, batch_frames_matrix, temperature=1.0):
        # Reshape the matrix from (Batch, Frames, M, N) to (Batch*Frames, M, N)
        batch_size, num_frames, M, N = batch_frames_matrix.size()
        reshaped_matrix = batch_frames_matrix.reshape(batch_size * num_frames, M, N)
        
        # Normalize each row (vector) in the matrix
        normalized_matrix = reshaped_matrix / reshaped_matrix.norm(dim=2, keepdim=True)
        
        # Compute the cosine similarity matrix for each (M, N) matrix
        # (Batch*Frames, M, M): Cosine similarity between M vectors in each matrix
        cosine_sim_matrix = torch.bmm(normalized_matrix, normalized_matrix.transpose(1, 2))
        
        # Apply temperature scaling to the cosine similarity
        cosine_sim_matrix = cosine_sim_matrix / temperature
        
        # Create an identity matrix to exclude diagonal elements (self-similarity)
        identity_matrix = torch.eye(M, device=batch_frames_matrix.device).expand_as(cosine_sim_matrix)
        
        # Mask the diagonal elements (self-similarity)
        # cosine_sim_matrix = cosine_sim_matrix - identity_matrix[None, :, :]  # Broadcast identity over batch
        
        # Apply softmax to each row (cross-entropy-like calculation)
        # Convert the cosine similarities into probabilities
        # cosine_sim_matrix_exp = torch.exp(cosine_sim_matrix)
        # cosine_sim_matrix_prob = cosine_sim_matrix_exp / cosine_sim_matrix_exp.sum(dim=2, keepdim=True)
        
        # Calculate the negative log likelihood (cross-entropy-like loss)
        # For orthogonality, we want all off-diagonal elements to be low, so we focus on the cross-entropy-like term.
        # Here we assume that we want to "penalize" the model for having high values in off-diagonal elements.
        # cross_entropy_loss = -torch.log(cosine_sim_matrix_prob + 1e-8).mean()
        cross_entropy_loss = F.cross_entropy(cosine_sim_matrix,identity_matrix)
    
        return cross_entropy_loss
    def SlotMask_regulation_loss(self, slot_keep: torch.Tensor, linear_weight = 0.1,
        quadratic_weight=  0.1,
        quadratic_bias = 0.5) :
         
        # Calculate the mean sparsity degree across all frames and features
        sparse_degree = torch.mean(slot_keep)  # Averaged over batch, T, and N

     

        # Sparse penalty calculation
        loss = linear_weight * sparse_degree + quadratic_weight * (sparse_degree - quadratic_bias) ** 2
        return loss
    def forward(self,input,batch_files=None, features=None,epoch=None,read_id=None,Output_root=None, Enable_student=False):
        # self.res_f = self.resnet(input)
        start_time = time.time()

        bz, ch, D, H, W = input.size()
        activationLU = nn.ReLU()


        self.input_resample =   F.interpolate(input,  size=(D, self.input_size, self.input_size), mode='trilinear', align_corners=False)
        self.input_resample= (self.input_resample-124.0)/60.0 # size: batch, image_channel:3, frames, height, width
        video_input = {"video":  self.input_resample.permute(0,2,1,3,4)}
        feature_stack = self.model.forward_feature_stack(video_input,self.use_bert, slot_ini=self.slot_ini,Mask_feat=self.Mask_feat,img_sim = self.img_sim,epoch=epoch,batch_files =batch_files,Output_root=Output_root)
        # feature_stack_s = self.model_s.forward_feature_stack(video_input,self.use_bert, slot_ini=self.slot_ini,Mask_feat=self.Mask_feat,img_sim = self.img_sim,epoch=epoch,batch_files =batch_files,Output_root=Output_root)
        
        self.output = self.model(video_input,feature_stack,self.use_bert, slot_ini=self.slot_ini,Mask_feat=self.Mask_feat,img_sim = self.img_sim,epoch=epoch,read_id=read_id,Output_root=Output_root)
        out_puteval = self.output
        
        if Enable_student == True:
            self.output_s = self.model_s(video_input,feature_stack,self.use_bert, slot_ini=self.slot_ini,Mask_feat=self.Mask_feat,img_sim = self.img_sim,epoch=epoch,read_id=read_id,Output_root=Output_root,Student =True )
            out_puteval = self.output_s
        self.cam3D = self.convert_mask(video_input,out_puteval)
        bz, ch_n, D, H, W = self.cam3D.size() # size: batch, maskchannel:11, frames, height, width

        self.raw_cam = self.cam3D
        self.final_output = torch.ones(bz,ch_n,1,1,1)
        self.gradcam = None
        self.direct_frame_output = None


        end_time = time.time()
        print("forward time:", end_time - start_time)
        # modelVis.plot_patch_pca_per_image(input,self.output['encoder']['backbone_features'][:,:,:,768+256:768+256+768])
        # modelVis.plot_patch_pca_per_image(input,self.output['encoder']['backbone_features'][:,:,:,768:768+256])
        # modelVis.plot_patch_pca_per_image(input,self.output['encoder']['backbone_features'][:,:,:,0:768])
        # modelVis.plot_patch_pca_per_image(input,self.output['encoder']['backbone_features'], title="backbone_features")
        # modelVis.plot_patch_pca_per_image(input,self.output['encoder']['features'],title="features")

        if Display_embedding:
                slots = self.output['processor']["state"] 
                modelVis.plot_all_frames_on_projection_vizdom_2d_tsne(slots)
                # modelVis.plot_all_frames_on_hypersphere_vizdom_tsne(slots)
                modelVis.plot_all_frames_on_projection_vizdom_2d(slots)
                modelVis.visdom_cosine_heatmap(slots)
                modelVis.save_cosine_similarity_maps(slots,output_root=Output_root,read_id=read_id)


    def update_ema(self,student, teacher, decay=0.999):
        with torch.no_grad():
            for s_param, t_param in zip(student.parameters(), teacher.parameters()):
                s_param.data = decay * s_param.data + (1 - decay) * t_param.data
    def average_models(self, student, teacher, weight=0.5):
        """
        Perform a weighted average of the student and teacher model parameters.
        The weight determines how much to trust the teacher:
        - weight = 0.0 -> only student parameters are kept
        - weight = 1.0 -> only teacher parameters are kept
        """
        with torch.no_grad():
            for s_param, t_param in zip(student.parameters(), teacher.parameters()):
                avg = (1 - weight) * s_param.data + weight * t_param.data
                s_param.data.copy_(avg)
                t_param.data.copy_(avg)
    def optimization(self, labels, global_state_dict=None, components_avg=None, mu=0.01,epoch=None,read_id=None, Enable_student=False):
        # for param_group in  self.optimizer.param_groups:
            self.optimizer.zero_grad()

            # self.set_requires_grad(self.VideoNets, False)
            slots = self.output['processor']["state"] 
            loss_ortho=torch.tensor(0).cuda()
            if self.use_contrast:
        # Use learned temperature
                learned_cTemp = self.get_temperature()  # or torch.exp(self.log_cTemp) directly
            else:
                learned_cTemp = self.cTemp
            # loss_ortho = self.slot_contrastive_loss3(slots,tau=self.cTemp)
            loss_ortho = self.slot_contrastive_loss3(slots,tau=learned_cTemp)
            print(f"Learned temperature: {learned_cTemp:.4f}")

            # if self.use_contrast == True:
            #     loss_ortho = self.slot_contrastive_loss3(slots,tau=self.cTemp)

            # loss_ortho = self.orthogonality_loss2(slots,temperature=self.cTemp)
            # loss_affin = slots_op.affinity_matrix_regularization(slots)
            loss = self.model.compute_loss(self.output)
            # self.output_s['targets'] = self.output['encoder']["features"]
            if Enable_student == True:
                loss_s = self.model_s.compute_loss(self.output_s)
            # final_loss = loss[0]  


            

             
            if self.Using_SP_regu == False:
                loss_presence_p = 0
                self. lossDisplay_p = torch.tensor(0).cuda()
                self. lossDisplay_p.to(self.device)

            else:
                loss_presence_p = self.SlotMask_regulation_loss(slot_keep=self.output["presence_p"])
                self. lossDisplay_p = loss_presence_p


            if self.use_contrast == False:
                final_loss = loss[0] +  loss_presence_p #  + 0.1*self.model.adapter.pruning_loss    #+ 0.00001*loss_ortho#+ 0.000001 * loss_affin
                
            else:
                final_loss = loss[0] +  loss_presence_p+ self.alpha*loss_ortho#+ 0.000001 * loss_affin
            
            # Add FedProx proximal term
            proximal_loss = torch.tensor(0.0).cuda()
            proximal_loss=proximal_loss.to(final_loss.get_device())
            if global_state_dict is not None and components_avg is not None:
                for comp in components_avg:
                    component = getattr(self.model, comp)
                    for name, param in component.named_parameters():
                        global_param_name = f"{comp}.{name}"
                        if global_param_name in global_state_dict:
                            global_param = global_state_dict[global_param_name].to(param.device)
                            proximal_loss += torch.norm(param - global_param, p=2)**2
                if proximal_loss >0:
                    print("proximal_loss",proximal_loss)
                final_loss += (mu / 2) * proximal_loss


            final_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)

            self.optimizer.step()
            # self.optimizer.zero_grad()
            if Enable_student == True:
                self.optimizer_s.zero_grad()
                final_loss_s = loss_s[0] 
                final_loss_s.backward()
                nn.utils.clip_grad_norm_(self.model_s.parameters(), max_norm=0.05)

                self.optimizer_s.step()

             # EMA update student adapter
            if Enable_student == True:
                pass
                 
                # self.update_ema(self.model_s.processor, self.model.processor, decay=0.5)

            # if gap too large, update ema, else average
            if Enable_student == True and read_id%1000==0:
                # self.update_ema(self.model_s.adapter, self.model.adapter, decay=0.0)
                self.update_ema(self.model_s.initializer, self.model.initializer, decay=0.0)

                # self.average_models(self.model_s.processor, self.model.processor,weight=0.5)
                # self.average_models(self.model_s.adapter, self.model.adapter,weight=0.5)
                if  read_id>20000: # larger from scratch
                # if  read_id>900: #interupted and restarted

                    self.average_models(self.model_s.processor, self.model.processor,weight=0.5)
                    self.average_models(self.model_s.adapter, self.model.adapter,weight=0.5)
                    # self.update_ema(self.model_s.adapter, self.model.adapter, decay=0.00)

                else:
                    # self.update_ema(self.model_s.processor, self.model.processor, decay=0.9)
                    # self.update_ema(self.model_s.adapter, self.model.adapter, decay=0.9)
                    self.update_ema(self.model_s.processor, self.model.processor, decay=0.00)
                    self.update_ema(self.model_s.adapter, self.model.adapter, decay=0.00)

                # self.update_ema(self.model_s.processor, self.model.processor, decay=0.0)
                # self.average_models(self.model_s.adapter, self.model.adapter)
                pass
            if read_id%10000==0:
                self.scheduler.step()
                if Enable_student == True:
                    self.schedulers.step()
                # self.scheduler.step()
                # self.schedulers.step()


            self.lossDisplay = loss[0]
            self. lossDisplay_p= torch.tensor(0).cuda()
            self. lossDisplay_s= torch.tensor(0).cuda()
            self. lossDisplay_orth= loss_ortho


            if Enable_student == True:
                self. lossDisplay_p =  loss_s[0]
            if self.model.adapter is not None:
                # self.lossDisplay_s = self.model.adapter.pruning_loss
                self. lossDisplay_s = self.model.adapter.pruning_loss 


 
