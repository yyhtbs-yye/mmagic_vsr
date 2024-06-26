import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .memc_modules import MotionCompensator
from .espcn_modules import ESPCN_multiframe2
from .huber_loss import Approx_Huber_Loss

class VESPCN(nn.Module):
    def __init__(self, args):
        super(VESPCN, self).__init__()

        self.mseloss = nn.MSELoss()
        self.huberloss = Approx_Huber_Loss(args)
        self.motionCompensator = MotionCompensator(args)
        self.espcn = ESPCN_multiframe2(args)
        
        self.motionCompensator.load_state_dict(torch.load('./experiment/model_best_mc.pt'), strict=False)
        self.espcn.load_state_dict(torch.load('./experiment/model_best_espcn.pt'), strict=False)

    def forward(self, frame_list):
        # squeeze frames n_sequence * [N, 1, n_colors, H, W] -> n_sequence * [N, n_colors, H, W]
        frame_list = [torch.squeeze(frame, dim = 1) for frame in frame_list]
        
        frame1 = frame_list[0]
        frame2 = frame_list[1]
        frame3 = frame_list[2]

        frame1_compensated, flow1 = self.motionCompensator(frame2, frame1)
        frame3_compensated, flow2 = self.motionCompensator(frame2, frame3)
        
        loss_mc_mse = self.mseloss(frame1_compensated, frame2) + self.mseloss(frame3_compensated, frame2)
        loss_mc_huber = self.huberloss(flow1) + self.huberloss(flow2)
        
        #print(frame1_compensated.shape, frame2.shape, frame3_compensated.shape)
        # n_sequence * [N, n_colors, H, W] -> [N, n_sequence * n_colors, H, W]
        lr_frames_cat = torch.cat((frame1_compensated, frame2, frame3_compensated), dim = 1) 
        #print(lr_frames_cat.shape)
        return self.espcn(lr_frames_cat), loss_mc_mse, loss_mc_huber