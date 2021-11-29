import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init
from torch.autograd import Function
#torch.manual_seed(42)
from models.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell



class FFSTLSTM(nn.Module):

    def __init__(self, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):
        super(FFSTLSTM, self).__init__()
        self.timesteps = timesteps
        self.num_iter = num_iter
        self.exp_name = exp_name
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.num_layers = 4
        self.batch_size = 16
        
        # self.conv0 = nn.Conv2d(3, 25, kernel_size=7, padding=3) # using 3 channel input now
        self.conv0 = nn.Conv3d(3, 25, kernel_size=7, bias=False, padding=3)
        self.pool = nn.MaxPool3d(2,2)
        self.conv1 = nn.Conv3d(25, 10, kernel_size=7, bias=False, padding=3)
        self.conv2 = nn.Conv3d(10, 8, kernel_size=7, bias=False, padding=3)
        part1 = np.load('utils/gabor_serre.npy')
        # inflate the weight file to accomodate 3 channel, 3D video input
        part1=np.repeat(part1,3,axis=1)
        part1 = np.expand_dims(part1, axis=0)
        print(part1.shape)
        # self.conv0.weight.data = torch.FloatTensor(part1)
        
        # self.unit1 = hConvGRUCell(25, 25, filt_size)
        self.unit1 = SpatioTemporalLSTMCell(in_channel=8, num_hidden=8, width=16, filter_size=filt_size,
                                       stride=1, layer_norm=1)
        print("Training with filter size:", filt_size, "x", filt_size)
        self.bn = nn.BatchNorm2d(8, eps=1e-03, track_running_stats=False)
        self.conv6 = nn.Conv3d(25, 2, kernel_size=1)
        init.xavier_normal_(self.conv6.weight)
        init.constant_(self.conv6.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))

        # self.fc4 = nn.Linear(25*128*128*2, 2) # the first 2 is for batch size
        self.fc4 = nn.Linear(self.batch_size*8*8*8, self.batch_size) # the first 2 is for batch size
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x, epoch, itr, target, criterion, testmode=False):
        out = self.pool(F.relu(self.conv0(x)))
        out = self.pool(F.relu(self.conv1(out)))
        out = self.pool(F.relu(self.conv2(out)))
        # out = torch.pow(out, 2)
        out = out.permute(2,0,1,3,4)
        # internal_state = torch.zeros_like(out[0], requires_grad=False)

        h_t=[]
        c_t=[]

        for i in range(self.num_layers):
            zeros = torch.zeros([self.batch_size, 8, 16, 16]).cuda() # the first dim is the batch size
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([self.batch_size, 8, 16, 16]).cuda() # forcing it to be on cuda for now
        # h_t = torch.zeros([2, 64, 128, 128]).cuda() # forcing it to be on cuda for now
        # c_t = torch.zeros([2, 64, 128, 128]).cuda() # forcing it to be on cuda for now

        # h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
        



        for t in range(0,out.shape[0]):
            y=out[t]
            h_t[0], c_t[0], memory = self.unit1(y, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.unit1(h_t[i - 1], h_t[i], c_t[i], memory)
            
            # internal_state, g2t = self.unit1(y, h_t, c_t, memory)
            # if t == self.timesteps - 2:
            #     state_2nd_last = internal_state
            # elif t == self.timesteps - 1:
            #     last_state = internal_state        
        output = h_t[self.num_layers-1]

        output = self.bn(output)
        # output = torch.mean(output,1)
        output = self.avgpool(output)
        # import pdb; pdb.set_trace()
        output=output.view(1,-1)
        output=self.fc4(output)
        output=torch.squeeze(output)
        output=torch.sigmoid(output.clone())
        loss = criterion(output, target.float())


        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        mu = 0.9
        double_neg = False
        if self.training and self.jacobian_penalty:
            if pen_type == 'l1':
                norm_1_vect = torch.ones_like(last_state)
                norm_1_vect.requires_grad = False
                jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
                                              create_graph=self.jacobian_penalty, allow_unused=True)[0]
                jv_penalty = (jv_prod - mu).clamp(0) ** 2
                if double_neg is True:
                    neg_norm_1_vect = -1 * norm_1_vect.clone()
                    jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[neg_norm_1_vect], retain_graph=True,
                                                  create_graph=True, allow_unused=True)[0]
                    jv_penalty2 = (jv_prod - mu).clamp(0) ** 2
                    jv_penalty = jv_penalty + jv_penalty2
            elif pen_type == 'idloss':
                norm_1_vect = torch.rand_like(last_state).requires_grad_()
                jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
                                              create_graph=True, allow_unused=True)[0]
                jv_penalty = (jv_prod - norm_1_vect) ** 2
                jv_penalty = jv_penalty.mean()
                if torch.isnan(jv_penalty).sum() > 0:
                    raise ValueError('Nan encountered in penalty')
        if testmode: return output, states, loss
        return output, jv_penalty, loss