import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init
from torch.autograd import Function
#torch.manual_seed(42)


class LRCNStyle(nn.Module):

    def __init__(self, batch_size, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):
        '''
        LRCN model as described in https://arxiv.org/abs/1411.4389 
        The current implementation extarcts the visual representations 
        using Conv3D, and then processes it with 2 LSTMs, sharing the 
        hidden and cell states (averaged every few frame slices).
        The readout stage predicts a label from every frame and then 
        uses majority voting to select a label for the sample. 
        '''
        super(LRCNStyle, self).__init__()
        # self.timesteps = timesteps
        # self.num_iter = num_iter
        # self.exp_name = exp_name
        # self.jacobian_penalty = jacobian_penalty
        # self.grad_method = grad_method
        self.batch_size=batch_size
        self.hidden_size=32
        self.embedding_dim=4
        
        # self.conv0 = nn.Conv2d(3, 25, kernel_size=7, padding=3) # using 3 channel input now
        self.conv0 = nn.Conv3d(3, 3, kernel_size=7, bias=False, padding=3)
        nn.init.normal_(self.conv0.weight, mean=0.0, std=1.0)
        self.conv1 = nn.Conv3d(3, self.embedding_dim, kernel_size=7, padding=3)

        self.lstm1 = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=2, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size*2, self.embedding_dim, num_layers=2, bidirectional=True)
        self.fc1 = nn.Linear(64*64*self.embedding_dim*2, 1)
        # part1 = np.load('utils/gabor_serre.npy')
        # inflate the weight file to accomodate 3 channel, 3D video input
        # part1=np.repeat(part1,3,axis=1)
        # part1 = np.expand_dims(part1, axis=0)
        # print(part1.shape)
        # import pdb; pdb.set_trace()
        # self.conv00.weight.data = torch.FloatTensor(part1)

        # self.conv1 = nn.Conv3d(1, 25, kernel_size=7, padding=3)

        
        # self.unit1 = hConvGRUCell(self.hgru_size, self.hgru_size, filt_size)
        # self.unit1 = nn.LSTM(4, self.hidden_size, num_layers=2, bidirectional=True)#, batch_first=True)
        print("Training with filter size:", filt_size, "x", filt_size)
        # self.bn = nn.BatchNorm3d(self.hgru_size, eps=1e-03, track_running_stats=False)
        # self.bn = nn.InstanceNorm3d(self.hidden_size*2, eps=1e-03, track_running_stats=False) # not using normalization
        # self.conv6 = nn.Conv3d(25, 2, kernel_size=1)
        # init.xavier_normal_(self.conv6.weight)
        # init.constant_(self.conv6.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))

        # self.fc4 = nn.Linear(25*128*128*2, 2) # the first 2 is for batch size
        # self.fc4 = nn.Linear(1*self.hgru_size*65*65, 1) # the first 2 is for batch size, the second digit is for the dimension
        self.fc4 = nn.Linear(1*self.hidden_size*2*32*64*64, 1) # the first 2 is for batch size, the second digit is for the dimension
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)


    def forward(self, x, epoch, itr, target, criterion, testmode=False):
        out=F.leaky_relu(self.conv0(x))
        # out=F.dropout(out, p=0.5, training=self.training)
        out=F.leaky_relu(self.conv1(out))
        out=self.avgpool(out)
        out = out.permute(2,0,1,3,4)
        num_frames=out.shape[0]
        outputs=[]
        losses=[]
        for i in range(0,num_frames):
            if i ==0:
                output1, (h1_n, c1_n) = self.lstm1(out[i].reshape(-1, self.batch_size, self.embedding_dim))
                output2, (h2_n, c2_n) = self.lstm2(output1)
            else:
                output1, (h1_n, c1_n) = self.lstm1(out[i].reshape(-1, self.batch_size, self.embedding_dim), (h1_n, c1_n))
                # concat the old and new hidden states, and pass it to the second lstm
                # can average and then expand dim as well, but not sure if that looses a lot of information
                # ideally this should go in a function for modularity, and should be callable with arbitrary shapes
                hn_cat=torch.cat((h1_n,h2_n),2)
                hn_cat=torch.split(hn_cat,9,dim=2)
                hn_cat=[torch.mean(hn_cat[k], dim=2, keepdims=True) for k in range(0,len(hn_cat))]
                hn_cat=torch.stack(hn_cat, dim=2)
                hn_cat=hn_cat.squeeze(-1)
                cn_cat=torch.cat((c1_n,c2_n),2)
                cn_cat=torch.split(cn_cat,9,dim=2)
                cn_cat=[torch.mean(cn_cat[k], dim=2, keepdims=True) for k in range(0,len(cn_cat))]
                cn_cat=torch.stack(cn_cat, dim=2)
                cn_cat=cn_cat.squeeze(-1)
                output2, (h2_n, c2_n) = self.lstm2(output1, (hn_cat, cn_cat))
            # apply readout here (reading out a label from every frame)
            output=F.dropout(output2, p=0.5, training=self.training)
            output=self.fc1(output.view(self.batch_size, -1))
            output=torch.squeeze(output)
            output=torch.sigmoid(output.clone())
            loss = criterion(output, target.float())
            outputs.append(output)
            losses.append(loss)
            # collect loss and logits for every frame, then calculate average and majority across the entire video respectively
            # return output (convert back to around 0.5), and averaged loss
        
        outputs=torch.stack(outputs, dim=1) # transpose the column vector to row vector and stack, so that every row has responses for the entire example. If not using dim=1, the column vector will have all the responses
        losses=torch.stack(losses)

        # calculate majority vote here
        pts=[]
        for j in range(0,outputs.shape[0]):
            tt=target[j]
            pr=[]
            for k in outputs[j]:
                if k>0.5: pr.append(torch.tensor([1]))
                else: pr.append(torch.tensor([0]))
            pred=torch.stack(pr).cuda()
            if pred.sum()>(len(pred)/2): pts.append(torch.tensor([1]))
            else: pts.append(torch.tensor([0]))
        pts=torch.stack(pts)
        pts=pts.squeeze()

        outputs=[torch.tensor([0.7]) if k else torch.tensor([0.2]) for k in pts]
        outputs=torch.stack(outputs)
        outputs=outputs.squeeze()

        jv_penalty = torch.tensor([1]).float().cuda() # no use, just to keep the code running with the old framework

        if testmode: return outputs, states, losses.mean()
        return outputs, jv_penalty, losses.mean()






        # # out=out.permute(0,2,3,4,1)
        # # npnp=out.cpu().detach().numpy()
        # # np.save("gaussian_responses.npy", npnp)



        # # z-score normalize input video to the mean and std of the gaussian weight inits
        # # x = (x - torch.mean(x, axis=[1, 3, 4], keepdims=True)) / torch.std(x, axis=[1, 3, 4], keepdims=True)
        # # average across the RGB channel dimension
        # # x=torch.mean(x,axis=1, keepdims=True)

        # with torch.no_grad():   # stopping the first gausian init input layer from learning 
        #     out = self.conv00(x)
        #     # out = self.conv0(out) # 1x1 conv to inflate feature maps to 8 dims
        #     # out=out.repeat(1,self.hgru_size,1,1,1)
        # # import pdb; pdb.set_trace()
        # # out.requires_grad=True
        # out = torch.pow(out, 2)
        # # out = out.permute(2,0,1,3,4)
        # # internal_state = torch.zeros_like(out, requires_grad=False)
        # # internal_state = torch.zeros_like(torch.empty(4,4,8))
        # # internal_state = torch.zeros_like(torch.empty(4,4,64,64,64).cuda(), requires_grad=False)

        # # for t in range(0,out.shape[0]):
        # for t in range(0,self.timesteps):
        #     if t==0:
        #         output, (h_n, c_n) = self.unit1(out.view(-1, self.batch_size,self.embedding_dim))
        #     else:
        #         output, (h_n, c_n) = self.unit1(out.view(-1, self.batch_size,self.embedding_dim), (h_n, c_n))
        #     # internal_state, g2t = self.unit1(out, internal_state, timestep=t)
        #     # internal_state, g2t = self.unit1(out.view(-1, self.batch_size,self.embedding_dim), internal_state)
        #     # if t == self.timesteps - 2:
        #     #     state_2nd_last = internal_state
        #     # elif t == self.timesteps - 1:
        #     #     last_state = internal_state        

        # # import pdb; pdb.set_trace()
        # # output = self.bn(internal_state)
        # output = self.bn(output.view(self.batch_size, self.hgru_size*2, 64, 128, 128))
        # # output = torch.mean(output,1)
        # output = self.avgpool(output)
        # output=output.view(self.batch_size,-1)
        # output=self.fc4(output)
        # output=torch.squeeze(output)
        # output=torch.sigmoid(output.clone())
        # loss = criterion(output, target.float())


        # # pen_type = 'l1'
        # jv_penalty = torch.tensor([1]).float().cuda()
        # # mu = 0.9
        # # double_neg = False
        # # if self.training and self.jacobian_penalty:
        # #     if pen_type == 'l1':
        # #         norm_1_vect = torch.ones_like(last_state)
        # #         norm_1_vect.requires_grad = False
        # #         jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
        # #                                       create_graph=self.jacobian_penalty, allow_unused=True)[0]
        # #         jv_penalty = (jv_prod - mu).clamp(0) ** 2
        # #         if double_neg is True:
        # #             neg_norm_1_vect = -1 * norm_1_vect.clone()
        # #             jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[neg_norm_1_vect], retain_graph=True,
        # #                                           create_graph=True, allow_unused=True)[0]
        # #             jv_penalty2 = (jv_prod - mu).clamp(0) ** 2
        # #             jv_penalty = jv_penalty + jv_penalty2
        # #     elif pen_type == 'idloss':
        # #         norm_1_vect = torch.rand_like(last_state).requires_grad_()
        # #         jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
        # #                                       create_graph=True, allow_unused=True)[0]
        # #         jv_penalty = (jv_prod - norm_1_vect) ** 2
        # #         jv_penalty = jv_penalty.mean()
        # #         if torch.isnan(jv_penalty).sum() > 0:
        # #             raise ValueError('Nan encountered in penalty')
        # if testmode: return output, states, loss
        # return output, jv_penalty, loss

