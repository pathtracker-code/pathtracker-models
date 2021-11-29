import torch
from torch import nn
import torch.nn as nn
from torch.nn import functional as F


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding_mode='zeros'):
        " Referenced from https://github.com/happyjin/ConvGRU-pytorch"
        super(ConvGRUCell, self).__init__()
        self.hidden_dim = hidden_dim

        if padding_mode == 'zeros':
            if not isinstance(kernel_size, (list, tuple)):
                kernel_size = (kernel_size, kernel_size)

            padding = kernel_size[0] // 2, kernel_size[1] // 2
            self.conv_reset = nn.Conv2d(input_dim + hidden_dim, self.hidden_dim, kernel_size, padding=padding)
            self.conv_update = nn.Conv2d(input_dim + hidden_dim, self.hidden_dim, kernel_size, padding=padding)

            self.conv_state_new = nn.Conv2d(input_dim+hidden_dim, self.hidden_dim, kernel_size, padding=padding)
        else:
            self.conv_reset = conv_block(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1,
                                         padding=int(kernel_size // 2), batch_norm=False, relu=False,
                                         padding_mode=padding_mode)

            self.conv_update = conv_block(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1,
                                          padding=int(kernel_size // 2), batch_norm=False, relu=False,
                                          padding_mode=padding_mode)

            self.conv_state_new = conv_block(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1,
                                             padding=int(kernel_size // 2), batch_norm=False, relu=False,
                                             padding_mode=padding_mode)

    def forward(self, input, state_cur, testmode=False):
        input_state_cur = torch.cat([input, state_cur], dim=1)

        reset_gate = torch.sigmoid(self.conv_reset(input_state_cur))
        update_gate = torch.sigmoid(self.conv_update(input_state_cur))

        input_state_cur_reset = torch.cat([input, reset_gate*state_cur], dim=1)
        state_new = torch.tanh(self.conv_state_new(input_state_cur_reset))

        state_next = (1.0 - update_gate) * state_cur + update_gate * state_new
        if testmode:
            return state_next, reset_gate
        else:
            return state_next


def conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
               batch_norm=True, relu=True, padding_mode='zeros'):
    layers = []
    assert padding_mode == 'zeros' or padding_mode == 'replicate'

    if padding_mode == 'replicate' and padding > 0:
        assert isinstance(padding, int)
        layers.append(nn.ReflectionPad2d(padding))
        padding = 0

    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=bias))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class GRU(nn.Module):

    def __init__(self, dimensions, timesteps=8, kernel_size=15, jacobian_penalty=False, grad_method='bptt'):
        '''
        '''
        super(GRU, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = dimensions
        self.bn = nn.BatchNorm3d(self.hgru_size, eps=1e-03, track_running_stats=False)
        self.preproc = nn.Conv3d(3, dimensions, kernel_size=1, padding=1 // 2)
        # self.preproc = nn.Conv2d(3, dimensions, kernel_size=1, padding=1 // 2)
        # self.preproc = nn.Parameter(torch.empty((1, dimensions, 1, 1, 1)))
        # init.orthogonal_(self.preproc)
        self.unit1 = ConvGRUCell(
            input_dim=self.hgru_size, hidden_dim=self.hgru_size, kernel_size=kernel_size)
        # self.bn = nn.BatchNorm2d(self.hgru_size, eps=1e-03, track_running_stats=False)
        # self.readout = nn.Linear(timesteps * self.hgru_size, 1) # the first 2 is for batch size, the second digit is for the dimension
        # self.readout_bn = nn.BatchNorm2d(self.hgru_size, eps=1e-03, track_running_stats=False)
        self.readout_conv = nn.Conv2d(dimensions, 1, 1)
        self.target_conv = nn.Conv2d(2, 1, 5, padding=5 // 2)
        torch.nn.init.zeros_(self.target_conv.bias)
        # self.target_conv_0 = nn.Conv2d(3, 16, 5, padding=0)  # padding=5 // 2)
        # self.target_pool_0 = nn.MaxPool2d(2, 2, padding=0)
        # self.target_conv_1 = nn.Conv2d(16, 16, 5, padding=0)  # padding=7 // 2)
        # self.target_pool_1 = nn.MaxPool2d(2, 2, padding=0)
        # self.target_conv_2 = nn.Conv2d(16, 1, 5, padding=0)  # padding=7 // 2)
        self.readout_dense = nn.Linear(1, 1)
        # torch.nn.init.zeros_(self.readout_dense.bias)
        self.nl = F.softplus

    def forward(self, x, testmode=False):
        # First step: replicate x over the channel dim self.hgru_size times
        xbn = self.preproc(x)
        # xbn = self.bn(xbn)  # This might be hurting me...
        xbn = self.nl(xbn)  # TEST TO SEE IF THE NL STABLIZES

        # Now run RNN
        x_shape = xbn.shape
        excitation = torch.zeros((x_shape[0], x_shape[1], x_shape[3], x_shape[4]), requires_grad=False).to(x.device)

        # Loop over frames
        states = []
        gates = []
        for t in range(x_shape[2]):
            out = self.unit1(
                input=xbn[:, :, t], state_cur=excitation, testmode=testmode)
            if testmode:
                excitation, gate = out
                gates.append(gate)  # This should learn to keep the winner
                states.append(self.readout_conv(excitation))  # This should learn to keep the winner
            else:
                excitation = out

        output = torch.cat([self.readout_conv(excitation), x[:, 2, 0][:, None]], 1)

        # Potentially combine target_conv + readout_bn into 1
        output = self.target_conv(output)  # output.sum(1, keepdim=True))  # 2 channels -> 1. Is the dot in the target?
        output = F.avg_pool2d(output, kernel_size=output.size()[2:])
        output = output.reshape(x_shape[0], -1)
        output = self.readout_dense(output)
        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        if testmode: return output, torch.stack(states, 1), torch.stack(gates, 1)
        return output, jv_penalty


