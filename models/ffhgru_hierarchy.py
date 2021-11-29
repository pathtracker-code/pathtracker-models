import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init
from torch.autograd import Function
#torch.manual_seed(42)
from models import transformer


class dummyhgru(Function):
    @staticmethod
    def forward(ctx, state_2nd_last, last_state, *args):
        ctx.save_for_backward(state_2nd_last, last_state)
        ctx.args = args
        return last_state

    @staticmethod
    def backward(ctx, grad):
        neumann_g = neumann_v = None
        neumann_g_prev = grad.clone()
        neumann_v_prev = grad.clone()

        state_2nd_last, last_state = ctx.saved_tensors
        
        args = ctx.args
        truncate_iter = args[-1]
        exp_name = args[-2]
        i = args[-3]
        epoch = args[-4]

        normsv = []
        normsg = []
        normg = torch.norm(neumann_g_prev)
        normsg.append(normg.data.item())
        normsv.append(normg.data.item())
        for ii in range(truncate_iter):
            neumann_v = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=neumann_v_prev,
                                            retain_graph=True, allow_unused=True)
            normv = torch.norm(neumann_v[0])
            neumann_g = neumann_g_prev + neumann_v[0]
            normg = torch.norm(neumann_g)
            
            if normg > 1 or normv > normsv[-1] or normv < 1e-9:
                normsg.append(normg.data.item())
                normsv.append(normv.data.item())
                neumann_g = neumann_g_prev
                break

            neumann_v_prev = neumann_v
            neumann_g_prev = neumann_g
            
            normsv.append(normv.data.item())
            normsg.append(normg.data.item())
        
        return (None, neumann_g, None, None, None, None)


class hConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, hidden_size, kernel_size, timesteps, batchnorm=True, grad_method='bptt', use_attention=False):
        super(hConvGRUCell, self).__init__()
        self.padding = kernel_size // 2
        self.hidden_size = hidden_size
        self.batchnorm = batchnorm
        self.timesteps = timesteps
        self.use_attention = use_attention
        
        if self.use_attention:
            # self.a_w_gate = nn.Conv2d(hidden_size, 1, kernel_size, padding=self.padding)
            # self.a_u_gate = nn.Conv2d(hidden_size, 1, kernel_size, padding=self.padding)
            self.a_w_gate = nn.Conv2d(hidden_size, hidden_size, 1, padding=1 // 2)
            self.a_u_gate = nn.Conv2d(hidden_size, hidden_size, 1, padding=1 // 2)
            init.orthogonal_(self.a_w_gate.weight)
            init.orthogonal_(self.a_u_gate.weight)
            init.constant_(self.a_w_gate.bias, 1.)  # In future try setting to -1 -- originally set to 1
            init.constant_(self.a_u_gate.bias, 1.)
            # init.uniform_(self.a_w_gate.bias.data, 1, self.timesteps - 1)
            # init.uniform_(self.a_u_gate.bias.data, 1, self.timesteps - 1)
            # self.a_w_gate.bias.data.log()
            # self.a_w_gate.bias.data.log()

        self.i_w_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.i_u_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.e_w_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.e_u_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        spatial_h_size = kernel_size
        self.h_padding = spatial_h_size // 2
        self.w_exc = nn.Parameter(torch.empty(hidden_size, hidden_size, spatial_h_size, spatial_h_size))
        self.w_inh = nn.Parameter(torch.empty(hidden_size, hidden_size, spatial_h_size, spatial_h_size))
        
        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_size, eps=1e-03, affine=True, track_running_stats=False) for i in range(2)])

        init.orthogonal_(self.w_inh)
        init.orthogonal_(self.w_exc)
        
        init.orthogonal_(self.i_w_gate.weight)
        init.orthogonal_(self.i_u_gate.weight)
        init.orthogonal_(self.e_w_gate.weight)
        init.orthogonal_(self.e_u_gate.weight)
        
        for bn in self.bn:
            init.constant_(bn.weight, 0.1)
        
        init.constant_(self.alpha, 1.)
        init.constant_(self.mu, 0.)
        # init.constant_(self.alpha, 0.1)
        # init.constant_(self.mu, 1)
        init.constant_(self.gamma, 0.)
        init.constant_(self.w, 1.)
        init.constant_(self.kappa, 1.)

        if self.use_attention:
            self.i_w_gate.bias.data = -self.a_w_gate.bias.data
            self.e_w_gate.bias.data = -self.a_w_gate.bias.data
            self.i_u_gate.bias.data = -self.a_u_gate.bias.data
            self.e_u_gate.bias.data = -self.a_u_gate.bias.data
        else:
            init.uniform_(self.i_w_gate.bias.data, 1, self.timesteps - 1)
            self.i_w_gate.bias.data.log()
            self.i_u_gate.bias.data.log()
            self.e_w_gate.bias.data = -self.i_w_gate.bias.data
            self.e_u_gate.bias.data = -self.i_u_gate.bias.data

    def forward(self, input_, inhibition, excitation,  activ=F.softplus, testmode=False):  # Worked with tanh and softplus
        # Attention gate: filter input_ and excitation
        # att_gate = torch.sigmoid(self.a_w_gate(input_) + self.a_u_gate(excitation))  # Attention Spotlight
        # att_gate = torch.sigmoid(self.a_w_gate(input_) * self.a_u_gate(excitation))  # Attention Spotlight
        if self.use_attention:
            # att_gate = torch.sigmoid(self.a_w_gate(inhibition) + self.a_u_gate(excitation))  # Attention Spotlight -- MOST RECENT WORKING
            att_gate = torch.sigmoid(self.a_w_gate(input_) + self.a_u_gate(excitation))  # Attention Spotlight -- MOST RECENT WORKING

        # Gate E/I with attention immediately
        if self.use_attention:
            gated_input = input_  # * att_gate  # In activ range
            gated_excitation = att_gate * excitation
            gated_inhibition = att_gate  # * inhibition
        else:
            gated_input = input_

        # Compute inhibition
        inh_intx = self.bn[0](F.conv2d(gated_excitation, self.w_inh, padding=self.h_padding))  # in activ range
        # inhibition_hat = activ(gated_input - inh_intx * (self.alpha * gated_inhibition + self.mu))
        # inhibition_hat = activ(gated_input - activ(inh_intx * (self.alpha * gated_inhibition + self.mu)))  # Older version
        inhibition_hat = activ(input_ - activ(inh_intx * (self.alpha * gated_inhibition + self.mu)))

        # Integrate inhibition
        inh_gate = torch.sigmoid(self.i_w_gate(gated_input) + self.i_u_gate(gated_inhibition))
        inhibition = (1 - inh_gate) * inhibition + inh_gate * inhibition_hat  # In activ range

        # Pass to excitatory neurons
        # exc_gate = torch.sigmoid(self.e_w_gate(inhibition) + self.e_u_gate(excitation))
        exc_gate = torch.sigmoid(self.e_w_gate(gated_inhibition) + self.e_u_gate(gated_excitation))
        exc_intx = self.bn[1](F.conv2d(inhibition, self.w_exc, padding=self.h_padding))  # In activ range
        # exc_intx = activ(exc_intx)
        # excitation_hat = activ(self.kappa * inhibition + self.gamma * exc_intx + self.w * inhibition * exc_intx)  # Skip connection OR add OR add by self-sim
        excitation_hat = activ(exc_intx * (self.kappa * inhibition + self.gamma))  # Skip connection OR add OR add by self-sim

        excitation = (1 - exc_gate) * excitation + exc_gate * excitation_hat
        if testmode:
            return inhibition, excitation, att_gate
        else:
            return inhibition, excitation


class FFhGRU(nn.Module):

    def __init__(self, dimensions, timesteps=8, kernel_size=15, jacobian_penalty=False, grad_method='bptt'):
        '''
        '''
        super(FFhGRU, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = dimensions
        self.bn = nn.BatchNorm3d(self.hgru_size, eps=1e-03, track_running_stats=False)
        self.preproc = nn.Conv3d(3, dimensions, kernel_size=1, padding=1 // 2)
        # self.preproc = nn.Conv2d(3, dimensions, kernel_size=1, padding=1 // 2)
        # self.preproc = nn.Parameter(torch.empty((1, dimensions, 1, 1, 1)))
        # init.orthogonal_(self.preproc)
        self.unit1 = hConvGRUCell(
            hidden_size=self.hgru_size,
            kernel_size=kernel_size,
            use_attention=True,
            timesteps=timesteps)
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
        inhibition = torch.zeros((x_shape[0], x_shape[1], x_shape[3], x_shape[4]), requires_grad=False).to(x.device)
        # excitation = xbn.clone()[:, :, 0]
        # inhibition = xbn.clone()[:, :, 0]
        # with torch.no_grad():
        #     inhibition = xbn[:, :, 0].clone()

        # Loop over frames
        states = []
        gates = []
        for t in range(x_shape[2]):
            out = self.unit1(
                input_=xbn[:, :, t],
                inhibition=inhibition,
                excitation=excitation,
                activ=self.nl,
                testmode=testmode)
            if testmode:
                inhibition, excitation, gate = out
                gates.append(gate)  # This should learn to keep the winner
                states.append(self.readout_conv(excitation))  # This should learn to keep the winner
            else:
                inhibition, excitation = out
                # states.append(self.readout_conv(excitation))
                # if t == x_shape[2] - 1:
                #     states.append(self.readout_conv(excitation))  # This should learn to keep the winner

        # Conv 1x1 output
        # if testmode:
        #     out_states = [states[-1]]
        #     out_states.append(x[:, 2, 0][:, None])
        #     output = torch.cat(out_states, 1)  # .reshape(x_shape[0], -1)
        # else:
        # output = torch.cat([self.readout_conv(self.readout_bn(excitation)), x[:, 2, 0][:, None]], 1)

        output = torch.cat([self.readout_conv(excitation), x[:, 2, 0][:, None]], 1)
        # output = torch.cat(states, 1).mean(1, keepdim=True)  # Convert video -> spatial path
        # output = torch.cat([output, x[:, 1:, 0]], 1)  # Paste the start and end goals

        # Potentially combine target_conv + readout_bn into 1
        output = self.target_conv(output)  # output.sum(1, keepdim=True))  # 2 channels -> 1. Is the dot in the target?
        # output = self.target_conv(torch.cat([excitation, x[:, 2, 0][:, None]], 1))
        # # Mini readout network for solving PF without any clutter. RF=32x32
        # output = self.nl(self.target_conv_0(output))
        # output = self.target_pool_0(output)
        # output = self.nl(self.target_conv_1(output))
        # output = self.target_pool_0(output)
        # output = self.nl(self.target_conv_2(output))
        # if sum([*output.size()[2:]]) > 2:
        #     output = F.avg_pool2d(output, kernel_size=output.size()[2:])  # Spatial pool
        output = F.avg_pool2d(output, kernel_size=output.size()[2:])
        # output = self.readout_dense(output.reshape(x_shape[0], -1))  # scale + intercept
        output = output.reshape(x_shape[0], -1)
        output = self.readout_dense(output)
        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        if testmode: return output, torch.stack(states, 1), torch.stack(gates, 1)
        return output, jv_penalty


class FFhGRU_v2(nn.Module):

    def __init__(self, dimensions, timesteps=8, kernel_size=15, jacobian_penalty=False, grad_method='bptt'):
        '''
        '''
        super(FFhGRU_v2, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = dimensions
        # self.bn = nn.BatchNorm3d(self.hgru_size, eps=1e-03, track_running_stats=False)
        self.preproc = nn.Conv3d(3, dimensions, kernel_size=1, padding=1 // 2)
        # self.preproc = nn.Conv2d(3, dimensions, kernel_size=1, padding=1 // 2)
        # self.preproc = nn.Parameter(torch.empty((1, dimensions, 1, 1, 1)))
        # init.orthogonal_(self.preproc)
        self.unit1 = hConvGRUCell(
            hidden_size=self.hgru_size,
            kernel_size=kernel_size,
            use_attention=False,
            timesteps=timesteps)
        # self.bn = nn.BatchNorm2d(self.hgru_size, eps=1e-03, track_running_stats=False)
        # self.readout = nn.Linear(timesteps * self.hgru_size, 1) # the first 2 is for batch size, the second digit is for the dimension
        # self.readout_bn = nn.BatchNorm2d(self.hgru_size, eps=1e-03, track_running_stats=False)
        self.target_conv = nn.Conv2d(dimensions + 2, 1, 1)
        torch.nn.init.zeros_(self.target_conv.bias)
        # self.target_conv_0 = nn.Conv2d(3, 16, 5, padding=0)  # padding=5 // 2)
        # self.target_pool_0 = nn.MaxPool2d(2, 2, padding=0)
        # self.target_conv_1 = nn.Conv2d(16, 16, 5, padding=0)  # padding=7 // 2)
        # self.target_pool_1 = nn.MaxPool2d(2, 2, padding=0)
        # self.target_conv_2 = nn.Conv2d(16, 1, 5, padding=0)  # padding=7 // 2)
        # self.readout_dense = nn.Linear(1, 1)
        # torch.nn.init.zeros_(self.readout_dense.bias)
        self.nl = F.softplus

    def forward(self, x, testmode=False):
        # First step: replicate x over the channel dim self.hgru_size times
        xbn = self.preproc(x)
        # xbn = self.bn(prex)  # This might be hurting me...
        xbn = self.nl(xbn)  # TEST TO SEE IF THE NL STABLIZES

        # Now run RNN
        x_shape = xbn.shape
        excitation = torch.zeros((x_shape[0], x_shape[1], x_shape[3], x_shape[4]), requires_grad=False).to(x.device)
        inhibition = torch.zeros((x_shape[0], x_shape[1], x_shape[3], x_shape[4]), requires_grad=False).to(x.device)
        # with torch.no_grad():
        #     inhibition = xbn[:, :, 0].clone()

        # Loop over frames
        states = []
        gates = []
        for t in range(x_shape[2]):
            out = self.unit1(
                input_=xbn[:, :, t],
                inhibition=inhibition,
                excitation=excitation,
                activ=self.nl,
                testmode=testmode)
            if testmode:
                inhibition, excitation, gate = out
                gates.append(gate)  # This should learn to keep the winner
                states.append(excitation)  # This should learn to keep the winner
            else:
                inhibition, excitation = out
                # states.append(self.readout_conv(excitation))
                # if t == x_shape[2] - 1:
                #     states.append(self.readout_conv(excitation))  # This should learn to keep the winner

        # Potentially combine target_conv + readout_bn into 1
        output = self.target_conv(torch.cat([excitation, x[:, 1:, 0]], 1))  # 2 channels -> 1. Is the dot in the target?
        # # Mini readout network for solving PF without any clutter. RF=32x32
        # output = self.nl(self.target_conv_0(output))
        # output = self.target_pool_0(output)
        # output = self.nl(self.target_conv_1(output))
        # output = self.target_pool_0(output)
        # output = self.nl(self.target_conv_2(output))
        # if sum([*output.size()[2:]]) > 2:
        #     output = F.avg_pool2d(output, kernel_size=output.size()[2:])  # Spatial pool
        output = F.avg_pool2d(output, kernel_size=output.size()[2:])
        # output = self.readout_dense(output.reshape(x_shape[0], -1))  # scale + intercept
        output = output.reshape(x_shape[0], -1)
        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        if testmode: return output, torch.stack(states, 1), torch.stack(gates, 1)
        return output, jv_penalty


class FC(nn.Module):

    def __init__(self, dimensions, timesteps=8, kernel_size=15, jacobian_penalty=False, grad_method='bptt'):
        '''
        '''
        super(FC, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = dimensions
        self.bn = nn.BatchNorm3d(self.hgru_size, eps=1e-03, track_running_stats=False)
        self.preproc = nn.Conv3d(3, dimensions, kernel_size=1, padding=1 // 2)
        self.readout = nn.Linear(64*32*32*32, 1) # the first 2 is for batch size, the second digit is for the dimension
        # self.readout = nn.Linear(timesteps * self.hgru_size, 1) # the first 2 is for batch size, the second digit is for the dimension

    def forward(self, x, testmode=False):
        # First step: replicate x over the channel dim self.hgru_size times
        # x = x.repeat(1, self.hgru_size, 1, 1, 1)
        x = self.preproc(x)
        x = self.bn(x)
        x_shape = x.shape
        x = self.readout(x.reshape(x_shape[0], -1))
        jv_penalty = torch.tensor([1]).float().cuda()
        return x, jv_penalty


class ClockHGRU(nn.Module):

    def __init__(self, dimensions, timesteps=8, kernel_size=15, jacobian_penalty=False, grad_method='bptt', clock_type="dynamic"):
        '''
        '''
        super(ClockHGRU, self).__init__()
        assert clock_type in ["fixed", "dynamic"]
        self.clock_type = clock_type
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = dimensions

        self.preproc = nn.Conv3d(3, dimensions, kernel_size=1, padding=1 // 2)
        self.unit1 = ClockHConvGRUCell(
            hidden_size=self.hgru_size,
            kernel_size=kernel_size,
            clock_type=clock_type,
            timesteps=timesteps)
        # self.bn = nn.BatchNorm3d(self.hgru_size, eps=1e-03, track_running_stats=False)
        self.bn = nn.BatchNorm3d(self.hgru_size, eps=1e-03, track_running_stats=True)
        self.readout_conv = nn.Conv2d(dimensions, 1, 1)
        # self.readout_dense = nn.Linear(262144, 1)
        # self.readout_dense = nn.Linear(1048576, 1)
        self.readout_dense = nn.Linear(65536, 1)

    def forward(self, x, testmode=False):
        # First step: replicate x over the channel dim self.hgru_size times
        x = self.preproc(x)
        x = self.bn(x)

        # Now run RNN
        x_shape = x.shape
        excitation = torch.zeros((x_shape[0], x_shape[1], x_shape[3], x_shape[4]), requires_grad=False, device=x.device)
        inhibition = torch.zeros((x_shape[0], x_shape[1], x_shape[3], x_shape[4]), requires_grad=False, device=x.device)

        # Loop over frames
        state_2nd_last = None
        states = []
        for t in range(x_shape[2]):
            inhibition, excitation = self.unit1(
                input_=x[:, :, t],
                step=t,  # torch.as_tensor(t, device=x.device).float(),
                inhibition=inhibition,
                excitation=excitation)
            if t == self.timesteps - 2 and "rbp" in self.grad_method:
                state_2nd_last = excitation
            elif t == self.timesteps - 1:
                last_state = excitation
            # states.append(F.max_pool2d(self.readout_conv(excitation), kernel_size=excitation.size()[2:]))
            states.append(self.readout_conv(excitation))

        # Use the final frame to make a decision
        # output = self.bn(torch.cat(states, -1))  # Stack as timesteps
        # output = F.max_pool2d(output, kernel_size=2)

        # Fully connected output, gets down to 0.35-ish loss
        # output = self.readout_dense(torch.cat(states, -1).reshape(x_shape[0], -1))

        # Conv 1x1 output
        output = self.readout_dense(torch.cat(states,-1).reshape(x_shape[0], -1))

        # output = self.readout_conv(output)
        # output = F.avg_pool2d(output, kernel_size=output.size()[2:])
        # output = self.readout_dense(output.reshape(x_shape[0], -1))
        # output = excitation.mean(dim=(2, 3))
        # output = torch.stack(states, -1)
        # output = output.reshape(x_shape[0], -1)
        # output = self.readout(output)

        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        # mu = 0.9
        # double_neg = False
        # if self.training and self.jacobian_penalty:
        #     if pen_type == 'l1':
        #         norm_1_vect = torch.ones_like(last_state)
        #         norm_1_vect.requires_grad = False
        #         jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
        #                                       create_graph=self.jacobian_penalty, allow_unused=True)[0]
        #         jv_penalty = (jv_prod - mu).clamp(0) ** 2
        #         if double_neg is True:
        #             neg_norm_1_vect = -1 * norm_1_vect.clone()
        #             jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[neg_norm_1_vect], retain_graph=True,
        #                                           create_graph=True, allow_unused=True)[0]
        #             jv_penalty2 = (jv_prod - mu).clamp(0) ** 2
        #             jv_penalty = jv_penalty + jv_penalty2
        #     elif pen_type == 'idloss':
        #         norm_1_vect = torch.rand_like(last_state).requires_grad_()
        #         jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
        #                                       create_graph=True, allow_unused=True)[0]
        #         jv_penalty = (jv_prod - norm_1_vect) ** 2
        #         jv_penalty = jv_penalty.mean()
        #         if torch.isnan(jv_penalty).sum() > 0:
        #             raise ValueError('Nan encountered in penalty')
        if testmode: return output, states
        return output, jv_penalty

