import torch.nn as nn
import torch.nn.functional as F
import torch
try:
    from performer_pytorch import Performer
except:
    print("Failed to import Performer.")
try:
    from lambda_networks import LambdaLayer
except:
    print("Failed to import lambdanets.")
try:
    from timesformer_pytorch import TimeSformer
except:
    print("Failed to import Timesformers.")


class TransformerModel(nn.Module):

    def __init__(self, dimensions, timesteps=8, kernel_size=15, jacobian_penalty=False, grad_method='bptt'):
        '''
        '''
        super(TransformerModel, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = dimensions
        # self.bn = nn.BatchNorm3d(self.hgru_size, eps=1e-03, track_running_stats=False)
        # self.preproc = nn.Conv3d(3, dimensions, kernel_size=1, padding=1 // 2)
        self.Transformer = TimeSformer(
            image_size = 32,
            patch_size = 32,
            num_classes = 1,
            num_frames = timesteps,
            dim = dimensions,
            heads = 4,
            depth = 2,
            dim_head = dimensions,
            ff_dropout = 0.1,
            attn_dropout = 0.1
        )
        self.nl = F.softplus

    def forward(self, x, testmode=False):
        # First step: replicate x over the channel dim self.hgru_size times
        # xbn = self.preproc(x)
        # xbn = self.nl(xbn)  # TEST TO SEE IF THE NL STABLIZES

        x = x.permute(0, 2, 1, 3, 4)  # Expects xbn BCTHW -> BTCHW
        output = self.Transformer(x)
        jv_penalty = torch.tensor([1]).float().cuda()
        return output, jv_penalty

    def forwarddense(self, x, testmode=False):
        # First step: replicate x over the channel dim self.hgru_size times
        xbn = self.preproc(x)
        xbn = self.nl(xbn)  # TEST TO SEE IF THE NL STABLIZES

        # Now run Performer
        # excitation = self.Performer(xbn.permute(0, 2, 3, 4, 1).reshape(xbn.shape[0], -1, xbn.shape[1]))
        excitation = self.Performer(xbn.permute(0, 2, 3, 4, 1).view(xbn.shape[0], -1, xbn.shape[1]))
        output = excitation.view(xbn.shape[0], xbn.shape[2], xbn.shape[3] * xbn.shape[4], xbn.shape[1])  # BTH*WC
        output = output.mean(2).view(xbn.shape[0], -1)
        output = self.readout_dense(output)
        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        if testmode: return output, torch.stack(states, 1), torch.stack(gates, 1)
        return output, jv_penalty


class PerformerModel(nn.Module):

    def __init__(self, dimensions, timesteps=8, kernel_size=15, jacobian_penalty=False, grad_method='bptt'):
        '''
        '''
        super(PerformerModel, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = 32  # dimensions
        self.bn = nn.BatchNorm3d(self.hgru_size, eps=1e-03, track_running_stats=False)
        self.preproc = nn.Conv3d(3, self.hgru_size, kernel_size=1, padding=1 // 2)
        self.Performer = Performer(
            dim = self.hgru_size,
            dim_head = self.hgru_size,
            depth = 1,  # 1
            heads = 4,  # 4
            causal = True,
            feature_redraw_interval=1000,  # default is 1000
        )

        self.target_conv = nn.Conv2d(self.hgru_size + 1, 1, 5, padding=5 // 2)
        torch.nn.init.zeros_(self.target_conv.bias)
        self.readout_dense = nn.Linear(1, 1)
        # self.readout_dense = nn.Linear(2048, 1)
        self.nl = F.softplus

    def forward(self, x, testmode=False):
        # First step: replicate x over the channel dim self.hgru_size times
        xbn = self.preproc(x)
        # xbn = self.nl(xbn)  # TEST TO SEE IF THE NL STABLIZES
        xbn_shape = xbn.shape

        # Now run Performer
        excitation = self.Performer(xbn.permute(0, 2, 3, 4, 1).reshape(xbn.shape[0], -1, xbn.shape[1]))
        # (Pdb) xbn.permute(0, 1, 3, 4, 2).reshape(xbn.shape[0], -1, xbn.shape[2]).shape
        # torch.Size([6, 32768, 64])
        # (Pdb) xbn.view(xbn.shape[0], xbn.shape[1], -)
        # *** SyntaxError: invalid syntax
        # (Pdb) xbn.view(xbn.shape[0], xbn.shape[1], -1).shape
        # torch.Size([6, 32, 65536])
        ######
        # # xbn = xbn.permute(0, 1, 3, 4, 2).reshape(xbn_shape[0], -1, xbn_shape[2])  # Drew style less expensive
        # xbn = xbn.view(xbn.shape[0], xbn.shape[1], -1).permute(0, 2, 1)  # JK style super expensive  B (HWT) C
        # # excitation = self.Performer(xbn.permute(0, 2, 3, 4, 1).view(xbn.shape[0], -1, xbn.shape[1]))
        # excitation = self.Performer(xbn)
        # output = excitation.view(xbn_shape[0], xbn_shape[2], xbn_shape[3], xbn_shape[4], xbn_shape[1])
        # output = output[:, -1].permute(0, 3, 1, 2)  # Final timestep  and BHWC -> BCHW
        ####
        # output = excitation.view(xbn_shape[0], xbn_shape[1], xbn_shape[3], xbn_shape[4], xbn_shape[2])  # BTHWC
        output = excitation.view(xbn_shape[0], xbn_shape[2], xbn_shape[3], xbn_shape[4], xbn_shape[1])[:, -1].permute(0, 3, 1, 2)
        # output = output[..., -1]
        output = torch.cat([output, x[:, 2, 0][:, None]], 1)

        # Potentially combine target_conv + readout_bn into 1
        output = self.target_conv(output)  # output.sum(1, keepdim=True))  # 2 channels -> 1. Is the dot in the target?
        output = F.avg_pool2d(output, kernel_size=output.size()[2:])
        output = output.reshape(xbn.shape[0], -1)
        output = self.readout_dense(output)
        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        if testmode: return output, torch.stack(states, 1), torch.stack(gates, 1)
        return output, jv_penalty

    def forwarddense(self, x, testmode=False):
        # First step: replicate x over the channel dim self.hgru_size times
        xbn = self.preproc(x)
        xbn = self.nl(xbn)  # TEST TO SEE IF THE NL STABLIZES

        # Now run Performer
        # excitation = self.Performer(xbn.permute(0, 2, 3, 4, 1).reshape(xbn.shape[0], -1, xbn.shape[1]))
        excitation = self.Performer(xbn.permute(0, 2, 3, 4, 1).view(xbn.shape[0], -1, xbn.shape[1]))
        output = excitation.view(xbn.shape[0], xbn.shape[2], xbn.shape[3] * xbn.shape[4], xbn.shape[1])  # BTH*WC
        output = output.mean(2).view(xbn.shape[0], -1)
        output = self.readout_dense(output)
        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        if testmode: return output, torch.stack(states, 1), torch.stack(gates, 1)
        return output, jv_penalty


class LambdaModel(nn.Module):

    def __init__(self, dimensions, timesteps=8, kernel_size=15, jacobian_penalty=False, grad_method='bptt'):
        '''
        '''
        super(LambdaModel, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = dimensions
        self.bn = nn.BatchNorm3d(self.hgru_size, eps=1e-03, track_running_stats=False)
        self.preproc = nn.Conv3d(3, dimensions, kernel_size=1, padding=1 // 2)
        self.Lambda = LambdaLayer(
            dim = dimensions * timesteps,
            dim_out = dimensions * timesteps,
            n=32,  # Size of image
            heads = 4,  # 4
            dim_k = 8,
            dim_u = 4
        )

        self.target_conv = nn.Conv2d(2048 + 1, 1, 5, padding=5 // 2)
        torch.nn.init.zeros_(self.target_conv.bias)
        self.readout_dense = nn.Linear(1, 1)
        # self.readout_dense = nn.Linear(2048, 1)
        self.nl = F.softplus

    def forward(self, x, testmode=False):
        # First step: replicate x over the channel dim self.hgru_size times
        xbn = self.preproc(x)
        xbn = self.nl(xbn)  # TEST TO SEE IF THE NL STABLIZES

        # Now run Performer
        # excitation = self.Performer(xbn.permute(0, 2, 3, 4, 1).reshape(xbn.shape[0], -1, xbn.shape[1]))
        xbn = xbn.view(xbn.shape[0], -1, xbn.shape[3], xbn.shape[4])
        excitation = self.Lambda(xbn)
        output = torch.cat([excitation, x[:, 2, 0][:, None]], 1)

        # Potentially combine target_conv + readout_bn into 1
        output = self.target_conv(output)  # output.sum(1, keepdim=True))  # 2 channels -> 1. Is the dot in the target?
        output = F.avg_pool2d(output, kernel_size=output.size()[2:])
        output = output.reshape(xbn.shape[0], -1)
        output = self.readout_dense(output)
        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        if testmode: return output, torch.stack(states, 1), torch.stack(gates, 1)
        return output, jv_penalty

    def forwarddense(self, x, testmode=False):
        # First step: replicate x over the channel dim self.hgru_size times
        xbn = self.preproc(x)
        xbn = self.nl(xbn)  # TEST TO SEE IF THE NL STABLIZES

        # Now run Performer
        # excitation = self.Performer(xbn.permute(0, 2, 3, 4, 1).reshape(xbn.shape[0], -1, xbn.shape[1]))
        excitation = self.Performer(xbn.permute(0, 2, 3, 4, 1).view(xbn.shape[0], -1, xbn.shape[1]))
        output = excitation.view(xbn.shape[0], xbn.shape[2], xbn.shape[3] * xbn.shape[4], xbn.shape[1])  # BTH*WC
        output = output.mean(2).view(xbn.shape[0], -1)
        output = self.readout_dense(output)
        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        if testmode: return output, torch.stack(states, 1), torch.stack(gates, 1)
        return output, jv_penalty


