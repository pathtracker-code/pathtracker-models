import os
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from models.hgrucleanSEG import hConvGRU
from models.FFnet import FFConvNet
from models import InT
import imageio
from torchvision.models import video
from models import nostridetv as nostride_video
from models import nostridetv_cc as nostride_video_cc
from models import nostridetv_cc_smallest as nostride_video_cc_small
from models import nostridetv_positions as nostride_video_pos
from models.slowfast_utils import slowfast, slowfast_nl
from models import transformers
from models import kys
try:
    from models import resnet_TSM as rntsm
except:
    print("Failed to import spatial sampler.")
try:
    from tqdm import tqdm
except:
    print("Failed to import tqdm.")


TORCHVISION = ['r3d', 'mc3', 'r2plus1', 'nostride_r3d', 'nostride_r3d_pos']
SLOWFAST = ['slowfast', 'slowfast_nl']
ALL_DATASETS = [ 
    {"dist": 14, "speed": 1, "length": 64},
    {"dist": 14, "speed": 1, "length": 128},
    {"dist": 14, "speed": 1, "length": 32},
    {"dist": 14, "speed": 2, "length": 64},
    {"dist": 14, "speed": 4, "length": 64},
    {"dist": 0, "speed": 1, "length": 64},
    {"dist": 5, "speed": 1, "length": 64},
    {"dist": 25, "speed": 1, "length": 64},
]

def model_step(model, imgs, model_name, test=False):
    """Pass imgs through the model."""
    if model_name in TORCHVISION:
        output = model.forward(imgs)
        jv_penalty = torch.tensor([1]).float().cuda() 
    elif model_name in SLOWFAST:
        # frames = F.interpolate(imgs, 224, mode='trilinear', align_corners=True)  # F.interpolate(imgs, 224)
        frames = imgs
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        ALPHA = 4 
        slow_pathway = torch.index_select(
            frames,
            2,  # 1
            torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // ALPHA, device=imgs.device
            ).long(),
            )  # Note that this usually operates per-exemplar. Here we do per batch.
        frame_list = [slow_pathway, fast_pathway]
        output = model.forward(frame_list)
        jv_penalty = torch.tensor([1]).float().cuda()
    else:
        if test:
            output, states, gates = model.forward(imgs, testmode=True)
            return output, states, gates
        else:
            output, jv_penalty = model.forward(imgs)
    if test:
        return output, None, None
    else:
        return output, jv_penalty


def model_selector(args, timesteps, device, fb_kernel_size=7, dimensions=32):
    """Select a model."""
    if args.model == 'InT':
        print("Init model InT ", args.algo, 'penalty: ', args.penalty)
        model = InT.InT(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            grad_method='bptt')
    elif args.model == 'InT_no_inh':  # Exc-only
        print("Init model InT ", args.algo, 'penalty: ', args.penalty)
        model = InT.InT(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            no_inh=True,
            grad_method='bptt')
    elif args.model == 'InT_no_mult':  # Reverse mely
        print("Init model InT ", args.algo, 'penalty: ', args.penalty)
        model = InT.InT(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            lesion_alpha=True,  # No div inh
            lesion_gamma=True,  # No add exc
            grad_method='bptt')
    elif args.model == 'InT_no_add':  # Mely style
        print("Init model InT ", args.algo, 'penalty: ', args.penalty)
        model = InT.InT(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            lesion_mu=True,  # No sub inh
            lesion_kappa=True,  # No Mult exc
            grad_method='bptt')
    elif args.model == 'InT_mult_add':  # Div/Mult only
        print("Init model InT ", args.algo, 'penalty: ', args.penalty)
        model = InT.InT(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            lesion_alpha=False,  # Div inh
            lesion_gamma=True,  # Mult add
            lesion_mu=True,  # Sub inh
            lesion_kappa=False,  # Mult exc
            grad_method='bptt')
    elif args.model == 'InT_only_add':  # Sub/Add only
        print("Init model InT ", args.algo, 'penalty: ', args.penalty)
        model = InT.InT(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            lesion_alpha=True,  # No div inh
            lesion_gamma=False,  # No add exc
            lesion_mu=False,
            lesion_kappa=True,
            grad_method='bptt')
    elif args.model == 'InT_tanh':  # No dales
        print("Init model InT ", args.algo, 'penalty: ', args.penalty)
        model = InT.InT(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            nl=F.tanh,
            grad_method='bptt')
    elif args.model == 'gru':
        model = kys.GRU(
            dimensions=dimensions * 2,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            grad_method='bptt')
    elif args.model == 'fc':
        print("Init model InT ", args.algo, 'penalty: ', args.penalty)
        model = InT.FC(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            grad_method='bptt')
    elif args.model == 'performer':
        model = transformers.PerformerModel(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            grad_method='bptt')
    elif args.model == 'lambda':
        model = transformers.LambdaModel(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            grad_method='bptt')
    elif args.model == 'timesformer':
        model = transformers.TransformerModel(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            grad_method='bptt')
    elif args.model == 'slowfast':
        model = slowfast()
    elif args.model == 'slowfast_nl':
        model = slowfast_nl()
    elif args.model == 'r3d':
        model = video.r3d_18(pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif args.model == 'rntsm':
        model = rntsm.resnet50(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=1)
    elif args.model == 'nostride_r3d':
        model = nostride_video.r3d_18(pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif args.model == 'nostride_r3d_cc':
        model = nostride_video_cc.r3d_18(pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
    elif args.model == 'nostride_r3d_pos':
        model = nostride_video_pos.r3d_18(pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif args.model == 'nostride_video_cc_small':
        model = nostride_video_cc_small.r3d_18(pretrained=args.pretrained, timesteps=timesteps)
        num_ftrs = model.fc.in_features
    elif args.model == 'mc3':
        model = video.mc3_18(pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif args.model == 'r2plus1':
        model = video.r2plus1d_18(pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    else:
        raise NotImplementedError("Model not found.")
    return model


def prepare_data(imgs, target, args, device, disentangle_channels, use_augmentations=False):
    """Prepare the data for training or eval."""
    imgs = imgs.numpy()
    imgs = imgs.transpose(0,4,1,2,3)
    target = torch.from_numpy(np.vectorize(ord)(target.numpy()))
    target = target.to(device, dtype=torch.float)
    imgs = imgs / 255.  # Normalize to [0, 1]

    if disentangle_channels:
        mask = imgs.sum(1).round()
        proc_imgs = np.zeros_like(imgs)
        proc_imgs[:, 1] = (mask == 1).astype(imgs.dtype)
        proc_imgs[:, 2] = (mask == 2).astype(imgs.dtype)
        thing_layer = (mask == 3).astype(imgs.dtype)
        proc_imgs[:, 0] = thing_layer
    else:
        proc_imgs = imgs
    if use_augmentations:
        imgs = transforms(proc_imgs)
        imgs = np.stack(imgs, 0)
    else:
        imgs = proc_imgs
    imgs = torch.from_numpy(proc_imgs)
    imgs = imgs.to(device, dtype=torch.float)
    if args.pretrained:
        mu = torch.tensor([0.43216, 0.394666, 0.37645], device=device)[None, :, None, None, None]
        stddev = torch.tensor([0.22803, 0.22145, 0.216989], device=device)[None, :, None, None, None]
        imgs = (imgs - mu) / stddev

    if "_cc" in args.model and args.model != "nostride_video_cc_small":
        img_shape = imgs.shape
        hh, ww = torch.meshgrid(torch.arange(1, img_shape[3] + 1, device=imgs.device, dtype=imgs.dtype), torch.arange(1, img_shape[4] + 1, device=imgs.device, dtype=imgs.dtype))
        hh = hh[None, None, None].repeat(img_shape[0], 1, img_shape[2], 1, 1)
        ww = ww[None, None, None].repeat(img_shape[0], 1, img_shape[2], 1, 1)
        imgs = torch.cat([imgs, hh, ww], 1)
    return imgs, target


def load_ckpt(model, model_path):
    checkpoint = torch.load(model_path)
    # Check if "module" is the first part of the key
    # check = checkpoint['state_dict'].keys()[0]
    sd = checkpoint['state_dict']
    # if "module" in check and not args.parallel:
    #     new_sd = {}
    #     for k, v in sd.items():
    #         new_sd[k.replace("module.", "")] = v
    #     sd = new_sd
    model.load_state_dict(sd)
    return model


def plot_results(states, imgs, target, output, timesteps, gates=None, prep_gifs=False, results_folder=None, show_fig=False):
    states = states.detach().cpu().numpy()
    gates = gates.detach().cpu().numpy()
    cols = (timesteps / 8) + 1
    rng = np.arange(0, timesteps, 8)
    rng = np.concatenate((np.arange(0,timesteps,8), [timesteps-1]))
    img = imgs.cpu().numpy()
    # from matplotlib import pyplot as plt
    sel = target.float().reshape(-1, 1) == (output > 0).float()
    sel = sel.cpu().numpy()
    sel = np.where(sel)[0]
    sel = sel[0]
    fig = plt.figure()
    for idx, i in enumerate(rng):
        plt.subplot(3, cols, idx + 1)
        plt.axis("off")
        plt.imshow(img[sel, :, i].transpose(1, 2, 0))
        plt.title("Img")
        plt.subplot(3, cols, idx + 1 + cols)
        plt.axis("off")
        plt.imshow((gates[sel, i].squeeze() ** 2).mean(0))
        plt.title("Attn")
        plt.subplot(3, cols, idx + 1 + cols + (cols - 1))
        plt.title("Activity")
        plt.axis("off")
        plt.imshow(np.abs(states[sel, i].squeeze()))
    plt.suptitle("Batch acc: {}, Prediction: {}, Label: {}".format((target.reshape(-1).float() == (output.reshape(-1) > 0).float()).float().mean(), output[sel].cpu(), target[sel]))
    if results_folder is not None:
        plt.savefig(os.path.join(results_folder, "random_selection.pdf"))
    if show_fig:
        plt.show()
    plt.close(fig)

    if prep_gifs:
        assert isinstance(prep_gifs, int), "prep_gifs is an integer that says how many gifs to prepare"
        assert results_folder is not None, "if prepping gifs, also pass a results folder."
        for g in tqdm(range(prep_gifs), total=prep_gifs, desc="Making gifs"):
            gif_dir = os.path.join(results_folder, "gif_{}".format(g))
            os.makedirs(gif_dir, exist_ok=True)
            filenames = []
            min_gate, max_gate = None, None  # (gates[g] ** 2).reshape(img.shape[2], -1).min() * 0.75, (gates[g] ** 2).reshape(img.shape[2], -1).max() * 0.75
            min_act, max_act = None, None  # (states[g] ** 2).reshape(img.shape[2], -1).min() * 0.75, (states[g] ** 2).reshape(img.shape[2], -1).max() * 0.75
            for idx, i in tqdm(enumerate(range(img.shape[2])), total=img.shape[2], desc="Writing gif images"):  # Loop over all timesteps
                fig = plt.figure(dpi=100)
                plt.subplot(1, 3, 1)
                plt.axis("off")
                plt.imshow(img[g, :, i].transpose(1, 2, 0))
                plt.title("Img")
                plt.subplot(1, 3, 2)
                plt.axis("off")
                plt.imshow((gates[g, i].squeeze() ** 2).mean(0), vmin=min_gate, vmax=max_gate)
                plt.title("Attn")
                plt.subplot(1, 3, 3)
                plt.title("Activity")
                plt.axis("off")
                # plt.imshow(np.abs(states[g, i].squeeze()), vmin=min_act, vmax=max_act)
                plt.imshow(states[g, i].squeeze() ** 2, vmin=min_act, vmax=max_act)
                plt.suptitle("Prediction: {}, Label: {}".format(output[g].cpu() > 0., target[g].cpu() == 1.))
                out_path = os.path.join(gif_dir, "{}.png".format(idx))
                plt.savefig(out_path)
                plt.close(fig)
                filenames.append(out_path)
            # Now produce gif
            gif_path = os.path.join(gif_dir, "{}.gif".format(g))
            with imageio.get_writer(gif_path, mode='I') as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
                    os.remove(filename)


LOCAL = "/gpfs/data/tserre/data/tracking/tfrecords"

def dataset_selector(dist, speed, length, optical_flow=False):
    """Organize the datasets here."""
    stem = "tfrecords"
    if optical_flow:
        stem = "tfrecords_optic_flow"
    if dist == 14 and speed == 1 and length == 64:
        lp = os.path.join(LOCAL, "downsampled_constrained_red_blue_datasets_64_32_32_separate_channels/14_dist/tfrecords/")
        if os.path.exists(lp):
            print("Loading data from local storage.")
            return lp, 64, 20000, 20000
        else:
            return '/cifs/data/tserre_lrs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32_separate_channels/5_dist/tfrecords/', 64, 20000, 20000
        return '/cifs/data/tserre_lrs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32_separate_channels/14_dist/tfrecords/', 64, 20000, 20000

    elif dist == 14 and speed == 1 and length == 32:
        lp = os.path.join(LOCAL, "downsampled_constrained_red_blue_datasets_32_32_32_separate_channels/14_dist/tfrecords/")
        if os.path.exists(lp):
            print("Loading data from local storage.")
            return lp, 64, 20000, 20000
        else:
            return '/cifs/data/tserre_lrs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_32_32_32_separate_channels/14_dist/tfrecords/', 32, 20000, 20000
    elif dist == 5 and speed == 1 and length == 32:
        lp = os.path.join(LOCAL, "downsampled_constrained_red_blue_datasets_32_32_32_separate_channels/5_dist/tfrecords/")
        if os.path.exists(lp):
            print("Loading data from local storage.")
            return lp, 64, 20000, 20000
        else:
            return '/cifs/data/tserre_lrs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_32_32_32_separate_channels/5_dist/tfrecords/', 32, 20000, 20000
    elif dist == 0 and speed == 1 and length == 32:
        lp = os.path.join(LOCAL, "downsampled_constrained_red_blue_datasets_32_32_32_separate_channels/0_dist/tfrecords/")
        if os.path.exists(lp):
            print("Loading data from local storage.")
            return lp, 64, 20000, 20000
        else:
            return '/cifs/data/tserre_lrs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_32_32_32_separate_channels/0_dist/tfrecords/', 32, 20000, 20000

    elif dist == 14 and speed == 1 and length == 128:
        return '/cifs/data/tserre_lrs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_128_32_32_separate_channels/14_dist/tfrecords/', 128, 20000, 20000
    elif dist == 14 and speed == 1 and length == 32:
        return '/cifs/data/tserre_lrs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_32_32_32_separate_channels/14_dist/tfrecords/', 32, 20000, 20000
    elif dist == 25 and speed == 1 and length == 64:
        return '/cifs/data/tserre_lrs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32_separate_channels/25_dist/tfrecords/', 64, 20000, 20000
    elif dist == 14 and speed == 2 and length == 64:
        return '/cifs/data/tserre_lrs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32_separate_channels_skip_param_2/14_dist/tfrecords/', 64, 20000, 20000
    elif dist == 0 and speed == 1 and length == 64:
        lp = os.path.join(LOCAL, "downsampled_constrained_red_blue_datasets_64_32_32_separate_channels/0_dist/{}/".format(stem))
        if os.path.exists(lp):
            print("Loading data from local storage.")
            return lp, 64, 20000, 20000
        else:
            return '/cifs/data/tserre_lrs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32_separate_channels/0_dist/tfrecords/', 64, 20000, 20000
    elif dist == 5 and speed == 1 and length == 64:
        lp = os.path.join(LOCAL, "downsampled_constrained_red_blue_datasets_64_32_32_separate_channels/5_dist/{}/".format(stem))
        if os.path.exists(lp):
            print("Loading data from local storage.")
            return lp, 64, 20000, 20000
        else:
            return "/cifs/data/tserre_lrs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32_separate_channels/5_dist/tfrec/{}/".format(stem), 64, 20000, 20000
    elif dist == 14 and speed == 4 and length == 64:
        return '/cifs/data/tserre_lrs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32_separate_channels_skip_param_4/14_dist/tfrecords/', 64, 20000, 20000


def get_datasets():
    return ALL_DATASETS

