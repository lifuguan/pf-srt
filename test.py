import piqa,lpips
import torch
import torch.optim as optim
import numpy as np
from srt.utils.common import mse2psnr, reduce_dict
from collections import defaultdict

import os, sys, argparse, math
import yaml, json
from tqdm import tqdm

from srt.data import worker_init_fn
from srt.data import get_dataset
from srt.checkpoint import Checkpoint
from srt.utils.nerf import rotate_around_z_axis_torch, get_camera_rays, transform_points_torch, get_extrinsic_torch
from srt.model import SRT
from srt.trainer import SRTTrainer


def get_camera_rays_render(camera_pos, **kwargs):
    rays = get_camera_rays(camera_pos[0], **kwargs)
    return np.expand_dims(rays, 0)

def lerp(x, y, t):
    return x + (y-x) * t

def easeout(t):
    return -0.5 * t**2 + 1.5 * t

def apply_fade(t, t_fade=0.2):
    v_max = 1. / (1. - t_fade)
    acc = v_max / t_fade
    if t <= t_fade:
        return 0.5 * acc * t**2
    pos_past_fade = 0.5 * acc * t_fade**2
    if t <= 1. - t_fade:
        return pos_past_fade + v_max * (t - t_fade)
    else:
        return 1. - 0.5 * acc * (t - 1.)**2

def get_camera_closeup(camera_pos, rays, t, zoomout=1., closeup=0.2, z_closeup=0.1, lookup=3.):
    orig_camera_pos = camera_pos[0] * zoomout
    orig_track_point = torch.zeros_like(orig_camera_pos)
    orig_ext = get_extrinsic_torch(orig_camera_pos, track_point=orig_track_point, fourxfour=True)

    final_camera_pos = closeup * orig_camera_pos
    final_camera_pos[2] = z_closeup * orig_camera_pos[2]
    final_track_point = orig_camera_pos + (orig_track_point - orig_camera_pos) * lookup
    final_track_point[2] = 0.

    cur_camera_pos = lerp(orig_camera_pos, final_camera_pos, t)
    cur_camera_pos[2] = lerp(orig_camera_pos[2], final_camera_pos[2], easeout(t))
    cur_track_point = lerp(orig_track_point, final_track_point, t)

    new_ext = get_extrinsic_torch(cur_camera_pos, track_point=cur_track_point, fourxfour=True)

    cur_rays = transform_points_torch(rays, torch.inverse(new_ext) @ orig_ext, translate=False)
    return cur_camera_pos.unsqueeze(0), cur_rays


def rotate_camera(camera_pos, rays, t):
    theta = math.pi * 2 * t
    camera_pos = rotate_around_z_axis_torch(camera_pos, theta)
    rays = rotate_around_z_axis_torch(rays, theta)

    return camera_pos, rays


def evalute(val_dataset):
    device = torch.device("cuda" if is_cuda else "cpu")
    render_path = os.path.join(out_dir, 'render', args.name)
    if os.path.exists(render_path):
        print(f'Warning: Path {render_path} exists. Contents will be overwritten.')

    os.makedirs(render_path, exist_ok=True)
    subdirs = ['renders', 'depths']
    for d in subdirs:
        os.makedirs(os.path.join(render_path, d), exist_ok=True)
    
    eval_lists = defaultdict(list)

    for eval_idx,eval_dataset_idx in enumerate(tqdm(torch.linspace(0,len(val_dataset)-1,min(args.n_eval,len(val_dataset))).int())):
        data = val_dataset[eval_dataset_idx]

        input_images = data.get('input_images').to(device).unsqueeze(0)
        input_camera_pos = data.get('input_camera_pos').to(device).unsqueeze(0)
        input_rays = data.get('input_rays').to(device).unsqueeze(0)
        target_pixels = data.get('target_pixels').to(device)
        target_camera_pos = data.get('target_camera_pos').to(device)
        target_rays = data.get('target_rays').to(device)
        with torch.no_grad():
            z = model.encoder(input_images, input_camera_pos, input_rays)
            pred_pixels, extras = model.decoder(z, target_camera_pos, target_rays, **render_kwargs)

        # mse = ((pred_pixels - target_pixels)**2).mean((1, 2))
        # psnr = mse2psnr(mse)
        psnr = piqa.PSNR()(pred_pixels.clip(0,1).contiguous(), target_pixels.clip(0,1).contiguous())
        print("psnr: ", psnr)
        eval_lists['psnr'].append(psnr)

        pred_pixels = pred_pixels.reshape(target_pixels.shape[0], 76, 250, 3)
        target_pixels = data.get('target_rgb').reshape(target_pixels.shape[0], 76, 250, 3)
        pred_pixels = pred_pixels.cpu().numpy()
        pred_pixels = (pred_pixels * 255.).astype(np.uint8)
        continue

    eval_dict = {k: torch.cat(v, 0) for k, v in eval_lists.items()}
    eval_dict = reduce_dict(eval_dict, average=True)  # Average across processes
    eval_dict = {k: v.mean().item() for k, v in eval_dict.items()}  # Average across batch_size    
    print('Evaluation results:')
    print(eval_dict)          

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Render a video of a scene.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--num-frames', type=int, default=360, help='Number of frames to render.')
    parser.add_argument('--sceneid', type=int, default=0, help='Id of the scene to render.')
    parser.add_argument('--sceneid-start', type=int, help='Id of the scene to render.')
    parser.add_argument('--sceneid-stop', type=int, help='Id of the scene to render.')
    parser.add_argument('--height', type=int, help='Rendered image height in pixels. Defaults to input image height.')
    parser.add_argument('--width', type=int, help='Rendered image width in pixels. Defaults to input image width.')
    parser.add_argument('--name', type=str, help='Name of this sequence.')
    parser.add_argument('--motion', type=str, default='rotate', help='Type of sequence.')
    parser.add_argument('--sharpen', action='store_true', help='Square density values for sharper surfaces.')
    parser.add_argument('--parallel', action='store_true', help='Wrap model in DataParallel.')
    parser.add_argument('--train', action='store_true', help='Use training data.')
    parser.add_argument('--fade', action='store_true', help='Add fade in/out.')
    parser.add_argument('--it', type=int, help='Iteration of the model to load.')
    parser.add_argument('--render-kwargs', type=str, help='Renderer kwargs as JSON dict')
    parser.add_argument('--novideo', action='store_true', help="Don't compile rendered images into video")
    parser.add_argument('--n_eval', type=int,default=int(1e8),help="Number of eval samples to run")

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.CLoader)
    print('configs loaded')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    out_dir = os.path.dirname(args.config)
    exp_name = os.path.basename(out_dir)
    if args.name is None:
        args.name = args.motion
    if args.render_kwargs is not None:
        render_kwargs = json.loads(args.render_kwargs)
    else:
        render_kwargs = dict()

    model = SRT(cfg['model']).to(device)
    model.eval()

    mode = 'train' if args.train else 'val'
    eval_dataset = get_dataset(mode, cfg['data'])

    render_kwargs |= eval_dataset.render_kwargs

    optimizer = optim.Adam(model.parameters())
    trainer = SRTTrainer(model, optimizer, cfg, device, out_dir, eval_dataset.render_kwargs)

    checkpoint = Checkpoint(out_dir, encoder=model.encoder, decoder=model.decoder, optimizer=optimizer)
    if args.it is not None:
        load_dict = checkpoint.load(f'model_{args.it}.pt')
    else:
        load_dict = checkpoint.load('model.pt')

    evalute(eval_dataset)

