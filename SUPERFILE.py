from __future__ import print_function
import argparse
import yaml
import git
from tensorboardX import SummaryWriter
import datetime
import time
import os
import logging
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import copy
import math
from tqdm import tqdm
from scipy.interpolate import interp1d
import h5py
import ipdb
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from scipy.ndimage import zoom
from torch.distributions import Normal

# import warnings
# warnings.filterwarnings('error')
# torch.set_printoptions(profile="default")

import tensorboardX.x2num
from tensorboardX.x2num import check_nan as original_check_nan
# Monkey patching
def check_nan_patched(array):
    tmp = np.sum(array)
    if np.isnan(tmp) or np.isinf(tmp):
        raise ValueError('NaN or Inf found in input tensor.')
    return array

# Replace original function with patched version
tensorboardX.x2num.check_nan = check_nan_patched

# # Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)

class Logger(object):
    """
    Hooks for print statements and tensorboard logging
    """

    def __init__(self, configs):

        self.configs = configs

        time_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M")
        prefix_str = time_str + "_" + configs["notes"]
        if configs["dev"]:
            prefix_str = "dev_" + prefix_str

        self.log_folder = os.path.join(self.configs["logging_folder"], prefix_str)
        self.tb_prefix = prefix_str

        # self.setup_checks()
        self.create_folder_structure()
        self.setup_loggers()
        # self.dump_init_info()

    def create_folder_structure(self):
        """
        Creates the folder structure for logging. Subfolders can be added here
        """
        base_dir = self.log_folder
        sub_folders = ["runs", "models"]

        if not os.path.exists(self.configs["logging_folder"]):
            os.mkdir(self.configs["logging_folder"])

        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        for sf in sub_folders:
            if not os.path.exists(os.path.join(base_dir, sf)):
                os.mkdir(os.path.join(base_dir, sf))

    def setup_loggers(self):
        """
        Sets up a logger that logs to both file and stdout
        """
        log_path = os.path.join(self.log_folder, "log.log")

        self.print_logger = logging.getLogger()
        self.print_logger.setLevel(
            getattr(logging, self.configs["log_level"].upper(), None)
        )
        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)]
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s:%(lineno)d - %(asctime)s - %(message)s"
        )
        for h in handlers:
            h.setFormatter(formatter)
            self.print_logger.addHandler(h)

        # Setup Tensorboard
        self.tb = SummaryWriter(os.path.join(self.log_folder, "runs", self.tb_prefix))

    # def setup_checks(self):
    #     """
    #     Verifies that all changes have been committed
    #     Verifies that hashes match (if continuation)
    #     """
    #     repo = git.Repo(search_parent_directories=True)
    #     sha = repo.head.object.hexsha


    #     # Test for continuation
    #     if self.configs["continuation"]:
    #         self.log_folder = self.configs["logging_folder"]
    #         with open(os.path.join(self.log_folder, "log.log"), "r") as old_log:
    #             for line in old_log:
    #                 find_str = "Git hash"
    #                 if line.find(find_str) != -1:
    #                     old_sha = line[line.find(find_str) + len(find_str) + 2 : -4]
    #                     assert sha == old_sha

    def dump_init_info(self):
        """
        Saves important info for replicability
        """
        if not self.configs["continuation"]:
            self.configs["logging_folder"] = self.log_folder
        else:
            self.print("=" * 80)
            self.print("Continuing log")
            self.print("=" * 80)

        # repo = git.Repo(search_parent_directories=True)
        # sha = repo.head.object.hexsha

        # self.print("Git hash: {}".format(sha))
        self.print("Dumping YAML file")
        self.print("Configs: ", yaml.dump(self.configs))

        # Save the start of every run
        if "start_weights" not in self.configs:
            self.configs["start_weights"] = []
        self.configs["start_weights"].append(self.configs["load"])

        with open(os.path.join(self.log_folder, "configs.yml"), "w") as outfile:
            yaml.dump(self.configs, outfile)
            self.tb.add_text("hyperparams", str(self.configs))

    def end_itr(self, weights_path):
        """
        Perform all operations needed at end of iteration
        1). Save configs with latest weights
        """
        self.configs["latest_weights"] = weights_path
        with open(os.path.join(self.log_folder, "configs.yml"), "w") as outfile:
            yaml.dump(self.configs, outfile)

    def print(self, *args):
        """
        Wrapper for print statement
        """
        self.print_logger.info(args)


class SensorFusion(nn.Module):
    """
    #
        Regular SensorFusionNetwork Architecture
        Number of parameters:
        Inputs:
            image:   batch_size x 3 x 128 x 128
            force:   batch_size x 6 x 32
            proprio: batch_size x 8
            action:  batch_size x action_dim
    """

    def __init__(
        self, device, z_dim, z_depth, action_dim, encoder=False, deterministic=False
    ):
        super().__init__()

        self.z_dim = z_dim
        self.z_depth = z_depth
        self.encoder_bool = encoder
        self.device = device
        self.deterministic = deterministic

        # print("SensorFusion z_dim: " + str(z_dim))
        # print("SensorFusion z_depth: " + str(z_depth))

        # zero centered, 1 std normal distribution
        self.z_prior_m = torch.nn.Parameter(
            torch.zeros(1, z_dim), requires_grad=False
        )
        self.z_prior_v = torch.nn.Parameter(
            torch.ones(1, z_dim), requires_grad=False
        )
        self.z_prior = (self.z_prior_m, self.z_prior_v)

        # -----------------------
        # Modality Encoders
        # -----------------------
        self.img_encoder = ImageEncoder(self.z_dim, self.z_depth)
        self.depth_encoder = DepthEncoder(self.z_dim, self.z_depth)
        self.frc_encoder = ForceEncoder(self.z_dim, self.z_depth)
        self.proprio_encoder = ProprioEncoder(self.z_dim, self.z_depth)

        # -----------------------
        # Action Encoders
        # -----------------------
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # -----------------------
        # action fusion network
        # -----------------------
        adjusted = int(self.z_dim*z_depth/2)
        # print("adjusted: " + str(adjusted))

        self.st_fusion_fc1 = nn.Sequential(
            # nn.Linear(32 + self.z_dim, 128), nn.LeakyReLU(0.1, inplace=True)
            nn.Linear(32 + adjusted, 128), nn.LeakyReLU(0.1, inplace=True)
        )
        self.st_fusion_fc2 = nn.Sequential(
            nn.Linear(128, self.z_dim), nn.LeakyReLU(0.1, inplace=True)
        )

        if deterministic:
            # -----------------------
            # modality fusion network
            # -----------------------
            # 4 Total modalities each (2 * z_dim)
            self.fusion_fc1 = nn.Sequential(
                nn.Linear(4 * 2 * self.z_dim, 128), nn.LeakyReLU(0.1, inplace=True)
            )
            self.fusion_fc2 = nn.Sequential(
                nn.Linear(self.z_dim, self.z_dim), nn.LeakyReLU(0.1, inplace=True)
            )

        # -----------------------
        # weight initialization
        # -----------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_encoder(self, vis_in, frc_in, proprio_in, depth_in, action_in, z_depth):

        # batch size
        batch_dim = vis_in.size()[0]
        ###print("batch_dim: " + str(batch_dim))

        image = rescaleImage(vis_in)
        depth = filter_depth(depth_in)

        # Get encoded outputs
        img_out, img_out_convs = self.img_encoder(image, z_depth)
        depth_out, depth_out_convs = self.depth_encoder(depth, z_depth) ###
        frc_out = self.frc_encoder(frc_in)
        proprio_out = self.proprio_encoder(proprio_in, z_depth) ###

        # print("batch_dim: " + str(batch_dim))
        # print("image: " + str(image))
        # print("depth: " + str(depth))
        # print("img_out: " + str(img_out))
        # print("img_out_convs: " + str(img_out_convs))
        # print("depth_out: " + str(depth_out))
        # print("depth_out_convs: " + str(depth_out_convs))
        # print("frc_out: " + str(frc_out))
        # print("proprio_out: " + str(proprio_out))

        # print("self.deterministic: " + str(self.deterministic))

        if self.deterministic:
            # multimodal embedding
            mm_f1 = torch.cat([img_out, frc_out, proprio_out, depth_out], 1).squeeze()
            mm_f2 = self.fusion_fc1(mm_f1)
            z = self.fusion_fc2(mm_f2)

        else:
            # Encoder priors
            mu_prior, var_prior = self.z_prior

            # print("z_depth: " + str(z_depth))
            # print("mu_prior: " + str(mu_prior))
            # print("SIZE OF mu_prior: " + str(mu_prior.shape))
            # print("duplicate function: " + str((duplicate(mu_prior, batch_dim)).shape))

            # Duplicate prior parameters for each data point in the batch
            mu_prior_resized = duplicate(mu_prior, batch_dim).unsqueeze(2)
            var_prior_resized = duplicate(var_prior, batch_dim).unsqueeze(2)

            # print("mu_prior_resized: " + str(mu_prior_resized.shape))
            # print("var_prior_resized: " + str(var_prior_resized.shape))

            # Modality Mean and Variances
            mu_z_img, var_z_img = gaussian_parameters(img_out, dim=1)
            mu_z_frc, var_z_frc = gaussian_parameters(frc_out, dim=1)
            mu_z_proprio, var_z_proprio = gaussian_parameters(proprio_out, dim=1)
            mu_z_depth, var_z_depth = gaussian_parameters(depth_out, dim=1)

            # Tile distribution parameters using concatonation

            pos2 = int(self.z_dim*z_depth/2)
            ###print("pos2: " + str(pos2))

            mu_prior_resized = torch.zeros((batch_dim, pos2, 1), dtype=torch.float32).to(self.device) # .to('mps:0')
            var_prior_resized = torch.zeros((batch_dim, pos2, 1), dtype=torch.float32).to(self.device) # .to('mps:0')

            # print("SHAPE OF mu_z_img: " + str(mu_z_img.shape) + str(mu_z_img)) #########
            # print("SHAPE OF mu_z_frc: " + str(mu_z_frc.shape) + str(mu_z_frc)) #########
            # print("SHAPE OF mu_z_proprio: " + str(mu_z_proprio.shape) + str(mu_z_proprio)) #########
            # print("SHAPE OF mu_z_depth: " + str(mu_z_depth.shape) + str(mu_z_proprio)) #########
            # print("SHAPE OF mu_prior_resized: " + str(mu_z_proprio.shape) + str(mu_z_proprio)) #########

            m_vect = torch.cat([mu_z_img, mu_z_frc, mu_z_proprio, mu_z_depth, mu_prior_resized], dim=2 )
            var_vect = torch.cat([var_z_img, var_z_frc, var_z_proprio, var_z_depth, var_prior_resized], dim=2 )

            m_vect = remove_zeros(m_vect)
            var_vect = remove_zeros(var_vect)

            # print("m_vect: " + str(m_vect.size()) + str(m_vect))
            # print("var_vect: " + str(var_vect.size()) + str(var_vect))

            # Fuse modalities mean / variances using product of experts
            mu_z, var_z = product_of_experts(m_vect, var_vect) # => contain both 0 values 

            # Sample Gaussian to get latent
            z = sample_gaussian(mu_z, var_z, self.device)

        # print("z: " + str(z.shape) + str(z)) # nan
        # print("m_vect: " + str(m_vect.shape) + str(m_vect))
        # print("var_vect: " + str(var_vect.shape) + str(var_vect))
        # print("mu_z: " + str(mu_z.shape) + str(mu_z)) # nan
        # print("var_z: " + str(var_z.shape) + str(var_z))

        if self.encoder_bool or action_in is None:
            if self.deterministic:
                return img_out, frc_out, proprio_out, depth_out, z
            else:
                return img_out_convs, img_out, frc_out, proprio_out, depth_out, z
        else:
            # action embedding
            act_feat = self.action_encoder(action_in)

            # state-action feature
            # print("act_feat: " + str(act_feat))
            mm_act_f1 = torch.cat([z, act_feat], 1)
            # print("mm_act_f1: " + str(mm_act_f1.shape))
            mm_act_f2 = self.st_fusion_fc1(mm_act_f1)
            mm_act_feat = self.st_fusion_fc2(mm_act_f2)

            if self.deterministic:
                return img_out_convs, mm_act_feat, z
            else:
                return img_out_convs, mm_act_feat, z, mu_z, var_z, mu_prior, var_prior

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]


class SensorFusionSelfSupervised(SensorFusion):
    """
        Regular SensorFusionNetwork Architecture
        Inputs:
            image:   batch_size x 3 x 128 x 128
            force:   batch_size x 6 x 32
            proprio: batch_size x 8
            action:  batch_size x action_dim
    """

    def __init__(
        #self, device, z_dim=128, action_dim=4, encoder=False, deterministic=False
        self, device, z_dim, z_depth, action_dim, encoder=False, deterministic=False
    ):

        super().__init__(device, z_dim, z_depth, action_dim, encoder, deterministic)

        self.deterministic = deterministic

        # -----------------------
        # optical flow predictor
        # -----------------------
        self.optical_flow_decoder = OpticalFlowDecoder(z_dim)

        # -----------------------
        # ee delta decoder
        # -----------------------
        self.ee_delta_decoder = EeDeltaDecoder(z_dim, action_dim)

        # -----------------------
        # pairing decoder
        # -----------------------
        adjusted = int(z_dim*z_depth/2)
        self.pair_fc = nn.Sequential(nn.Linear(adjusted, 1))

        # -----------------------
        # contact decoder
        # -----------------------
        self.contact_fc = nn.Sequential(nn.Linear(self.z_dim, 1))

        # -----------------------
        # weight initialization
        # -----------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(
        self,
        vis_in,
        frc_in,
        proprio_in,
        depth_in,
        action_in,
    ):

        # print("self.encoder_bool: " + str(self.encoder_bool))
        # print("action_in: " + str(action_in))
        # print("self.deterministic: " + str(self.deterministic))

        if self.encoder_bool:
            # returning latent space representation if model is set in encoder mode
            z = self.forward_encoder(vis_in, frc_in, proprio_in, depth_in, action_in, self.z_depth)
            return z

        elif action_in is None:
            z = self.forward_encoder(vis_in, frc_in, proprio_in, depth_in, None, self.z_depth)
            pair_out = self.pair_fc(z)
            return pair_out

        else:
            if self.deterministic:
                img_out_convs, mm_act_feat, z = self.forward_encoder(
                    vis_in, frc_in, proprio_in, depth_in, action_in, self.z_depth
                )
            else:
                img_out_convs, mm_act_feat, z, mu_z, var_z, mu_prior, var_prior = self.forward_encoder(
                    vis_in,
                    frc_in,
                    proprio_in,
                    depth_in,
                    action_in,
                    self.z_depth
                )

        # ---------------- Training Objectives ----------------

        # tile state-action features and append to conv map
        batch_dim = mm_act_feat.size(0)  # batch size
        tiled_feat = mm_act_feat.view(batch_dim, self.z_dim, 1, 1).expand(-1, -1, 2, 2)

        # -------------------------------------#
        # Pairing / Contact / EE Delta Decoder #
        # -------------------------------------#
        pair_out = self.pair_fc(z)
        contact_out = self.contact_fc(mm_act_feat)
        ee_delta_out = self.ee_delta_decoder(mm_act_feat)

        # -------------------------#
        # Optical Flow Prediction #
        # -------------------------#
        optical_flow2, optical_flow2_mask = self.optical_flow_decoder(
            tiled_feat, img_out_convs
        )

        if self.deterministic:
            return (
                pair_out,
                contact_out,
                optical_flow2,
                optical_flow2_mask,
                ee_delta_out,
                z,
            )
        else:
            return (
                pair_out,
                contact_out,
                optical_flow2,
                optical_flow2_mask,
                ee_delta_out,
                z,
                mu_z,
                var_z,
                mu_prior,
                var_prior,
                self.z_depth
            )

def detach_var(var):
    """Detaches a var from torch

    Args:
        var (torch var): Torch variable that requires grad

    Returns:
        TYPE: numpy array
    """
    return var.cpu().detach().numpy()


def set_seeds(seed, use_cuda):
    """Set Seeds

    Args:
        seed (int): Sets the seed for numpy, torch and random
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def quaternion_to_euler(x, y, z, w):

    # t0 = +2.0 * (w * x + y * z)
    # t1 = +1.0 - 2.0 * (x * x + y * y)
    # X = np.arctan2(t0, t1)

    # t2 = +2.0 * (w * y - z * x)
    # t2 = +1.0 if t2 > +1.0 else t2
    # t2 = -1.0 if t2 < -1.0 else t2
    # Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = np.arctan2(t3, t4)

    Z = -Z - np.pi / 2

    return Z


def compute_accuracy(pred, target):
    pred_1 = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
    target_1 = torch.where(target > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
    batch_size = target.size()[0] * 1.0

    num_correct = 1.0 * torch.where(
        pred_1 == target_1, torch.ones_like(pred), torch.zeros_like(pred)
    ).sum().float()

    accuracy = num_correct / batch_size
    return accuracy


def rescaleImage(image, output_size=128, scale=1 / 255.0):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    image_transform = image * scale
    # torch.from_numpy(img.transpose((0, 3, 1, 2))).float()
    return image_transform.transpose(1, 3).transpose(2, 3)


def log_normal(x, m, v):

    log_prob = -((x - m) ** 2 / (2 * v)) - 0.5 * torch.log(2 * math.pi * v)

    return log_prob


def enlarge_tensor_by_factor(tensor, factor):
    # Check if the tensor has the right number of dimensions (2)
    if len(tensor.shape) != 2:
        raise ValueError("Input tensor must be 2D.")
    if factor % 1 != 0:
        raise ValueError("Factor must be an integer.")

    # Convert the tensor to numpy array
    array = tensor.cpu().numpy()

    # Create a new array to hold the interpolated data
    new_shape = (array.shape[0], array.shape[1]*factor)
    new_array = np.zeros(new_shape, dtype=np.float32)

    # Perform the interpolation for each batch
    for i in range(array.shape[0]):
        # Define the x coordinates for the original data points
        x = np.arange(array.shape[1])
        # Define the x coordinates for the interpolated data points
        x_new = np.linspace(0, array.shape[1] - 1, new_shape[1])
        # Create a function to perform the interpolation
        f = interp1d(x, array[i], kind='quadratic')
        # Perform the interpolation and store the result in the new array
        new_array[i] = f(x_new)

    # Convert the new numpy array back to tensor
    tensor = torch.from_numpy(new_array)
    return tensor


def kl_normal(qm, qv, pm, pv):
    # SHAPE OF mu_z: torch.Size([64, 256])
    # SHAPE OF var_z: torch.Size([64, 256])
    # SHAPE OF mu_prior.squeeze(0): torch.Size([128])
    # SHAPE OF var_prior.squeeze(0): torch.Size([128])
    # kl_normal(mu_z, var_z, mu_prior.squeeze(0), var_prior.squeeze(0))
        # [64, 256],[64, 256],      [128],               [128]
    # torch.log(var_prior.squeeze(0)) - torch.log(var_z) + var_z / var_prior.squeeze(0) + (mu_z - mu_prior.squeeze(0)).pow(2) / var_prior.squeeze(0) - 1
                            # [128]   -       [64, 256] + [64, 256]/[128]
    element_wise = 0.5 * (
        torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1
    )
    kl = element_wise.sum(-1)
    return kl


def augment_val(val_filename_list, filename_list):

    filename_list1 = copy.deepcopy(filename_list)
    val_filename_list1 = []

    for name in tqdm(val_filename_list):
        filename = name[:-8]
        found = True

        if filename[-2] == "_":
            file_number = int(filename[-1])
            filename = filename[:-1]
        else:
            file_number = int(filename[-2:])
            filename = filename[:-2]

        if file_number < 10:
            comp_number = 19
            filename1 = filename + str(comp_number) + "_1000.h5"
            while (filename1 not in filename_list1) and (
                filename1 not in val_filename_list1
            ):
                comp_number += -1
                filename1 = filename + str(comp_number) + "_1000.h5"
                if comp_number < 0:
                    found = False
                    break
        else:
            comp_number = 0
            filename1 = filename + str(comp_number) + "_1000.h5"
            while (filename1 not in filename_list1) and (
                filename1 not in val_filename_list1
            ):
                comp_number += 1
                filename1 = filename + str(comp_number) + "_1000.h5"
                if comp_number > 19:
                    found = False
                    break

        if found:
            if filename1 in filename_list1:
                filename_list1.remove(filename1)

            if filename1 not in val_filename_list:
                val_filename_list1.append(filename1)

    val_filename_list1 += val_filename_list

    return val_filename_list1, filename_list1


def flow2rgb(flow_map, max_value=None):
    global args
    _, h, w = flow_map.shape
    # flow_map[:,(flow_map[0] == 0) & (flow_map[1] == 0)] = float('nan')
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map / max_value
    else:
        normalized_flow_map = flow_map / (np.abs(flow_map).max())
    rgb_map[:, :, 0] += normalized_flow_map[0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[:, :, 2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)


def scene_flow2rgb(flow_map):
    global args

    flow_map = np.where(flow_map > 1e-6, flow_map, np.zeros_like(flow_map))

    indices1 = np.nonzero(flow_map[0, :, :])
    indices2 = np.nonzero(flow_map[1, :, :])
    indices3 = np.nonzero(flow_map[2, :, :])

    normalized_flow_map = np.zeros_like(flow_map)

    divisor_1 = 0
    divisor_2 = 0
    divisor_3 = 0

    if np.array(indices1).size > 0:
        divisor_1 = (
            flow_map[0, :, :][indices1].max() - flow_map[0, :, :][indices1].min()
        )

    if np.array(indices2).size > 0:
        divisor_2 = (
            flow_map[1, :, :][indices2].max() - flow_map[1, :, :][indices2].min()
        )

    if np.array(indices3).size > 0:
        divisor_3 = (
            flow_map[2, :, :][indices3].max() - flow_map[2, :, :][indices3].min()
        )

    if divisor_1 > 0:
        normalized_flow_map[0, :, :][indices1] = (
            flow_map[0, :, :][indices1] - flow_map[0, :, :][indices1].min()
        ) / divisor_1

    if divisor_2 > 0:
        normalized_flow_map[1, :, :][indices2] = (
            flow_map[1, :, :][indices2] - flow_map[1, :, :][indices2].min()
        ) / divisor_2

    if divisor_3 > 0:
        normalized_flow_map[2, :, :][indices3] = (
            flow_map[2, :, :][indices3] - flow_map[2, :, :][indices3].min()
        ) / divisor_3

    return normalized_flow_map


def point_cloud2rgb(flow_map):
    global args

    flow_map = np.where(flow_map > 5e-4, flow_map, np.zeros_like(flow_map))

    flow_map = np.tile(
        np.expand_dims(np.sqrt(np.sum(np.square(flow_map), axis=0)), axis=0), (3, 1, 1)
    )
    return flow_map


def EPE(input_flow, target_flow, device, sparse=False, mean=True):
    # torch.cuda.init()

    EPE_map = torch.norm(target_flow.cpu() - input_flow.cpu(), 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

        EPE_map = EPE_map[~mask.data]
    if mean:
        return EPE_map.mean().to(device)
    else:
        return (EPE_map.sum() / batch_size).to(device)


def realEPE(output, target, device, sparse=False):
    b, _, h, w = target.size()

    # upsampled_output = nn.functional.upsample(output, size=(h, w), mode="bilinear")
    upsampled_output = nn.functional.interpolate(output, size=(h, w), mode="bilinear")
    return EPE(upsampled_output, target, device, sparse, mean=True)


def realAAE(output, target, device, sparse=False):
    b, _, h, w = target.size()
    # upsampled_output = nn.functional.upsample(output, size=(h, w), mode="bilinear")
    upsampled_output = nn.functional.interpolate(output, size=(h, w), mode="bilinear")
    return AAE(upsampled_output, target, device, sparse, mean=True)


def AAE(input_flow, target_flow, device, sparse=False, mean=True):
    b, _, h, w = target_flow.size()
    ones = torch.ones([b, 1, h, w])
    target = torch.cat((target_flow.cpu(), ones), 1)
    inp = torch.cat((input_flow.cpu(), ones), 1)
    target = target.permute(0, 2, 3, 1).contiguous().view(b * h * w, -1)
    inp = inp.permute(0, 2, 3, 1).contiguous().view(b * h * w, -1)

    target = target.div(torch.norm(target, dim=1, keepdim=True).expand_as(target))
    inp = inp.div(torch.norm(inp, dim=1, keepdim=True).expand_as(inp))

    dot_prod = torch.bmm((target.view(b * h * w, 1, -1)), inp.view(b * h * w, -1, 1))
    AAE_map = torch.acos(torch.clamp(dot_prod, -1, 1))

    return AAE_map.mean().to(device)

class MultimodalManipulationDataset(Dataset):
    """Multimodal Manipulation dataset."""

    def __init__(
        self,
        filename_list,
        transform=None,
        episode_length=50,
        training_type="selfsupervised",
        n_time_steps=1,
        action_dim=4,
        pairing_tolerance=0.06
    ):
        """
        Args:
            hdf5_file (handle): h5py handle of the hdf5 file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_path = filename_list
        self.transform = transform
        self.episode_length = episode_length
        self.training_type = training_type
        self.n_time_steps = n_time_steps
        self.dataset = {}
        self.action_dim = action_dim
        self.pairing_tolerance = pairing_tolerance

        self._config_checks()
        self._init_paired_filenames()

    def __len__(self):
        return len(self.dataset_path) * (self.episode_length - self.n_time_steps)

    def __getitem__(self, idx):

        list_index = idx // (self.episode_length - self.n_time_steps)
        dataset_index = idx % (self.episode_length - self.n_time_steps)
        filename = self.dataset_path[list_index][:-8]

        file_number, filename = self._parse_filename(filename)

        unpaired_filename, unpaired_idx = self.paired_filenames[(list_index, dataset_index)]

        if dataset_index >= self.episode_length - self.n_time_steps - 1:
            dataset_index = np.random.randint(
                self.episode_length - self.n_time_steps - 1
            )

        sample = self._get_single(
            self.dataset_path[list_index],
            list_index,
            unpaired_filename,
            dataset_index,
            unpaired_idx,
        )
        return sample

    def _get_single(
        self, dataset_name, list_index, unpaired_filename, dataset_index, unpaired_idx
    ):

        dataset = h5py.File(dataset_name, "r", swmr=True, libver="latest")
        unpaired_dataset = h5py.File(unpaired_filename, "r", swmr=True, libver="latest")

        if self.training_type == "selfsupervised":

            image = dataset["image"][dataset_index]
            depth = dataset["depth_data"][dataset_index]
            proprio = dataset["proprio"][dataset_index][:8]
            force = dataset["ee_forces_continuous"][dataset_index]

            if image.shape[0] == 3:
                image = np.transpose(image, (2, 1, 0))

            if depth.ndim == 2:
                depth = depth.reshape((128, 128, 1))

            flow = np.array(dataset["optical_flow"][dataset_index])
            flow_mask = np.expand_dims(
                np.where(
                    flow.sum(axis=2) == 0,
                    np.zeros_like(flow.sum(axis=2)),
                    np.ones_like(flow.sum(axis=2)),
                ),
                2,
            )

            unpaired_image = image
            unpaired_depth = depth
            unpaired_proprio = unpaired_dataset["proprio"][unpaired_idx][:8]
            unpaired_force = unpaired_dataset["ee_forces_continuous"][unpaired_idx]

            sample = {
                "image": image,
                "depth": depth,
                "flow": flow,
                "flow_mask": flow_mask,
                "action": dataset["action"][dataset_index + 1],
                "force": force,
                "proprio": proprio,
                "ee_yaw_next": dataset["proprio"][dataset_index + 1][:self.action_dim],
                "contact_next": np.array(
                    [dataset["contact"][dataset_index + 1].sum() > 0]
                # ).astype(np.float),
                ).astype(np.float64),
                "unpaired_image": unpaired_image,
                "unpaired_force": unpaired_force,
                "unpaired_proprio": unpaired_proprio,
                "unpaired_depth": unpaired_depth,
            }

        dataset.close()
        unpaired_dataset.close()

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _init_paired_filenames(self):
        """
        Precalculates the paired filenames.
        Imposes a distance tolerance between paired images
        """
        tolerance = self.pairing_tolerance

        all_combos = set()

        self.paired_filenames = {}
        for list_index in tqdm(range(len(self.dataset_path)), desc="pairing_files"):
            filename = self.dataset_path[list_index]
            file_number, _ = self._parse_filename(filename[:-8])

            dataset = h5py.File(filename, "r", swmr=True, libver="latest")

            for idx in range(self.episode_length - self.n_time_steps):

                proprio_dist = None
                while proprio_dist is None or proprio_dist < tolerance:
                    # Get a random idx, file that is not the same as current
                    unpaired_dataset_idx = np.random.randint(self.__len__())
                    unpaired_filename, unpaired_idx, _ = self._idx_to_filename_idx(unpaired_dataset_idx)

                    while unpaired_filename == filename:
                        unpaired_dataset_idx = np.random.randint(self.__len__())
                        unpaired_filename, unpaired_idx, _ = self._idx_to_filename_idx(unpaired_dataset_idx)

                    with h5py.File(unpaired_filename, "r", swmr=True, libver="latest") as unpaired_dataset:
                        proprio_dist = np.linalg.norm(dataset['proprio'][idx][:3] - unpaired_dataset['proprio'][unpaired_idx][:3])

                self.paired_filenames[(list_index, idx)] = (unpaired_filename, unpaired_idx)
                all_combos.add((unpaired_filename, unpaired_idx))

            dataset.close()

    def _idx_to_filename_idx(self, idx):
        """
        Utility function for finding info about a dataset index

        Args:
            idx (int): Dataset index

        Returns:
            filename (string): Filename associated with dataset index
            dataset_index (int): Index of data within that file
            list_index (int): Index of data in filename list
        """
        list_index = idx // (self.episode_length - self.n_time_steps)
        dataset_index = idx % (self.episode_length - self.n_time_steps)
        filename = self.dataset_path[list_index]
        return filename, dataset_index, list_index

    def _parse_filename(self, filename):
        """ Parses the filename to get the file number and filename"""
        if filename[-2] == "_":
            file_number = int(filename[-1])
            filename = filename[:-1]
        else:
            file_number = int(filename[-2:])
            filename = filename[:-2]

        return file_number, filename

    def _config_checks(self):
        if self.training_type != "selfsupervised":
            raise ValueError(
                "Training type not supported: {}".format(self.training_type)
            )

class ProcessFlow(object):
    """Process optical flow into a pyramid.
    Args:
        pyramid_scale (list): scaling factors to downsample
            the spatial pyramid
    """

    def __init__(self, pyramid_scales=[2, 4, 8]):
        assert isinstance(pyramid_scales, list)
        self.pyramid_scales = pyramid_scales

    def __call__(self, sample):
        # subsampling to create small flow images
        for scale in self.pyramid_scales:
            scaled_flow = sample['flow'][::scale, ::scale]
            sample['flow{}'.format(scale)] = scaled_flow
        return sample

class ProcessForce(object):
    """Truncate a time series of force readings with a window size.
    Args:
        window_size (int): Length of the history window that is
            used to truncate the force readings
    """

    def __init__(self, window_size, key='force', tanh=False):
        assert isinstance(window_size, int)
        self.window_size = window_size
        self.key = key
        self.tanh = tanh

    def __call__(self, sample):
        force = sample[self.key]
        force = force[-self.window_size:]
        if self.tanh:
            force = np.tanh(force)  # remove very large force readings
        sample[self.key] = force.transpose()
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device=None):
        self.device = device

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        # transpose flow into 2 x H x W
        for k in sample.keys():
            if k.startswith('flow'):
                sample[k] = sample[k].transpose((2, 0, 1))

        # convert numpy arrays to pytorch tensors
        new_dict = dict()
        for k, v in sample.items():
            if self.device is None:
                # torch.tensor(v, device = self.device, dtype = torch.float32)
                new_dict[k] = torch.FloatTensor(v)
            else:
                new_dict[k] = torch.from_numpy(v).float()

        return new_dict

class selfsupervised:
    def __init__(self, configs, logger):

        # ------------------------
        # Sets seed and cuda
        # ------------------------
        use_cuda = configs["cuda"] and torch.cuda.is_available()

        self.configs = configs
        self.logger = logger
        # self.device = torch.device("cuda" if use_cuda else "cpu")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else("cuda" if torch.cuda.is_available() else "cpu"))

        if (self.device == "mps") or (self.device == "cpu"):
            logger.print("Let's use", torch.cuda.device_count(), "GPUs!")

        set_seeds(configs["seed"], use_cuda)

        # model
        self.model = SensorFusionSelfSupervised(
            device=self.device,
            encoder=configs["encoder"],
            deterministic=configs["deterministic"],
            z_dim=configs["zdim"],
            z_depth=configs["zdepth"],
            action_dim=configs["action_dim"],
        ).to(self.device)

        # print("selfsupervised z_depth: " + str(configs["zdepth"]))

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.configs["lr"],
            betas=(self.configs["beta1"], 0.999),
            weight_decay=0.0,
        )

        self.deterministic = configs["deterministic"]
        self.encoder = configs["encoder"]

        # losses
        self.loss_ee_pos = nn.MSELoss()
        self.loss_contact_next = nn.BCEWithLogitsLoss()
        self.loss_optical_flow_mask = nn.BCEWithLogitsLoss()
        self.loss_reward_prediction = nn.MSELoss()
        self.loss_is_paired = nn.BCEWithLogitsLoss()
        self.loss_dynamics = nn.MSELoss()

        # validation set variables
        self.val_contact_accuracy = 0.0
        self.val_paired_accuracy = 0.0

        # test set variables
        self.test_flow_loss = 0.0
        self.test_paired_accuracy = 0.0
        self.test_contact_accuracy = 0.0

        # Weights for loss
        self.alpha_optical_flow = 10.0 * configs["opticalflow"]
        self.alpha_optical_flow_mask = 1.0
        self.alpha_kl = 0.05
        self.alpha_contact = 1.0 * configs["contact"]
        self.alpha_pair = 0.5 * configs["pairing"]
        self.alpha_ee_fut = 1.0 * configs["eedelta"]

        # Weights for input
        self.alpha_vision = configs["vision"]
        self.alpha_depth = configs["depth"]
        self.alpha_proprio = configs["proprio"]
        self.alpha_force = configs["force"]

        # Global Counts For Logging
        self.global_cnt = {"train": 0, "val": 0}

        # ------------------------
        # Handles Initialization
        # ------------------------
        if configs["load"]:
            self.load_model(configs["load"])

        self._init_dataloaders()

    def train(self):

        for i_epoch in tqdm(range(self.configs["max_epoch"])):
            # ---------------------------
            # Train Step
            # ---------------------------
            self.logger.print("Training epoch #{}...".format(i_epoch))
            self.model.train()

            for i_iter, sample_batched in tqdm(enumerate(self.dataloaders["val"])):

                t_st = time.time()
                self.optimizer.zero_grad()

                # print("sample_batched: " + str(sample_batched))

                loss, mm_feat, results, image_packet = self.loss_calc(sample_batched)
                # print("loss_1: " + str(loss)) # nan
                # print("mm_feat :" + str(mm_feat)) # nan
                # print("results :" + str(results))
                # print("image_packet :" + str(image_packet)) # nan

                loss.backward()
                self.optimizer.step()


                # print("loss_2: " + str(loss))
                # print("results: " + str(results))
                # print("self.global_cnt[train]: " + str(self.global_cnt["train"]))
                # print("t_st: " + str(t_st))
                self.record_results(loss, results, self.global_cnt["train"], t_st)

                if self.global_cnt["train"] % self.configs["img_record_n"] == 0:
                    self.logger.print(
                        "processed {} mini-batches...".format(self.global_cnt["train"])
                    )
                    self._record_image(image_packet, self.global_cnt["train"])

                self.global_cnt["train"] += 1

            if self.configs["val_ratio"] != 0:
                self.validate(i_epoch)

            # ---------------------------
            # Save weights
            # ---------------------------
            ckpt_path = os.path.join(
                self.logger.log_folder, "models", "weights_itr_{}.ckpt".format(i_epoch)
            )
            self.logger.print("checkpoint path: ", ckpt_path)
            self.logger.print("Saving checkpoint after epoch #{}".format(i_epoch))

            torch.save(self.model.state_dict(), ckpt_path)
            self.logger.end_itr(ckpt_path)

    def validate(self, i_epoch):
        self.logger.print(
            "calculating validation results after #{} epochs".format(i_epoch)
        )

        self.val_contact_accuracy = 0.0
        self.val_paired_accuracy = 0.0

        for i_iter, sample_batched in enumerate(self.dataloaders["val"]):
            self.model.eval()

            loss_val, mm_feat_val, results_val, image_packet_val = self.loss_calc(
                sample_batched
            )

            flow_loss, contact_loss, is_paired_loss, contact_accuracy, is_paired_accuracy, ee_delta_loss, kl = (
                results_val
            )

            self.val_contact_accuracy += contact_accuracy.item() / self.val_len_data
            self.val_paired_accuracy += is_paired_accuracy.item() / self.val_len_data

            if i_iter == 0:
                self._record_image(
                    image_packet_val, self.global_cnt["val"], string="val/"
                )

                self.logger.tb.add_scalar("val/loss/optical_flow", flow_loss.item(), self.global_cnt["val"])
                self.logger.tb.add_scalar("val/loss/contact", contact_loss.item(), self.global_cnt["val"])
                self.logger.tb.add_scalar("val/loss/is_paired", is_paired_loss.item(), self.global_cnt["val"])
                self.logger.tb.add_scalar("val/loss/kl", kl.item(), self.global_cnt["val"])
                self.logger.tb.add_scalar("val/loss/total_loss", loss_val.item(), self.global_cnt["val"])
                self.logger.tb.add_scalar("val/loss/ee_delta", ee_delta_loss, self.global_cnt["val"])
                self.global_cnt["val"] += 1

        # ---------------------------
        # Record Epoch Level Variables
        # ---------------------------
        self.logger.tb.add_scalar(
            "val/accuracy/contact", self.val_contact_accuracy, self.global_cnt["val"]
        )
        self.logger.tb.add_scalar(
            "val/accuracy/is_paired", self.val_paired_accuracy, self.global_cnt["val"]
        )

    def load_model(self, path):
        self.logger.print("Loading model from {}...".format(path))
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt)
        self.model.eval()

    def loss_calc(self, sampled_batched): ################################################

        # input data
        image = self.alpha_vision * sampled_batched["image"].to(self.device)
        force = self.alpha_force * sampled_batched["force"].to(self.device)
        proprio = self.alpha_proprio * sampled_batched["proprio"].to(self.device)
        depth = self.alpha_depth * sampled_batched["depth"].to(self.device).transpose(
            1, 3
        ).transpose(2, 3)

        action = sampled_batched["action"].to(self.device)

        # print("image: " + str(image) + str(image.size()))
        # print("force: " + str(force) + str(force.size()))
        # print("proprio: " + str(proprio) + str(proprio.size()))
        # print("depth: " + str(depth) + str(depth.size()))
        # print("action: " + str(action) + str(action.size()))
        # print("#########################################################################\n#######################################################")

        contact_label = sampled_batched["contact_next"].to(self.device)
        optical_flow_label = sampled_batched["flow"].to(self.device)
        optical_flow_mask_label = sampled_batched["flow_mask"].to(self.device)

        # print("contact_label: " + str(contact_label) + str(contact_label.size()))
        # print("optical_flow_label: " + str(optical_flow_label) + str(optical_flow_label.size()))
        # print("optical_flow_mask_label: " + str(optical_flow_mask_label) + str(optical_flow_mask_label.size()))

        # unpaired data for sampled point
        unpaired_image = self.alpha_vision * sampled_batched["unpaired_image"].to(
            self.device
        )
        unpaired_force = self.alpha_force * sampled_batched["unpaired_force"].to(
            self.device
        )
        unpaired_proprio = self.alpha_proprio * sampled_batched["unpaired_proprio"].to(
            self.device
        )
        unpaired_depth = self.alpha_depth * sampled_batched["unpaired_depth"].to(
            self.device
        ).transpose(1, 3).transpose(2, 3)

        # print("unpaired_image: " + str(unpaired_image.size()) + str(unpaired_image))
        # print("unpaired_force: " + str(unpaired_force.size()) + str(unpaired_force))
        # print("unpaired_proprio: " + str(unpaired_proprio.size()) + str(unpaired_proprio))
        # print("unpaired_depth: " + str(unpaired_depth.size()) + str(unpaired_depth))


        # labels to predict
        gt_ee_pos_delta = sampled_batched["ee_yaw_next"].to(self.device)

        if self.deterministic:
            paired_out, contact_out, flow2, optical_flow2_mask, ee_delta_out, mm_feat = self.model(
                image, force, proprio, depth, action
            )
            kl = torch.tensor([0]).to(self.device).type(torch.cuda.FloatTensor)
        else:
            paired_out, contact_out, flow2, optical_flow2_mask, ee_delta_out, mm_feat, mu_z, var_z, mu_prior, var_prior, z_depth = self.model(
                image, force, proprio, depth, action
            )
            # print("SHAPE OF mu_z: " + str(mu_z.shape))
            # print("mu_z: " + str(mu_z))
            # print("SHAPE OF var_z: " + str(var_z.shape))
            # print("var_z: " + str(var_z))

            # print("SHAPE OF mu_prior: " + str(mu_prior.size()))
            # print("z_depth: " + str(z_depth))

            mu_prior = enlarge_tensor_by_factor(mu_prior, int(z_depth/2)).to(self.device)
            var_prior = enlarge_tensor_by_factor(var_prior, int(z_depth/2)).to(self.device)

            # print("SHAPE OF mu_prior: " + str(mu_prior.size()))
            # print("mu_prior: " + str(mu_prior))
            # print("SHAPE OF var_prior: " + str(var_prior.size()))
            # print("var_prior: " + str(var_prior))
            kl = self.alpha_kl * torch.mean(
                kl_normal(mu_z, var_z, mu_prior.squeeze(0), var_prior.squeeze(0))
            )

        flow_loss = self.alpha_optical_flow * realEPE(
            flow2, optical_flow_label, self.device
        )

        # Scene flow losses

        b, _, h, w = optical_flow_label.size()

        # optical_flow_mask = nn.functional.upsample(
        optical_flow_mask = nn.functional.interpolate(
            optical_flow2_mask, size=(h, w), mode="bilinear"
        )

        flow_mask_loss = self.alpha_optical_flow_mask * self.loss_optical_flow_mask(
            optical_flow_mask, optical_flow_mask_label
        )

        contact_loss = self.alpha_contact * self.loss_contact_next(
            contact_out, contact_label
        )

        ee_delta_loss = self.alpha_ee_fut * self.loss_ee_pos(
            ee_delta_out, gt_ee_pos_delta
        )

        paired_loss = self.alpha_pair * self.loss_is_paired(
            paired_out, torch.ones(paired_out.size(0), 1).to(self.device)
        )


        # print("unpaired_image: " + str(unpaired_image))
        # print("unpaired_force: " + str(unpaired_force))
        # print("unpaired_proprio: " + str(unpaired_proprio))
        # print("unpaired_depth: " + str(unpaired_depth))
        # print("action: " + str(action))
        unpaired_total_losses = self.model(
            unpaired_image, unpaired_force, unpaired_proprio, unpaired_depth, action
        )
        unpaired_out = unpaired_total_losses[0]

        # print("FIRST") ##################################
        # print("self.alpha_pair: " + str(self.alpha_pair))
        # print("self.loss_is_paired :" + str(self.loss_is_paired(unpaired_out, torch.zeros(unpaired_out.size(0), 1).to(self.device))))
        # print("unpaired_out :" + str(unpaired_out)) # nan
        # print("torch.zeros(unpaired_out.size(0), 1).to(self.device) :" + str(torch.zeros(unpaired_out.size(0), 1).to(self.device)))
        
        unpaired_loss = self.alpha_pair * self.loss_is_paired(unpaired_out, torch.zeros(unpaired_out.size(0), 1).to(self.device))

        xyz = self.loss_is_paired(unpaired_out, torch.zeros(unpaired_out.size(0), 1).to(self.device))

        # print("unpaired_out: " + str(unpaired_out.size()) + str(unpaired_out))
        # print("self.loss_is_paired(unpaired_out, torch.zeros(unpaired_out.size(0), 1).to(self.device)): " + str(xyz.size()) + str(xyz))
        # print("self.alpha_pair: " + str(self.alpha_pair))
        # print("unpaired_loss: " + str(unpaired_loss.size()) + str(unpaired_loss))

        loss = (
            contact_loss
            + paired_loss
            + unpaired_loss
            + ee_delta_loss
            + kl
            + flow_loss
            + flow_mask_loss
        )

        contact_pred = nn.Sigmoid()(contact_out).detach()
        contact_accuracy = compute_accuracy(contact_pred, contact_label.detach())

        paired_pred = nn.Sigmoid()(paired_out).detach()
        paired_accuracy = compute_accuracy(
            paired_pred, torch.ones(paired_pred.size()[0], 1, device=self.device)
        )

        unpaired_pred = nn.Sigmoid()(unpaired_out).detach()
        unpaired_accuracy = compute_accuracy(
            unpaired_pred, torch.zeros(unpaired_pred.size()[0], 1, device=self.device)
        )

        is_paired_accuracy = (paired_accuracy + unpaired_accuracy) / 2.0

        # logging
        is_paired_loss = paired_loss + unpaired_loss

        return (
            loss,
            mm_feat,
            (
                flow_loss,
                contact_loss,
                is_paired_loss,
                contact_accuracy,
                is_paired_accuracy,
                ee_delta_loss,
                kl,
            ),
            (flow2, optical_flow_label, image),
        )

    def record_results(self, total_loss, results, global_cnt, t_st):

        flow_loss, contact_loss, is_paired_loss, contact_accuracy, is_paired_accuracy, ee_delta_loss, kl = (
            results
        )

        # print("flow_loss.item(): " + str(flow_loss.item()))
        # print("global_cnt: " + str(global_cnt))

        self.logger.tb.add_scalar("loss/optical_flow", flow_loss.item(), global_cnt)
        self.logger.tb.add_scalar("loss/contact", contact_loss.item(), global_cnt)
        self.logger.tb.add_scalar("loss/is_paired", is_paired_loss.item(), global_cnt)
        self.logger.tb.add_scalar("loss/kl", kl.item(), global_cnt)
        self.logger.tb.add_scalar("loss/total_loss", total_loss.item(), global_cnt)
        self.logger.tb.add_scalar("loss/ee_delta", ee_delta_loss, global_cnt)

        self.logger.tb.add_scalar(
            "accuracy/contact", contact_accuracy.item(), global_cnt
        )
        self.logger.tb.add_scalar(
            "accuracy/is_paired", is_paired_accuracy.item(), global_cnt
        )

        self.logger.tb.add_scalar("stats/iter_time", time.time() - t_st, global_cnt)

    def _init_dataloaders(self):

        filename_list = []
        for file in os.listdir(self.configs["dataset"]):
            if file.endswith(".h5"):
                filename_list.append(self.configs["dataset"] + file)

        self.logger.print(
            "Number of files in multifile dataset = {}".format(len(filename_list))
        )

        val_filename_list = []

        val_index = np.random.randint(
            0, len(filename_list), int(len(filename_list) * self.configs["val_ratio"])
        )

        for index in val_index:
            val_filename_list.append(filename_list[index])

        while val_index.size > 0:
            filename_list.pop(val_index[0])
            val_index = np.where(val_index > val_index[0], val_index - 1, val_index)
            val_index = val_index[1:]

        self.logger.print("Initial finished")

        val_filename_list1, filename_list1 = augment_val(
            val_filename_list, filename_list
        )

        self.logger.print("Listing finished")

        self.dataloaders = {}
        self.samplers = {}
        self.datasets = {}

        self.samplers["val"] = SubsetRandomSampler(
            range(len(val_filename_list1) * (self.configs["ep_length"] - 1))
        )
        self.samplers["train"] = SubsetRandomSampler(
            range(len(filename_list1) * (self.configs["ep_length"] - 1))
        )

        self.logger.print("Sampler finished")

        self.datasets["train"] = MultimodalManipulationDataset(
            filename_list1,
            transform=transforms.Compose(
                [
                    ProcessForce(32, "force", tanh=True),
                    ProcessForce(32, "unpaired_force", tanh=True),
                    ToTensor(device=self.device),
                ]
            ),
            episode_length=self.configs["ep_length"],
            training_type=self.configs["training_type"],
            action_dim=self.configs["action_dim"],

        )

        self.datasets["val"] = MultimodalManipulationDataset(
            val_filename_list1,
            transform=transforms.Compose(
                [
                    ProcessForce(32, "force", tanh=True),
                    ProcessForce(32, "unpaired_force", tanh=True),
                    ToTensor(device=self.device),
                ]
            ),
            episode_length=self.configs["ep_length"],
            training_type=self.configs["training_type"],
            action_dim=self.configs["action_dim"],

        )

        self.logger.print("Dataset finished")

        self.dataloaders["val"] = DataLoader(
            self.datasets["val"],
            batch_size=self.configs["batch_size"],
            num_workers=self.configs["num_workers"],
            sampler=self.samplers["val"],
            pin_memory=True,
            drop_last=True,
        )
        self.dataloaders["train"] = DataLoader(
            self.datasets["train"],
            batch_size=self.configs["batch_size"],
            num_workers=self.configs["num_workers"],
            sampler=self.samplers["train"],
            pin_memory=True,
            drop_last=True,
        )

        self.len_data = len(self.dataloaders["train"])
        self.val_len_data = len(self.dataloaders["val"])

        self.logger.print("Finished setting up date")

    def _record_image(self, image_packet, global_cnt, string=None):

        if string is None:
            string = ""

        flow2, flow_label, image = image_packet
        image_index = 0

        b, c, h, w = flow_label.size()

        # upsampled_flow = nn.functional.upsample(flow2, size=(h, w), mode="bilinear")
        upsampled_flow = nn.functional.interpolate(flow2, size=(h, w), mode="bilinear")
        upsampled_flow = upsampled_flow.cpu().detach().numpy()
        orig_image = image[image_index].cpu().numpy()

        orig_flow = flow2rgb(
            flow_label[image_index].cpu().detach().numpy(), max_value=None
        )
        pred_flow = flow2rgb(upsampled_flow[image_index], max_value=None)

        concat_image = np.concatenate([orig_image, orig_flow, pred_flow], 1)

        concat_image = concat_image * 255
        concat_image = concat_image.astype(np.uint8)
        concat_image = concat_image.transpose(2, 0, 1)

        self.logger.tb.add_image(string + "predicted_flow", concat_image, global_cnt)

def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, : target.size(2), : target.size(3)]


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    """`same` convolution with LeakyReLU, i.e. output shape equals input shape.
  Args:
    in_planes (int): The number of input feature maps.
    out_planes (int): The number of output feature maps.
    kernel_size (int): The filter size.
    dilation (int): The filter dilation factor.
    stride (int): The filter stride.
  """
    # compute new filter size after dilation
    # and necessary padding for `same` output size
    dilated_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    same_padding = (dilated_kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=same_padding,
            dilation=dilation,
            bias=bias,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Flatten(nn.Module):
    """Flattens convolutional feature maps for fc layers.
  """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(x.size(0), -1)


class CausalConv1D(nn.Conv1d):
    """A causal 1D convolution.
  """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True
    ):
        self.__padding = (kernel_size - 1) * dilation

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        res = super().forward(x)
        if self.__padding != 0:
            return res[:, :, : -self.__padding]
        return res


class ResidualBlock(nn.Module):
    """A simple residual block.
  """

    def __init__(self, channels):
        super().__init__()

        self.conv1 = conv2d(channels, channels, bias=False)
        self.conv2 = conv2d(channels, channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)  # nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(x)
        out = self.act(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        return out + x

class ProprioEncoder(nn.Module):
    def __init__(self, z_dim, z_depth, initailize_weights=True):
        """
        Image encoder taken from selfsupervised code
        """
        super().__init__()
        self.z_dim = z_dim
        self.z_depth = z_depth
        out = z_depth * z_dim

        self.proprio_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, z_depth * z_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, proprio, z_depth):
        return self.proprio_encoder(proprio).unsqueeze(2)


class ForceEncoder(nn.Module):
    def __init__(self, z_dim, z_depth, initailize_weights=True):
        """
        Force encoder taken from selfsupervised code
        """
        super().__init__()
        self.z_dim = z_dim

        self.z_depth = z_depth

        self.frc_encoder = nn.Sequential(
            CausalConv1D(6, 16, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(16, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(32, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(64, 128, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(128, self.z_depth * self.z_dim, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, force):
        return self.frc_encoder(force)


class ImageEncoder(nn.Module):
    def __init__(self, z_dim, z_depth, initailize_weights=True):
        """
        Image encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.z_dim = z_dim

        self.z_depth = z_depth

        self.img_conv1 = conv2d(3, 16, kernel_size=7, stride=2)
        self.img_conv2 = conv2d(16, 32, kernel_size=5, stride=2)
        self.img_conv3 = conv2d(32, 64, kernel_size=5, stride=2)
        self.img_conv4 = conv2d(64, 64, stride=2)
        self.img_conv5 = conv2d(64, 128, stride=2)
        self.img_conv6 = conv2d(128, self.z_dim, stride=2)
        self.img_encoder = nn.Linear(4 * self.z_dim, self.z_depth * self.z_dim)
        self.flatten = Flatten()

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, image, z_depth):

        self.z_depth = z_depth

        # image encoding layers
        out_img_conv1 = self.img_conv1(image)
        out_img_conv2 = self.img_conv2(out_img_conv1)
        out_img_conv3 = self.img_conv3(out_img_conv2)
        out_img_conv4 = self.img_conv4(out_img_conv3)
        out_img_conv5 = self.img_conv5(out_img_conv4)
        out_img_conv6 = self.img_conv6(out_img_conv5)

        img_out_convs = (
            out_img_conv1,
            out_img_conv2,
            out_img_conv3,
            out_img_conv4,
            out_img_conv5,
            out_img_conv6,
        )

        # image embedding parameters
        flattened = self.flatten(out_img_conv6)
        ###print("SHAPE OF FLATTENED TENSOR: " + str(flattened.shape)) #########
        img_out = self.img_encoder(flattened).unsqueeze(2)

        return img_out, img_out_convs


class DepthEncoder(nn.Module):
    def __init__(self, z_dim, z_depth, initailize_weights=True):
        """
        Simplified Depth Encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.z_dim = z_dim

        self.z_depth = z_depth

        self.depth_conv1 = conv2d(1, 32, kernel_size=3, stride=2)
        self.depth_conv2 = conv2d(32, 64, kernel_size=3, stride=2)
        self.depth_conv3 = conv2d(64, 64, kernel_size=4, stride=2)
        self.depth_conv4 = conv2d(64, 64, stride=2)
        self.depth_conv5 = conv2d(64, 128, stride=2)
        self.depth_conv6 = conv2d(128, z_dim, stride=2)
        self.depth_conv7 = conv2d(z_dim, z_depth * z_dim, stride=2)

        self.depth_encoder = nn.Linear(z_depth * z_dim, z_depth * z_dim)
        self.flatten = Flatten()

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, depth, z_depth):
        # depth encoding layers
        out_depth_conv1 = self.depth_conv1(depth)
        out_depth_conv2 = self.depth_conv2(out_depth_conv1)
        out_depth_conv3 = self.depth_conv3(out_depth_conv2)
        out_depth_conv4 = self.depth_conv4(out_depth_conv3)
        out_depth_conv5 = self.depth_conv5(out_depth_conv4)
        out_depth_conv6 = self.depth_conv6(out_depth_conv5)
        out_depth_conv7 = self.depth_conv7(out_depth_conv6)

        depth_out_convs = (
            out_depth_conv1,
            out_depth_conv2,
            out_depth_conv3,
            out_depth_conv4,
            out_depth_conv5,
            out_depth_conv6,
            out_depth_conv7
        )

        # depth embedding parameters
        flattened = self.flatten(out_depth_conv7)
        ###print("SHAPE OF FLATTENED TENSOR 2: " + str(flattened.shape)) #########
        depth_out = self.depth_encoder(flattened).unsqueeze(2)

        return depth_out, depth_out_convs

class OpticalFlowDecoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
        """
        Decodes the optical flow and optical flow mask.
        """
        super().__init__()

        self.optical_flow_conv = conv2d(2 * z_dim, 64, kernel_size=1, stride=1)

        self.img_deconv6 = deconv(64, 64)
        self.img_deconv5 = deconv(64, 32)
        self.img_deconv4 = deconv(162, 32)
        self.img_deconv3 = deconv(98, 32)
        self.img_deconv2 = deconv(98, 32)

        self.predict_optical_flow6 = predict_flow(64)
        self.predict_optical_flow5 = predict_flow(162)
        self.predict_optical_flow4 = predict_flow(98)
        self.predict_optical_flow3 = predict_flow(98)
        self.predict_optical_flow2 = predict_flow(66)

        self.upsampled_optical_flow6_to_5 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False
        )
        self.upsampled_optical_flow5_to_4 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False
        )
        self.upsampled_optical_flow4_to_3 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False
        )
        self.upsampled_optical_flow3_to_2 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False
        )

        self.predict_optical_flow2_mask = nn.Conv2d(
            66, 1, kernel_size=3, stride=1, padding=1, bias=False
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, tiled_feat, img_out_convs):
        """
        Predicts the optical flow and optical flow mask.

        Args:
            tiled_feat: action conditioned z (output of fusion + action network)
            img_out_convs: outputs of the image encoders (skip connections)
        """
        out_img_conv1, out_img_conv2, out_img_conv3, out_img_conv4, out_img_conv5, out_img_conv6 = (
            img_out_convs
        )

        optical_flow_in_f = torch.cat([out_img_conv6, tiled_feat], 1)
        optical_flow_in_f2 = self.optical_flow_conv(optical_flow_in_f)
        optical_flow_in_feat = self.img_deconv6(optical_flow_in_f2)

        # predict optical flow pyramids
        optical_flow6 = self.predict_optical_flow6(optical_flow_in_feat)
        optical_flow6_up = crop_like(
            self.upsampled_optical_flow6_to_5(optical_flow6), out_img_conv5
        )
        out_img_deconv5 = crop_like(
            self.img_deconv5(optical_flow_in_feat), out_img_conv5
        )

        concat5 = torch.cat((out_img_conv5, out_img_deconv5, optical_flow6_up), 1)
        optical_flow5 = self.predict_optical_flow5(concat5)
        optical_flow5_up = crop_like(
            self.upsampled_optical_flow5_to_4(optical_flow5), out_img_conv4
        )
        out_img_deconv4 = crop_like(self.img_deconv4(concat5), out_img_conv4)

        concat4 = torch.cat((out_img_conv4, out_img_deconv4, optical_flow5_up), 1)
        optical_flow4 = self.predict_optical_flow4(concat4)
        optical_flow4_up = crop_like(
            self.upsampled_optical_flow4_to_3(optical_flow4), out_img_conv3
        )
        out_img_deconv3 = crop_like(self.img_deconv3(concat4), out_img_conv3)

        concat3 = torch.cat((out_img_conv3, out_img_deconv3, optical_flow4_up), 1)
        optical_flow3 = self.predict_optical_flow3(concat3)
        optical_flow3_up = crop_like(
            self.upsampled_optical_flow3_to_2(optical_flow3), out_img_conv2
        )
        out_img_deconv2 = crop_like(self.img_deconv2(concat3), out_img_conv2)

        concat2 = torch.cat((out_img_conv2, out_img_deconv2, optical_flow3_up), 1)

        optical_flow2_unmasked = self.predict_optical_flow2(concat2)

        optical_flow2_mask = self.predict_optical_flow2_mask(concat2)

        optical_flow2 = optical_flow2_unmasked * torch.sigmoid(optical_flow2_mask)

        return optical_flow2, optical_flow2_mask


class EeDeltaDecoder(nn.Module):
    def __init__(self, z_dim, action_dim, initailize_weights=True):
        """
        Decodes the EE Delta
        """
        super().__init__()

        self.ee_delta_decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, action_dim),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, mm_act_feat):
        return self.ee_delta_decoder(mm_act_feat)

def init_weights(modules):
    """
    Weight initialization from original SensorFusion Code
    """
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def sample_gaussian(m, v, device):

    epsilon = Normal(0, 1).sample(m.size())
    z = m + torch.sqrt(v) * epsilon.to(device)

    # print("m: " + str(m.size) + str(m))
    # print("v: " + str(v.size) + str(v))
    # print("epsilon: " + str(epsilon.size) + str(epsilon))

    return z


def gaussian_parameters(h, dim=-1):

    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v


def product_of_experts(m_vect, v_vect):

    T_vect = 1.0 / v_vect

    mu = (m_vect * T_vect).sum(2) * (1 / T_vect.sum(2))
    var = 1 / T_vect.sum(2)

    # print("T_vect: " + str(T_vect.size) + str(T_vect))
    # print("mu: " + str(mu.size) + str(mu))
    # print("var: " + str(var.size) + str(var))

    return mu, var


def duplicate(x, rep):
    return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])


def depth_deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(
            in_planes, 16, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=True
        ),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=True),
        nn.LeakyReLU(0.1, inplace=True),
        nn.ConvTranspose2d(
            16, out_planes, kernel_size=4, stride=2, padding=1, bias=True
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def rescaleImage(image, output_size=128, scale=1 / 255.0):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    image_transform = image * scale
    return image_transform.transpose(1, 3).transpose(2, 3)


def filter_depth(depth_image):
    depth_image = torch.where(
        depth_image > 1e-7, depth_image, torch.zeros_like(depth_image)
    )
    return torch.where(depth_image < 2, depth_image, torch.zeros_like(depth_image))

def remove_zeros(tensor):
    """
    This function takes a torch tensor as input and adds 1e-9 to each element of the tensor until it no longer contains
    any zero values. The modified tensor is then returned.

    Parameters:
    tensor (torch.Tensor): The input tensor.

    Returns:
    tensor (torch.Tensor): The modified tensor with no zero values.
    """
    # Check if the tensor contains any zero values
    while torch.any(tensor == 0):
        # If it does, add 1e-9 to each element of the tensor
        tensor += 1e-9

    # Return the modified tensor
    return tensor


if __name__ == "__main__":

    # Load the config file
    parser = argparse.ArgumentParser(description="Sensor fusion model")
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--notes", default="", help="run notes")
    parser.add_argument("--dev", type=bool, default=False, help="run in dev mode")
    parser.add_argument(
        "--continuation",
        type=bool,
        default=False,
        help="continue a previous run. Will continue the log file",
    )
    args = parser.parse_args()

    # Add the yaml to the config args parse
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Merge configs and args
    for arg in vars(args):
        configs[arg] = getattr(args, arg)

    # Initialize the loggers
    logger = Logger(configs)

    # Initialize the trainer
    trainer = selfsupervised(configs, logger)

    trainer.train()
