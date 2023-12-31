import torch
import torch.nn as nn
from models.models_utils import (
    duplicate,
    gaussian_parameters,
    rescaleImage,
    product_of_experts,
    sample_gaussian,
    filter_depth,
    remove_zeros
)
from models.base_models.encoders import (
    ProprioEncoder,
    ForceEncoder,
    ImageEncoder,
    DepthEncoder,
)
from models.base_models.decoders import (
    OpticalFlowDecoder,
    EeDeltaDecoder
)


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