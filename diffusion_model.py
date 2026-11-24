"""MMMI backbone/neck/TPAVI/decoder with the paper's conditional latent diffusion."""
import torch
from torch import nn
from torch.nn import functional as F
from model.encoder.get_encoder import get_encoder
from model.neck.get_neck import get_neck
from model.decoder.get_decoder import get_decoder
from model.fusion_module.tpavi import TPAVIModule
from ddpm1d import GaussianDiffusion1D


class LatentEncoder(nn.Module):
    def __init__(self, input_channels, dim=24):
        super().__init__()
        channels = [input_channels, 16, 32, 64, 64, 64]
        layers = []
        for a, b in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(a, b, 4, 2, 1), nn.BatchNorm2d(b), nn.LeakyReLU(.01)]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 128), nn.LeakyReLU(.01), nn.Linear(128, dim))

    def forward(self, image):
        return self.fc(self.conv(image).flatten(1))


class AVConditionEncoder(nn.Module):
    def __init__(self, dim=24):
        super().__init__()
        self.visual = LatentEncoder(3, dim)
        self.audio = nn.Sequential(nn.Linear(128, 128), nn.LeakyReLU(.01), nn.Linear(128, dim))
        self.fuse = nn.Sequential(nn.Linear(2 * dim, 128), nn.LeakyReLU(.01), nn.Linear(128, dim))

    def forward(self, image, audio):
        return self.fuse(torch.cat([self.visual(image), self.audio(audio)], dim=1))


class DiffusionAVS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.MODEL.FUSION == 'tpavi' and cfg.PARAM.TPAVI_STAGES == [0, 1, 2, 3]
        self.encoder, channels = get_encoder(cfg)
        self.neck_prior = get_neck(cfg, channels)
        self.decoder_prior = get_decoder(cfg)
        self.fusion_modules = nn.ModuleList(TPAVIModule(cfg.MODEL.NECK_CHANNEL) for _ in range(4))
        self.enc_z = LatentEncoder(1, cfg.DIFFUSION.LATENT_DIM)
        self.enc_c = AVConditionEncoder(cfg.DIFFUSION.LATENT_DIM)
        self.diffusion = GaussianDiffusion1D(cfg.DIFFUSION.LATENT_DIM, cfg.DIFFUSION.TRAIN_STEPS,
                                            cfg.DIFFUSION.SAMPLE_STEPS, cfg.DIFFUSION.get('PARAMETERIZATION', 'epsilon'),
                                            cfg.DIFFUSION.get('OBJECTIVE', 'epsilon'), cfg.DIFFUSION.get('LOSS_TYPE', 'mse'))
        self.noise_conv = nn.Conv2d(cfg.MODEL.NECK_CHANNEL + cfg.DIFFUSION.LATENT_DIM,
                                    cfg.MODEL.NECK_CHANNEL, 3, padding=1)
        self.temperature = cfg.DIFFUSION.TEMPERATURE
        self.task = cfg.TRAIN.TASK

    def forward(self, images, audio, masks=None, generator=None, latent_mode='sample'):
        features = self.neck_prior(self.encoder(images))
        visual, acoustic = [], []
        for block, feature in zip(self.fusion_modules, features):
            v, a = block(feature, audio.unsqueeze(1))
            visual.append(v)
            acoustic.append(a)
        condition = self.enc_c(images, audio)
        aux = {}
        if masks is not None:
            if not self.training:
                raise ValueError('Evaluation must not receive GT masks')
            if self.task == 'S4':
                if len(images) != 5 * len(masks):
                    raise ValueError('S4 training requires one GT mask per five-frame clip')
                # Clip-major frame order: AAAAA BBBBB, not ABABABABAB.
                masks = masks.repeat_interleave(5, dim=0)
            target = self.enc_z(masks)
            diff_loss, latent = self.diffusion(target, condition)
            # Eq.11: for each condition, paired GT versus other GTs in the frame batch.
            logits = F.normalize(condition, dim=-1) @ F.normalize(target, dim=-1).T / self.temperature
            contrastive = F.cross_entropy(logits, torch.arange(len(images), device=images.device))
            aux = {'diffusion': diff_loss, 'contrastive': contrastive,
                   'target_std': target.detach().std(), 'sample_std': latent.detach().std(),
                   'condition_std': condition.detach().std()}
        else:
            latent = self.diffusion.sample(condition, generator=generator)
            if latent_mode == 'zero':
                latent = torch.zeros_like(latent)
            elif latent_mode != 'sample':
                raise ValueError(latent_mode)
        # Paper: tile stochastic z_hat_0 into deepest *fused* deterministic feature.
        decoder_features = list(visual)
        expanded = latent[:, :, None, None].expand(-1, -1, *visual[-1].shape[-2:])
        decoder_features[-1] = self.noise_conv(torch.cat([visual[-1], expanded], dim=1))
        prediction = self.decoder_prior(decoder_features)
        return prediction, aux, (visual, acoustic)
