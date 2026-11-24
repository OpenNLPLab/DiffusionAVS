"""S4/MS3 diffusion experiments on the accepted MMMI+TPAVI runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT
sys.path.insert(0, str(BASE / 'lib'))
import torch
from easydict import EasyDict
from diffusion_model import DiffusionAVS
from torchvggish.vggish import VGGish
from datasets.get_loader import get_loader
from loss import structure_loss_ms3, structure_loss_s4, A_MaskedV_SimmLoss
from utils.pyutils import set_seed
from utils.utility import mask_iou, Eval_Fmeasure


def configuration(args):
    cfg = EasyDict(json.loads((BASE / 'configs/default.json').read_text()))
    cfg.TRAIN.TASK = args.task
    # Task-specific augmentation.
    if args.task == 'S4':
        cfg.DATA.S4_AFFINE = False
    if args.s4_affine is not None:
        cfg.DATA.S4_AFFINE = args.s4_affine == 'on'
    if args.task == 'S4':
        data_root = Path(args.data_root).resolve() / 'Single-source'
        cfg.DATA.ANNO_CSV = str(data_root / 's4_meta_data.csv')
        cfg.DATA.DIR_IMG = str(data_root / 's4_data/visual_frames')
        cfg.DATA.DIR_AUDIO_LOG_MEL = str(data_root / 's4_data/audio_log_mel')
        cfg.DATA.DIR_MASK = str(data_root / 's4_data/gt_masks')
        cfg.BACKUP.BASE_NAME = 'S4_pvt_basic'
        cfg.BACKUP.FOLDER_NAME = 'S4_pvt_diffusion'
    cfg.PARAM.EPOCHS = args.epochs
    if args.seed is not None:
        cfg.PARAM.SEED = args.seed
    if args.lr_decay_step is not None:
        cfg.PARAM.LR_DECAY_STEP = args.lr_decay_step
    cfg.DIFFUSION = EasyDict(LATENT_DIM=24, TRAIN_STEPS=20, SAMPLE_STEPS=10,
                            PARAMETERIZATION=args.parameterization,
                            OBJECTIVE=args.objective, LOSS_TYPE=args.diffusion_loss,
                            WEIGHT=1., CONTRASTIVE_WEIGHT=args.contrastive, TEMPERATURE=1.,
                            SEGMENTATION=args.segmentation,
                            SEMANTIC_WEIGHT=2 * cfg.PARAM.SIM_LOSS if args.segmentation == 'baseline' else 0.,
                            EVAL_SEED=20261116, INIT='ImageNet PVT + pretrained frozen VGGish')
    if args.task == 'S4':
        cfg.DIFFUSION.S4_TARGET = 'repeat_first_gt_5_times'
    if args.semantic_weight is not None:
        cfg.DIFFUSION.SEMANTIC_WEIGHT = args.semantic_weight
    return relocate_assets(cfg, args)


def relocate_assets(cfg, args):
    data_root = Path(args.data_root).expanduser().resolve()
    subset, prefix = ('Single-source', 's4') if cfg.TRAIN.TASK == 'S4' else ('Multi-sources', 'ms3')
    folder = data_root / subset
    cfg.DATA.ANNO_CSV = str(folder / f'{prefix}_meta_data.csv')
    for key, subdir in [('DIR_IMG','visual_frames'), ('DIR_MASK','gt_masks'), ('DIR_AUDIO_LOG_MEL','audio_log_mel')]:
        cfg.DATA[key] = str(folder / f'{prefix}_data' / subdir)
    pretrained = Path(args.pretrained_root).expanduser().resolve()
    cfg.TRAIN.PRETRAINED_PVTV2_PATH = str(pretrained / 'pvt_v2_b5.pth')
    cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = str(pretrained / 'vggish-10086976.pth')
    return cfg


def batch_to_device(batch, audio_model):
    images, audio, masks, _ = batch
    images, audio, masks = images.cuda().flatten(0, 1), audio.cuda().flatten(0, 1), masks.cuda().flatten(0, 1)
    with torch.no_grad():
        audio = audio_model(audio)
    return images, audio, masks


def train_epoch(model, audio, loader, optimizer, cfg, smoke=False):
    model.train()
    total, count = {}, 0
    for index, batch in enumerate(loader):
        images, sound, masks = batch_to_device(batch, audio)
        optimizer.zero_grad(set_to_none=True)
        target_hook = None
        if smoke and cfg.TRAIN.TASK == 'S4':
            def check_repeated_target(module, inputs):
                expected = masks.repeat_interleave(5, dim=0)
                assert len(masks) * 5 == len(images)
                assert torch.equal(inputs[0], expected), 'Wrong S4 GT repetition/order'
                print(json.dumps({'s4_target_check': 'passed', 'clips': len(masks),
                                  'diffusion_targets': len(inputs[0])}), flush=True)
            target_hook = model.enc_z.register_forward_pre_hook(check_repeated_target)
        pred, aux, (visual, acoustic) = model(images, sound, masks)
        if target_hook is not None:
            target_hook.remove()
        seg_fn = structure_loss_s4 if cfg.TRAIN.TASK == 'S4' else structure_loss_ms3
        supervised_pred = pred[::5] if cfg.TRAIN.TASK == 'S4' else pred
        seg = (seg_fn(pred, masks) if cfg.DIFFUSION.SEGMENTATION == 'baseline'
               else torch.nn.functional.binary_cross_entropy_with_logits(supervised_pred, masks))
        semantic = (A_MaskedV_SimmLoss(pred, acoustic, visual, 'avg', kl_flag=True)
                    if cfg.DIFFUSION.SEMANTIC_WEIGHT else pred.new_zeros(()))
        loss = (seg + cfg.DIFFUSION.SEMANTIC_WEIGHT * semantic + aux['diffusion']
                + cfg.DIFFUSION.CONTRASTIVE_WEIGHT * aux['contrastive'])
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite train loss at batch {index}')
        if smoke:
            # A segmentation-only gradient must reach the denoiser AND both encoders.
            groups = {'denoiser': model.diffusion.model, 'gt_encoder': model.enc_z,
                      'condition_encoder': model.enc_c}
            for name, module in groups.items():
                gradients = torch.autograd.grad(seg, tuple(module.parameters()), retain_graph=True, allow_unused=True)
                norm = sum(float(g.abs().sum()) for g in gradients if g is not None)
                if not np.isfinite(norm) or norm <= 0:
                    raise AssertionError(f'Broken segmentation gradient: {name} {norm}')
                print(json.dumps({'segmentation_gradient': name, 'l1': norm}), flush=True)
        loss.backward()
        if smoke:
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        values = {**aux, 'segmentation': seg, 'semantic': semantic, 'loss': loss}
        for key, val in values.items():
            total[key] = total.get(key, 0.) + float(val.detach()) * len(images)
        count += len(images)
        if smoke:
            break
    return {key: val / count for key, val in total.items()}


@torch.no_grad()
def evaluate(model, audio, loader, seed, smoke=False, latent_mode='sample'):
    model.eval()
    ious, fscores = [], []
    # Use fixed initial noise per video for consistent evaluation.
    # Preserve global RNG so evaluation does not alter training random sequences.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        gen = torch.Generator(device='cuda').manual_seed(seed)
        for batch in loader:
            images, sound, masks = batch_to_device(batch, audio)
            pred, _, _ = model(images, sound, generator=gen, latent_mode=latent_mode)
            ious.append(mask_iou(pred[:, 0], masks[:, 0]).item())
            fscores.append(float(Eval_Fmeasure(pred[:, 0], masks[:, 0])))
            if smoke:
                break
    result = {'miou': float(np.mean(ious)), 'fscore': float(np.mean(fscores)), 'videos': len(ious)}
    if not all(np.isfinite(result[k]) for k in ['miou', 'fscore']):
        raise FloatingPointError(result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='data/AVSBench-object')
    p.add_argument('--pretrained-root', default='pretrained')
    p.add_argument('--task', choices=['MS3', 'S4'], default='MS3')
    p.add_argument('--epochs', type=int, help='Defaults to 15 for S4, 30 for MS3')
    p.add_argument('--seed', type=int, help='Independent initialization and training RNG seed')
    p.add_argument('--contrastive', type=float, default=0.1)
    p.add_argument('--parameterization', choices=['epsilon', 'vp', 'x0'], default='vp')
    p.add_argument('--objective', choices=['epsilon', 'x0', 'v'], default='epsilon')
    p.add_argument('--diffusion-loss', choices=['mse', 'l1'], default='mse')
    p.add_argument('--segmentation', choices=['baseline', 'bce'], default='baseline')
    p.add_argument('--s4-affine', choices=['on', 'off'])
    p.add_argument('--semantic-weight', type=float)
    p.add_argument('--output', default='runs/ddpm_only_seed1234')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--evaluate', help='Evaluate checkpoint without training')
    p.add_argument('--resume', help='Continue a trusted local last.pt in a NEW output directory')
    p.add_argument('--lr-decay-step', type=int, help='StepLR interval for a fresh training run')
    p.add_argument('--latent-mode', choices=['sample', 'zero'], default='sample')
    args = p.parse_args()
    if args.s4_affine is not None and args.task != 'S4':
        p.error('--s4-affine requires --task S4')
    if args.semantic_weight is not None and (not np.isfinite(args.semantic_weight) or args.semantic_weight < 0):
        p.error('--semantic-weight must be finite and nonnegative')
    if (args.s4_affine is not None or args.semantic_weight is not None) and (args.resume or args.evaluate):
        p.error('Augmentation/loss overrides require a fresh run')
    if args.epochs is None:
        args.epochs = 15 if args.task == 'S4' else 30
    if args.seed is not None and (args.resume or args.evaluate):
        p.error('--seed is for fresh runs; resume/evaluation use saved configuration')
    if args.lr_decay_step is not None:
        if args.lr_decay_step < 1:
            p.error('--lr-decay-step must be positive')
        if args.resume or args.evaluate:
            p.error('--lr-decay-step is for fresh runs; resume/evaluation use saved configuration')
    if args.evaluate and args.resume:
        p.error('--evaluate and --resume are mutually exclusive')
    saved_config = args.evaluate or args.resume
    cfg = (EasyDict(json.loads((Path(saved_config).parent / 'config.json').read_text()))
           if saved_config else configuration(args))
    cfg = relocate_assets(cfg, args)
    if args.resume:
        cfg.PARAM.EPOCHS = args.epochs
    set_seed(cfg.PARAM.SEED)
    model = DiffusionAVS(cfg).cuda()
    audio = VGGish(cfg, torch.device('cuda')).cuda().eval().requires_grad_(False)
    train_loader, eval_loader = get_loader(cfg, evaluate_only=bool(args.evaluate))
    if args.evaluate:
        model.load_state_dict(torch.load(args.evaluate, map_location='cpu', weights_only=True), strict=True)
        print(json.dumps({'evaluation': evaluate(model, audio, eval_loader, cfg.DIFFUSION.EVAL_SEED,
                                               latent_mode=args.latent_mode),
                          'checkpoint': args.evaluate, 'split': eval_loader.dataset.split,
                          'latent_mode': args.latent_mode}), flush=True)
        return
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    hashes = {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.relative_to(BASE)):
              hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [*ROOT.glob('*.py'), *BASE.joinpath('lib').rglob('*.py')]}
    (out / 'source_sha256.json').write_text(json.dumps(hashes, indent=2))
    with zipfile.ZipFile(out / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in ROOT.glob('*.py'):
            archive.write(path, 'experiment/' + path.name)
        for path in BASE.joinpath('lib').rglob('*.py'):
            archive.write(path, str(path.relative_to(BASE)))
    (out / 'environment.json').write_text(json.dumps({
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(),
        'python': sys.version}, indent=2))
    print(json.dumps({'parameters': sum(p.numel() for p in model.parameters()), 'config': cfg}), flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.PARAM.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.PARAM.LR_DECAY_STEP, cfg.PARAM.LR_DECAY_RATE)
    best = -1.
    start_epoch = 0
    if args.resume:
        # Only load a checkpoint produced by this project; it includes Python/NumPy RNG objects.
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state['metrics'].get('split') != 'val':
            raise ValueError('Resume requires a checkpoint selected with the validation split')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['metrics']['epoch']
        if cfg.PARAM.EPOCHS <= start_epoch and not args.smoke:
            raise ValueError('Requested total epochs must exceed the completed epoch')
        source = Path(args.resume).resolve().parent
        history = [json.loads(line) for line in (source / 'metrics.jsonl').read_text().splitlines()]
        history = [row for row in history if row['epoch'] <= start_epoch]
        assert len(history) == start_epoch and history[-1] == state['metrics']
        selected = json.loads((source / 'best_metrics.json').read_text())
        best = state['metrics']['best_miou']
        assert selected['miou'] == best and selected['epoch'] <= start_epoch
        if not args.smoke:
            shutil.copyfile(source / 'best.pth', out / 'best.pth')
            (out / 'best_metrics.json').write_text(json.dumps(selected, indent=2))
            (out / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history))
        torch.set_rng_state(state['rng']['torch'])
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
        np.random.set_state(state['rng']['numpy'])
        random.setstate(state['rng']['python'])
        provenance = {'resumed_from': str(Path(args.resume).resolve()), 'completed_epoch': start_epoch,
                      'total_epochs': cfg.PARAM.EPOCHS, 'restored_lr': optimizer.param_groups[0]['lr'],
                      'optimizer_scheduler_rng_restored': True, 'previous_best_miou': best}
        (out / 'resume_provenance.json').write_text(json.dumps(provenance, indent=2))
        print(json.dumps(provenance), flush=True)
        del state
    stop_epoch = start_epoch + 1 if args.smoke else cfg.PARAM.EPOCHS
    for epoch in range(start_epoch, stop_epoch):
        began = time.monotonic()
        train_lr = optimizer.param_groups[0]['lr']
        losses = train_epoch(model, audio, train_loader, optimizer, cfg, smoke=args.smoke)
        scheduler.step()
        if args.smoke:
            def reject_gt(*unused):
                raise AssertionError('GT encoder called during inference')
            hook = model.enc_z.register_forward_pre_hook(reject_gt)
        scores = evaluate(model, audio, eval_loader, cfg.DIFFUSION.EVAL_SEED, smoke=args.smoke)
        if args.smoke:
            repeat = evaluate(model, audio, eval_loader, cfg.DIFFUSION.EVAL_SEED, smoke=True)
            assert scores == repeat, (scores, repeat)
            hook.remove()
        improved = scores['miou'] > best
        best = max(best, scores['miou'])
        row = {'epoch': epoch + 1, 'split': eval_loader.dataset.split, **scores, 'train': losses,
               'best_miou': best, 'selected_best': improved, 'seconds': time.monotonic() - began,
               'lr': optimizer.param_groups[0]['lr'], 'train_lr': train_lr}
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        if not args.smoke:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'config': dict(cfg), 'metrics': row,
                     'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                             'numpy': np.random.get_state(), 'python': random.getstate()}}
            torch.save(state, out / 'last.pt.tmp')
            (out / 'last.pt.tmp').replace(out / 'last.pt')
            if improved:
                torch.save(model.state_dict(), out / 'best.pth.tmp')
                (out / 'best.pth.tmp').replace(out / 'best.pth')
                (out / 'best_metrics.json').write_text(json.dumps(row, indent=2))
        print(json.dumps(row), flush=True)
    (out / 'completed.json').write_text(json.dumps({'epochs': epoch + 1, 'best_miou': best, 'smoke': args.smoke}))
    print('COMPLETED', flush=True)


if __name__ == '__main__':
    main()
