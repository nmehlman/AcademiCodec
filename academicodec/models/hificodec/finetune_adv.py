import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import argparse
import json
import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from academicodec.models.hificodec.env import AttrDict, build_env
from academicodec.models.hificodec.meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from academicodec.models.encodec.msstftd import MultiScaleSTFTDiscriminator
from academicodec.models.hificodec.models import Generator
from academicodec.models.hificodec.models import MultiPeriodDiscriminator
from academicodec.models.hificodec.models import MultiScaleDiscriminator
from academicodec.models.hificodec.models import EmotionClassifier
from academicodec.models.hificodec.emotion_callback import emotion_callback
from academicodec.models.hificodec.models import feature_loss
from academicodec.models.hificodec.models import generator_loss
from academicodec.models.hificodec.models import discriminator_loss
from academicodec.models.hificodec.models import Encoder
from academicodec.models.hificodec.models import Quantizer
from academicodec.utils import plot_spectrogram
from academicodec.utils import save_checkpoint

torch.backends.cudnn.benchmark = True


def reconstruction_loss(x, G_x, device, eps=1e-7):
    L = 100 * F.mse_loss(x, G_x)  # wav L1 loss
    for i in range(6, 11):
        s = 2**i
        melspec = MelSpectrogram(
            sample_rate=24000,
            n_fft=s,
            hop_length=s // 4,
            n_mels=64,
            wkwargs={"device": device}).to(device)
        S_x = melspec(x)
        S_G_x = melspec(G_x)
        loss = ((S_x - S_G_x).abs().mean() + (
            ((torch.log(S_x.abs() + eps) - torch.log(S_G_x.abs() + eps))**2
             ).mean(dim=-2)**0.5).mean()) / (i)
        L += loss
    return L


def train(rank, a, h):
    torch.cuda.set_device(rank)
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    encoder = Encoder(h).to(device)
    generator = Generator(h).to(device)
    quantizer = Quantizer(h).to(device)
    
    # Freeze generator/quantizer as requested
    if h.freeze_quantizer:
        for param in quantizer.parameters():
            param.requires_grad = False
        print("Quantizer frozen, not training")
    if h.freeze_generator:
        for param in generator.parameters():
            param.requires_grad = False
        print("Generator frozen, not training")

    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    mstftd = MultiScaleSTFTDiscriminator(32).to(device)
    emotion_classifier = EmotionClassifier(latent_size=512).to(device)
    if rank == 0:
        print(encoder)
        print(quantizer)
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)
        print("Running in adversarial mode")

    steps = 0
    state_dict_do = None
    last_epoch = -1
    
    if a.pretrained_ckpt:
        print(f"Loading pretrained checkpoint from {a.pretrained_ckpt}")
        state_dict = torch.load(a.pretrained_ckpt, map_location=device)
        generator.load_state_dict(state_dict['generator'])
        encoder.load_state_dict(state_dict['encoder'])
        quantizer.load_state_dict(state_dict['quantizer'])

    if h.num_gpus > 1:
        generator = DistributedDataParallel(
            generator, device_ids=[rank]).to(device)
        encoder = DistributedDataParallel(encoder, device_ids=[rank]).to(device)
        quantizer = DistributedDataParallel(
            quantizer, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)
        mstftd = DistributedDataParallel(mstftd, device_ids=[rank]).to(device)
        emotion_classifier = DistributedDataParallel(
            emotion_classifier, device_ids=[rank]).to(device)

    optim_g = torch.optim.Adam(
        itertools.chain(generator.parameters(),
                        encoder.parameters(), quantizer.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.Adam(
        itertools.chain(msd.parameters(), mpd.parameters(),
                        mstftd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2])
    
    optim_e = torch.optim.Adam(
            emotion_classifier.parameters(),
            h.emo_learning_rate,
            betas=[h.adam_b1, h.adam_b2])
    
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(
        training_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        n_cache_reuse=0,
        shuffle=False if h.num_gpus > 1 else True,
        fmax_loss=h.fmax_for_loss,
        device=device,
        fine_tuning=a.fine_tuning,
        base_mels_path=a.input_mels_dir,
        emotion_labels=a.training_emotion_labels)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True)

    if rank == 0:
        validset = MelDataset(
            validation_filelist,
            h.segment_size,
            h.n_fft,
            h.num_mels,
            h.hop_size,
            h.win_size,
            h.sampling_rate,
            h.fmin,
            h.fmax,
            False,
            False,
            n_cache_reuse=0,
            fmax_loss=h.fmax_for_loss,
            device=device,
            fine_tuning=a.fine_tuning,
            base_mels_path=a.input_mels_dir,
            emotion_labels=a.validation_emotion_labels)
        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True)
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
    plot_gt_once = False
    generator.train()
    encoder.train()
    quantizer.train()
    mpd.train()
    msd.train()
    t = 1 # Counter for annealing adversarial loss
    warmup_started = False
    warmup_ended = False
    for epoch in range(max(0, last_epoch), a.training_epochs):
        warmup = epoch < h.emo_warmup_epochs
        if rank == 0:
            if warmup and not warmup_started:
                print(f"Warmup started (epoch {epoch + 1})")
                warmup_started = True
            if not warmup and not warmup_ended:
                print(f"Warmup ended. Starting adversarial training (epoch {epoch + 1})")
                warmup_ended = True
        if warmup:
            update_encoder = False
            update_quantizer = False
            update_generator = False
            for param in encoder.parameters():
                param.requires_grad = False
            for param in quantizer.parameters():
                param.requires_grad = False
            for param in generator.parameters():
                param.requires_grad = False
        else:
            update_encoder = True
            update_quantizer = not h.freeze_quantizer
            update_generator = not h.freeze_generator
            for param in encoder.parameters():
                param.requires_grad = True
            if update_quantizer:
                for param in quantizer.parameters():
                    param.requires_grad = True
            if update_generator:
                for param in generator.parameters():
                    param.requires_grad = True
        if rank == 0:
            if not warmup:
                print(f"[Epoch {epoch + 1}] Warmup ended. Starting adversarial training.")
                print("Epoch: {}".format(epoch + 1))
        
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        
        for i, batch in enumerate(train_loader):
            
            x, y, _, y_mel, emotion_labels = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)
            
            c = encoder(y)
            q, loss_q, c = quantizer(c)
            
            optim_e.zero_grad()
            emo_preds = emotion_classifier(q)
            emo_loss = F.mse_loss(emo_preds["valence"], emotion_labels["valence"].to(device).float()) + F.mse_loss(emo_preds["arousal"], emotion_labels["arousal"].to(device).float())
            
            if not warmup:
                y_g_hat = generator(q)
            
            if update_generator or update_quantizer or update_encoder: # Compute non-adversarial losses
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                    h.hop_size, h.win_size, h.fmin,
                    h.fmax_for_loss)
                y_r_mel_1 = mel_spectrogram(
                    y.squeeze(1), 512, h.num_mels, h.sampling_rate, 120, 512,
                    h.fmin, h.fmax_for_loss)
                y_g_mel_1 = mel_spectrogram(
                    y_g_hat.squeeze(1), 512, h.num_mels, h.sampling_rate, 120, 512,
                    h.fmin, h.fmax_for_loss)
                y_r_mel_2 = mel_spectrogram(
                    y.squeeze(1), 256, h.num_mels, h.sampling_rate, 60, 256, h.fmin,
                    h.fmax_for_loss)
                y_g_mel_2 = mel_spectrogram(
                    y_g_hat.squeeze(1), 256, h.num_mels, h.sampling_rate, 60, 256,
                    h.fmin, h.fmax_for_loss)
                y_r_mel_3 = mel_spectrogram(
                    y.squeeze(1), 128, h.num_mels, h.sampling_rate, 30, 128, h.fmin,
                    h.fmax_for_loss)
                y_g_mel_3 = mel_spectrogram(
                    y_g_hat.squeeze(1), 128, h.num_mels, h.sampling_rate, 30, 128,
                    h.fmin, h.fmax_for_loss)
        
                optim_d.zero_grad()
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                    y_df_hat_r, y_df_hat_g)
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                    y_ds_hat_r, y_ds_hat_g)
                y_disc_r, fmap_r = mstftd(y)
                y_disc_gen, fmap_gen = mstftd(y_g_hat.detach())
                loss_disc_stft, losses_disc_stft_r, losses_disc_stft_g = discriminator_loss(
                    y_disc_r, y_disc_gen)
                loss_disc_all = loss_disc_s + loss_disc_f + loss_disc_stft
                loss_disc_all.backward()
                optim_d.step()
                optim_g.zero_grad()
                loss_mel1 = F.l1_loss(y_r_mel_1, y_g_mel_1)
                loss_mel2 = F.l1_loss(y_r_mel_2, y_g_mel_2)
                loss_mel3 = F.l1_loss(y_r_mel_3, y_g_mel_3)
                loss_mel = F.l1_loss(y_mel,
                                    y_g_hat_mel) * 45 + loss_mel1 + loss_mel2
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                y_stftd_hat_r, fmap_stftd_r = mstftd(y)
                y_stftd_hat_g, fmap_stftd_g = mstftd(y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_fm_stft = feature_loss(fmap_stftd_r, fmap_stftd_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_stft, losses_gen_stft = generator_loss(y_stftd_hat_g)
            
            if not warmup:
                Lambda = h.emo_loss_weight * min(1, t/h.emo_anneal_steps) # Compute annealing for adversarial loss
                t += 1
            else:
                Lambda = 0
            
            if update_encoder or update_generator or update_quantizer:
                loss_gen_all = loss_gen_s + loss_gen_f + loss_gen_stft + loss_fm_s + loss_fm_f + loss_fm_stft + loss_mel + loss_q * 10 - Lambda * emo_loss
                loss_gen_all.backward()
                optim_g.step()
            else:
                loss_gen_all = 0.0
                loss_q = 0.0
            
            if not warmup:
                for _ in range(h.adv_update_steps):
                    optim_e.zero_grad()
                    emo_preds = emotion_classifier(q.clone().detach())
                    emo_loss = F.mse_loss(emo_preds["valence"], emotion_labels["valence"].to(device).float()) + F.mse_loss(emo_preds["arousal"], emotion_labels["arousal"].to(device).float())
                    emo_loss.backward()
                    optim_e.step()
            else: # Only do single update per batch during warm-up
                optim_e.zero_grad()
                emo_preds = emotion_classifier(q.clone().detach())
                emo_loss = F.mse_loss(emo_preds["valence"], emotion_labels["valence"].to(device).float()) + F.mse_loss(emo_preds["arousal"], emotion_labels["arousal"].to(device).float())
                emo_loss.backward()
                optim_e.step()
            
            if rank == 0:
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        if update_generator or update_quantizer or update_encoder:
                            mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                        else:
                            mel_error = 0.0
                    print(
                        'Steps : {:d}, Gen Loss Total : {:4.3f}, Loss Q : {:4.3f}, Mel-Spec. Error : {:4.3f}, Emo Loss : {:4.3f}, Weighted Emo Loss : {:4.3f}'.
                        format(steps, loss_gen_all, loss_q, mel_error,
                                emo_loss.item(), Lambda * emo_loss.item()))
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path,
                                                           steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'generator': (generator.module if h.num_gpus > 1
                                          else generator).state_dict(),
                            'encoder': (encoder.module if h.num_gpus > 1 else
                                        encoder).state_dict(),
                            'quantizer': (quantizer.module if h.num_gpus > 1
                                          else quantizer).state_dict()
                        },
                        num_ckpt_keep=a.num_ckpt_keep)
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path,
                                                            steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'mpd': (mpd.module
                                    if h.num_gpus > 1 else mpd).state_dict(),
                            'msd': (msd.module
                                    if h.num_gpus > 1 else msd).state_dict(),
                            'mstftd': (mstftd.module
                                       if h.num_gpus > 1 else msd).state_dict(),
                            'optim_g':
                            optim_g.state_dict(),
                            'optim_d':
                            optim_d.state_dict(),
                            'steps':
                            steps,
                            'epoch':
                            epoch
                        },
                        num_ckpt_keep=a.num_ckpt_keep)
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all,
                                  steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    sw.add_scalar("training/emo_loss", emo_loss, steps)
                    sw.add_scalar("training/lambda", float(Lambda), steps)
                if steps % a.validation_interval == 0 and steps != 0:
                    generator.eval()
                    encoder.eval()
                    quantizer.eval()
                    torch.cuda.empty_cache()
                    
                    # Callback to plot emotion results
                    emotion_callback(encoder, quantizer, generator, validation_loader, sw, n_batches=h.n_emotion_batches, steps=steps, device=device)
                    
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch[:4]
                            c = encoder(y.to(device).unsqueeze(1))
                            q, loss_q, c = quantizer(c)
                            y_g_hat = generator(q)
                            y_mel = torch.autograd.Variable(y_mel.to(device))
                            y_g_hat_mel = mel_spectrogram(
                                y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                h.sampling_rate, h.hop_size, h.win_size, h.fmin,
                                h.fmax_for_loss)
                            i_size = min(y_mel.size(2), y_g_hat_mel.size(2))
                            val_err_tot += F.l1_loss(
                                y_mel[:, :, :i_size],
                                y_g_hat_mel[:, :, :i_size]).item()
                            if j <= 8:
                                if not plot_gt_once:
                                    sw.add_audio('gt/y_{}'.format(j), y[0],
                                                 steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j),
                                                  plot_spectrogram(x[0]), steps)
                                sw.add_audio('generated/y_hat_{}'.format(j),
                                             y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(
                                    y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                    h.sampling_rate, h.hop_size, h.win_size,
                                    h.fmin, h.fmax)
                                sw.add_figure(
                                    'generated/y_hat_spec_{}'.format(j),
                                    plot_spectrogram(
                                        y_hat_spec.squeeze(0).cpu().numpy()),
                                    steps)
                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err,
                                      steps)
                        if not plot_gt_once:
                            plot_gt_once = True
                    generator.train()
            steps += 1
        scheduler_g.step()
        scheduler_d.step()

def main():
    print('Initializing Training Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default=None)
    parser.add_argument('--input_training_file', required=True)
    parser.add_argument('--input_validation_file', required=True)
    parser.add_argument('--training_emotion_labels', required=True)
    parser.add_argument('--validation_emotion_labels', required=True)
    parser.add_argument('--checkpoint_path', default='checkpoints')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=2000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument('--num_ckpt_keep', default=5, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--pretrained_ckpt', default='')

    a = parser.parse_args()
    assert a.training_emotion_labels is not None
    assert a.validation_emotion_labels is not None
    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass
    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h, ))
    else:
        train(0, a, h)

if __name__ == '__main__':
    main()
