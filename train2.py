import datetime
import logging
import os
import random
import time
from pathlib import Path
import sys
import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'
[sys.path.append(i) for i in ['.', '..']]

import matplotlib
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from conifg import parse_args
from model import speech2gesture

from train_speech2gesture import train_iter_speech2gesture
from average_meter import AverageMeter
from data_utils import convert_dir_vec_to_pose

matplotlib.use('Agg')  # we don't use interactive GUI

from torch import optim

import train_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from mainrun import data_train,data_dev,data_test
def init_model(args, pose_dim, _device):
    # init model
    n_frames = args.n_poses
    generator = discriminator = loss_fn = None

    if args.model == 'speech2gesture':
        generator = speech2gesture.Generator(n_frames, pose_dim, args.n_pre_poses).to(_device)
        discriminator = speech2gesture.Discriminator(pose_dim).to(_device)
        loss_fn = torch.nn.L1Loss()
    print("Mode initiated",args.model,loss_fn)
    return generator, discriminator, loss_fn


def train_epochs(args, train_data_loader, test_data_loader, pose_dim, speaker_model=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss'), AverageMeter('gen'), AverageMeter('dis'),
                   AverageMeter('KLD'), AverageMeter('DIV_REG')]
    best_val_loss = (1e+10, 0)  # value, epoch

    tb_path = args.name + '_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    tb_writer = SummaryWriter(log_dir=str(Path(args.model_save_path).parent / 'tensorboard_runs' / tb_path))

    # interval params
    print_interval = int(len(train_data_loader) / 5)
    save_sample_result_epoch_interval = 10
    save_model_epoch_interval = 20


    # init model
    generator, discriminator, loss_fn = init_model(args, pose_dim, device)

    # use multi GPUs
    # if torch.cuda.device_count() > 1:
    #     generator = torch.nn.DataParallel(generator)
    #     if discriminator is not None:
    #         discriminator = torch.nn.DataParallel(discriminator)

    # prepare an evaluator for FGD
    embed_space_evaluator = None


    # define optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    dis_optimizer = None
    if discriminator is not None:
        dis_optimizer = torch.optim.Adam(discriminator.parameters(),
                                         lr=args.learning_rate * args.discriminator_lr_weight,
                                         betas=(0.5, 0.999))

    # training
    global_iter = 0
    best_values = {}  # best values for all loss metrics
    for epoch in range(args.epochs):
        # evaluate the test set
        val_metrics = evaluate_testset(test_data_loader, generator, loss_fn, embed_space_evaluator, args)

        # write to tensorboard and save best values
        for key in val_metrics.keys():
            tb_writer.add_scalar(key + '/validation', val_metrics[key], global_iter)
            if key not in best_values.keys() or val_metrics[key] < best_values[key][0]:
                best_values[key] = (val_metrics[key], epoch)

        # best?
        if 'frechet' in val_metrics.keys():
            val_loss = val_metrics['frechet']
        else:
            val_loss = val_metrics['loss']
        is_best = val_loss < best_val_loss[0]
        if is_best:
            logging.info('  *** BEST VALIDATION LOSS: {:.3f}'.format(val_loss))
            best_val_loss = (val_loss, epoch)
        else:
            logging.info('  best validation loss so far: {:.3f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))

        # save model
        if is_best or (epoch % save_model_epoch_interval == 0 and epoch > 0):
            dis_state_dict = None
            try:  # multi gpu
                gen_state_dict = generator.module.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.module.state_dict()
            except AttributeError:  # single gpu
                gen_state_dict = generator.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.state_dict()

            if is_best:
                save_name = '{}/{}_checkpoint_best.bin'.format(args.model_save_path, args.name)
            else:
                save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'lang_model': lang_model, 'speaker_model': speaker_model,
                'pose_dim': pose_dim, 'gen_dict': gen_state_dict,
                'dis_dict': dis_state_dict,
            }, save_name)

        # save sample results
        if args.save_result_video and epoch % save_sample_result_epoch_interval == 0:
            evaluate_sample_and_save_video(
                epoch, args.name, test_data_loader, generator,
                args=args, lang_model=lang_model)

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            in_text, text_lengths, in_text_padded, _, target_vec, in_audio, in_spec, aux_info = data
            batch_size = target_vec.size(0)

            in_text = in_text.to(device)
            in_text_padded = in_text_padded.to(device)
            in_audio = in_audio.to(device)
            in_spec = in_spec.to(device)
            target_vec = target_vec.to(device)

            # speaker input
            vid_indices = []
            # if speaker_model and isinstance(speaker_model, vocab.Vocab):
            #     vids = aux_info['vid']
            #     vid_indices = [speaker_model.word2index[vid] for vid in vids]
            #     vid_indices = torch.LongTensor(vid_indices).to(device)

            # train
            loss = []

            if args.model == 'speech2gesture':
                loss = train_iter_speech2gesture(args, in_spec, target_vec, generator, discriminator,
                                                 gen_optimizer, dis_optimizer, loss_fn)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # write to tensorboard
            for key in loss.keys():
                tb_writer.add_scalar(key + '/train', loss[key], global_iter)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, train_utils.time_since(start),
                    batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()

    tb_writer.close()

    # print best losses
    logging.info('--------- best loss values ---------')
    for key in best_values.keys():
        logging.info('{}: {:.3f} at EPOCH {}'.format(key, best_values[key][0], best_values[key][1]))




    # print
    ret_dict = {'loss': losses.avg, 'joint_mae': joint_mae.avg}
    elapsed_time = time.time() - start
    if embed_space_evaluator and embed_space_evaluator.get_no_of_samples() > 0:
        frechet_dist, feat_dist = embed_space_evaluator.get_scores()
        logging.info(
            '[VAL] loss: {:.3f}, joint mae: {:.5f}, accel diff: {:.5f}, FGD: {:.3f}, feat_D: {:.3f} / {:.1f}s'.format(
                losses.avg, joint_mae.avg, accel.avg, frechet_dist, feat_dist, elapsed_time))
        ret_dict['frechet'] = frechet_dist
        ret_dict['feat_dist'] = feat_dist
    else:
        logging.info('[VAL] loss: {:.3f}, joint mae: {:.3f} / {:.1f}s'.format(
            losses.avg, joint_mae.avg, elapsed_time))

    return ret_dict





def main(config):
    args = config['args']

    # random seed
    if args.random_seed >= 0:
        train_utils.set_random_seed(args.random_seed)

    # set logger
    train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    # build vocab
    print("train_data_path",args.train_data_path)


    # train
    # pose_dim = 27  # 9 x 3
    # train_epochs(args, data_train, data_test, lang_model,
    #              pose_dim=pose_dim, speaker_model=data_train.speaker_model)




import argparse

if __name__ == '__main__':
    _args =parse_args()
    # parser = argparse.ArgumentParser()
    # _args, unknown = parser.parse_known_args()

    main({'args': _args})