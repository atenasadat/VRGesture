
#import configargparse
import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        # raise configargparse.ArgumentTypeError('Boolean value expected.')
        raise 'Boolean value expected.'

def parse_args():
    # parser = configargparse.ArgParser()
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default="/Users/atenasaghi/PycharmProjects/VRGesture/conifg.py", help='Config file path')
    parser.add_argument("--name", type=str, default="main")
    parser.add_argument("--train_data_path", action="append")
    parser.add_argument("--val_data_path", action="append")
    parser.add_argument("--test_data_path", action="append")
    parser.add_argument("--model_save_path", required=True)
    parser.add_argument("--pose_representation", type=str, default='3d_vec')
    parser.add_argument("--mean_dir_vec", action="append", type=float, nargs='*')
    parser.add_argument("--mean_pose", action="append", type=float, nargs='*')
    parser.add_argument("--random_seed", type=int, default=-1)
    parser.add_argument("--save_result_video", type=str2bool, default=True)

    # word embedding
    parser.add_argument("--wordembed_path", type=str, default=None)
    parser.add_argument("--wordembed_dim", type=int, default=100)
    parser.add_argument("--freeze_wordembed", type=str2bool, default=False)

    # model
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--dropout_prob", type=float, default=0.3)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--z_type", type=str, default='none')
    parser.add_argument("--input_context", type=str, default='both')

    # dataset
    parser.add_argument("--motion_resampling_framerate", type=int, default=24)
    parser.add_argument("--n_poses", type=int, default=50)
    parser.add_argument("--n_pre_poses", type=int, default=5)
    parser.add_argument("--subdivision_stride", type=int, default=5)
    parser.add_argument("--loader_workers", type=int, default=0)

    # GAN parameter
    parser.add_argument("--GAN_noise_size", type=int, default=0)

    # training
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--discriminator_lr_weight", type=float, default=0.2)
    parser.add_argument("--loss_regression_weight", type=float, default=50)
    parser.add_argument("--loss_gan_weight", type=float, default=1.0)
    parser.add_argument("--loss_kld_weight", type=float, default=0.1)
    parser.add_argument("--loss_reg_weight", type=float, default=0.01)
    parser.add_argument("--loss_warmup", type=int, default=-1)

    # eval
    parser.add_argument("--eval_net_path", type=str, default='')
    args, unknown = parser.parse_known_args()
    # args = parser.parse_args()
    return args