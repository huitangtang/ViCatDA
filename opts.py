import argparse


def opts():
    parser = argparse.ArgumentParser(description='Training ViCatDA', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path_source', type=str, default='/data/',
                        help='root of the source dataset')
    parser.add_argument('--data_path_target_tr', type=str, default='/data/',
                        help='root of the target dataset (for training)')
    parser.add_argument('--data_path_target_te', type=str, default='/data/',
                        help='root of the target dataset (for test)')
    parser.add_argument('--src', type=str, default='', help='source domain')
    parser.add_argument('--tar_tr', type=str, default='', help='target domain (for training)')
    parser.add_argument('--tar_te', type=str, default='', help='target domain (for test)')
    # general optimization options
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd'], help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='dann', choices=['dann', 'cosine', 'step'], 
                        help='lr scheduler of dann, cosine, or step')
    parser.add_argument('--decay_epoch', type=int, nargs='+', default=[80, 120], 
                        help='decrease learning rate at these epochs for step decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='lr is multiplied by gamma on decay step')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (L2 penalty)')
    parser.add_argument('--nesterov', action='store_true', help='whether to use nesterov SGD')
    parser.add_argument('--eps', type=float, default=1e-6, help='a small value to prevent underflow')
    # specific optimization options
    parser.add_argument('--vda', action='store_true', help='whether to use vicinal domain adaptation')
    parser.add_argument('--alpha', type=float, default=0.2, help='parameter of Beta distribution')
    parser.add_argument('--consistent', action='store_true', help='whether to use weight and output from the same classifier in target part of vicatda loss')
    parser.add_argument('--emp', action='store_true', help='whether to follow entropy minimization principle')
    parser.add_argument('--cls_blc', action='store_true', help='whether to use class balance loss')
    parser.add_argument('--div', type=str, default='kl', help='measure of divergence between one target instance and its perturbed counterpart')
    parser.add_argument('--gray_tar_agree', action='store_true', help='whether to enforce the consistency between RGB and gray images on the target domain')
    parser.add_argument('--aug_tar_agree', action='store_true', help='whether to enforce the consistency between RGB and augmented images on the target domain')
    parser.add_argument('--sigma', type=float, default=0.1, help='standard deviation of Gaussian')
    parser.add_argument('--two_consistency', action='store_true', help='whether to use two consistency losses')
    # checkpoints
    parser.add_argument('--start_epoch', type=int, metavar='N', default=0, help='start epoch (useful on restarts)')
    parser.add_argument('--resume', type=str, default='', help='checkpoint path to resume (default: '')')
    parser.add_argument('--test_only', action='store_true', help='flag of test only')
    # architecture
    parser.add_argument('--arch', type=str, default='resnet50', help='model name')
    parser.add_argument('--pretrained', action='store_true', default=True, help='whether using pretrained model')
    parser.add_argument('--num_classes', type=int, default=31, help='class number of new model to be trained or fine-tuned')
    # i/o
    parser.add_argument('--log', type=str, default='./checkpoints', help='log folder')
    parser.add_argument('--workers', type=int, metavar='N', default=4, 
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--no_da', action='store_true', help='whether using data augmentation')
    parser.add_argument('--stop_epoch', type=int, metavar='N', default=200, 
                        help='stop epoch (default: 200)')
    parser.add_argument('--print_freq', type=int, metavar='N', default=10, 
                        help='print frequency (default: 10)')
    
    args = parser.parse_args()

    args.log += '_' + args.src + '2' + args.tar_tr + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + '_vda' + str(args.vda) + '_' + args.arch

    return args
