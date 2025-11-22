import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
from spikingjelly.activation_based import functional
import numpy as np
import argparse
import sys
import os
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True
_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

import Models

def main():
    parser = argparse.ArgumentParser()
    # init_tau, batch_size, learning_rate, T_max, log_dir, use_plif
    parser.add_argument('-init_tau', type=float, help='tau of LIF neuron or tau_0 of PLIF neuron')
    parser.add_argument('-batch_size', type=int, default=16, help='batch size during training')
    parser.add_argument('-learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-T_max', type=int, default=64, help='T of learning rate optimizer')
    parser.add_argument('-use_plif', action='store_true', default=False, help='True-activate PLIF; False-deactivate PLIF')
    parser.add_argument('-alpha_learnable', action='store_true', default=False, help='True-parameter alpha in surrogate function is learnable; False-not learnable')
    parser.add_argument('-use_max_pool', action='store_true', default=False, help='True-use max pooling; False-no use of max pooling')
    parser.add_argument('-device', type=str, help='device type used during training')
    parser.add_argument('-dataset_name', type=str, help='dataset used')
    parser.add_argument('-dataset_name', type=str, help='dataset used')
    parser.add_argument('-dataset_dir', type=str, default='D:/Ph.D/DATASETS', help='directory of datasets')
    parser.add_argument('-log_dir_prefix', type=str, help='directory storing TensorBoard log')
    parser.add_argument('-T', type=int, help='network simulation time')
    parser.add_argument('-channels', type=int, help='number of convolutional layer output channels (for the neuromorphic network)')
    parser.add_argument('-number_layer', type=int, help='number of convolutional layer (for the neuromorphic network)')
    parser.add_argument('-split_by', type=str, help='The method for splitting neuromorphic data to integrate into frames')
    parser.add_argument('-normalization', type=str, help='The method for normalization of integrated frame data')
    parser.add_argument('-max_epoch', type=int, help='max training epochs')
    parser.add_argument('-detach_reset', action='store_true', default=False, help='whether to detach the voltage reset from the backpropagation graph')

    args = parser.parse_args()
    print("args:", args)
    argv = ' '.join(sys.argv)
    print("argv:", argv)

    init_tau = args.init_tau
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    T_max = args.T_max
    use_plif = args.use_plif
    alpha_learnable = args.alpha_learnable
    use_max_pool = args.use_max_pool
    device = args.device
    dataset_name = args.dataset_name
    dataset_dir = args.dataset_dir + dataset_name
    log_dir_prefix = args.log_dir_prefix
    T = args.T
    max_epoch = args.max_epoch
    detach_reset = args.detach_reset

    number_layer = args.number_layer
    channels = args.channels
    split_by = args.split_by
    normalization = args.normalization
    if normalization == 'None':
        normalization = None

    # Neuromophic datasets
    if dataset_name != 'MNIST' and dataset_name != 'FashionMNIST' and dataset_name != 'CIFAR10':
        dir_name = f'{dataset_name}_init_tau_{init_tau}_use_plif_{use_plif}_use_max_pool_{use_max_pool}_T_{T}_c_{channels}_n_{number_layer}_split_by_{split_by}_normalization_{normalization}_detach_reset_{detach_reset}'
    # Static datasets
    else:
        dir_name = f'{dataset_name}_init_tau_{init_tau}_use_plif_{use_plif}_use_max_pool_{use_max_pool}_T_{T}_detach_reset_{detach_reset}'

    log_dir = os.path.join(log_dir_prefix, dir_name)

    pt_dir = os.path.join(log_dir_prefix, 'pt_' + dir_name)
    print(log_dir, pt_dir)

    if not os.path.exists(pt_dir):
        os.mkdir(pt_dir)
    class_num = 10  # 绝大多数数据集为10，如果不一样，在新建dataloader的时候再修改

    if dataset_name == 'MNIST':
        transform_train, transform_test = Models.get_transforms(dataset_name)
        train_dataloader = DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=True,
                transform=transform_train,
                download=True
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True
        )

        test_dataloader = DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=False,
                transform=transform_test,
                download=True
            ),
            batch_size=batch_size * 8,      # In test, grad calculation is deactivated, so less memory is needed. Therefore, we can enlarge the batchsize to utilize the saved memory
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )
    elif dataset_name == 'FashionMNIST':
        transform_train, transform_test = Models.get_transforms(dataset_name)
        train_dataloader = DataLoader(
            dataset=torchvision.datasets.FashionMNIST(
                root=dataset_dir,
                train=True,
                transform=transform_train,
                download=True
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True
        )
        test_dataloader = DataLoader(
            dataset=torchvision.datasets.FashionMNIST(
                root=dataset_dir,
                train=False,
                transform=transform_test,
                download=True
            ),
            batch_size=batch_size * 16,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )
    elif dataset_name == 'CIFAR10':
        from torchvision.datasets.cifar import CIFAR10
        transform_train, transform_test = Models.get_transforms(dataset_name)
        train_dataloader = DataLoader(
            dataset=CIFAR10(
                root=dataset_dir,
                train=True,
                transform=transform_train,
                download=True
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True
        )
        test_dataloader = DataLoader(
            dataset=CIFAR10(
                root=dataset_dir,
                train=False,
                transform=transform_test,
                download=True
            ),
            batch_size=batch_size * 16,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )
    elif dataset_name == 'NMNIST':
        from spikingjelly.datasets.n_mnist import NMNIST
        train_dataloader = DataLoader(
            dataset=NMNIST(
                root=dataset_dir,
                data_type='frame',
                train=True,
                # use_frame = True,
                frames_number=T,
                split_by=split_by,
                # normalization=normalization
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True
        )
        test_dataloader = DataLoader(
            dataset=NMNIST(
                root=dataset_dir,
                train=False,
                # use_frame = True,
                frames_number=T,
                split_by=split_by,
                # normalization=normalization
            ),
            batch_size=batch_size * 4,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )
    elif dataset_name == 'CIFAR10DVS':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        train_dataloader = DataLoader(
            dataset=CIFAR10DVS(
                root=dataset_dir,
                data_type='frame',
                split_by=split_by,
                frames_number=T,
                # train = True,
                # split_ratio=0.9,
                # use_frame=True,
                # normalization=normalization
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True
        )
        test_dataloader = DataLoader(
            dataset=CIFAR10DVS(
                root=dataset_dir,
                data_type='frame',
                split_by=split_by,
                frames_number=T,
                # train = False,
                # split_ratio=0.9,
                # use_frame=True,
                # normalization=normalization
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )
    elif dataset_name == 'ASLDVS':
        class_num = 24
        from spikingjelly.datasets.asl_dvs import ASLDVS
        train_dataloader = DataLoader(
            dataset=ASLDVS(
                root=dataset_dir,
                data_type='frame',
                frames_number=T,
                split_by=split_by,
                # dataset_dir, train=True, split_ratio=0.8, use_frame=True, frames_num=T,
                #                            split_by=split_by, normalization=normalization
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True
        )
        test_dataloader = DataLoader(
            dataset=ASLDVS(
                root=dataset_dir,
                data_type='frame',
                frames_number=T,
                split_by=split_by,
                # dataset_dir, train=True, split_ratio=0.8, use_frame=True, frames_num=T,
                #                            split_by=split_by, normalization=normalization
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )
    elif dataset_name == 'DVS128Gesture':
        class_num = 11
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        train_dataloader = DataLoader(
            dataset=DVS128Gesture(
                root=dataset_dir,
                train=True,
                data_type='frame',
                frames_number=T,
                split_by=split_by
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True
        )
        test_dataloader = DataLoader(
            dataset=DVS128Gesture(
                root=dataset_dir,
                train=False,
                data_type='frame',
                frames_number=T,
                split_by=split_by
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )

    checkpoint_path = os.path.join(pt_dir, 'checkpoint.pt')
    checkpoint_max_path = os.path.join(pt_dir, 'check_point_max.pt')

    net_max_path = os.path.join(pt_dir, 'net_max.pt')
    optimizer_max_path = os.path.join(pt_dir, 'optimizer_max.pt')
    scheduler_max_path = os.path.join(pt_dir, 'scheduler_max.pt')
    checkpoint = None

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net = checkpoint['net']
        print(net.train_times, net.max_test_accuracy)
    else:
        if dataset_name == 'MNIST':
            net = Models.MNISTNet(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                                  alpha_learnable=alpha_learnable, detach_reset=detach_reset).to(device)

if __name__ == "__main__":
    main()