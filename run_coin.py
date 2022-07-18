import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random

import os
import argparse
import numpy as np

from s4 import S4
from tqdm.auto import tqdm
from models import ViS4mer

from datasets.coin_dataset import CustomDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
set_seed(1112)

print('Device', torch.cuda.device_count())

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Optimizer
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
# Scheduler
parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
# Dataset
parser.add_argument('--dataset', default='lvu', choices=['mnist', 'cifar10','lvu'], type=str, help='Dataset')
parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
# Dataloader
parser.add_argument('--num_workers', default=16, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--train_batch_size', default=16, type=int, help='Batch size')
parser.add_argument('--eval_batch_size', default=16, type=int, help='Batch size')
parser.add_argument('--l_secs', default=64, type=int, help='l_secs')
# Model
parser.add_argument('--n_layers', default=3, type=int, help='Number of layers')
parser.add_argument('--d_model', default=1024, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.2, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
parser.add_argument('--d_input', default=1024, type=int, help='Input dimension')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')

parser.add_argument('--feature_type', default='temporal_mean_pooling', type=str, help='Feature type')
parser.add_argument('--pool_type', default='mean', choices=['max', 'mean'], type=str, help='Pool type')
parser.add_argument('--long_term_task', default='writer', type=str, help='long_term_task')
parser.add_argument('--num_long_term_classes', default=10, type=int, help='num_long_term_classes')

parser.add_argument('--spatial_pool', default=0, type=int, help='First pooling layer')



# CUDA_VISIBLE_DEVICES=1 python example.py --feature_type cls --l_secs 60

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()


def setup_optimizer(model, lr, weight_decay, patience):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
    )

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in set(frozenset(hp.items()) for hp in hps)
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
                             f"Optimizer group {i}",
                             f"{len(g['params'])} tensors",
                         ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

def train(args, trainloader, model, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (video_name_batch, inputs, targets) in pbar:
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if args.num_long_term_classes == -1:
            targets = targets.to(torch.float32)
            outputs = outputs[:, 0]

        # if args.num_long_term_classes == -1:
        #     criterion = torch.nn.L1Loss(reduction='mean')

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if args.num_long_term_classes > 0:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if args.num_long_term_classes > 0:
            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(trainloader), train_loss / (batch_idx + 1), 100. * correct / total, correct, total)
            )
        else:
            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f' %
                (batch_idx, len(trainloader), train_loss / (batch_idx + 1))
            )

def eval(args, dataloader, model, epoch, criterion, split):
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0

    long_term_top1 = 0
    all_preds = []
    long_term_count = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (video_name_batch, inputs, targets) in pbar:
            inputs, targets = inputs.to(args.device), targets.to(args.device)   #.contiguous()
            outputs = model(inputs)

            if args.num_long_term_classes == -1:
                targets = targets.to(torch.float32)
                outputs = outputs[:, 0]

            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            if args.num_long_term_classes > 0:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            lt_pred = outputs.cpu()
            lt_labels = targets

            all_preds.append((video_name_batch, lt_pred, lt_labels))

            if args.num_long_term_classes > 0:
                long_term_top1 += correct
                long_term_count += targets.shape[0]

            if args.num_long_term_classes > 0:
                pbar.set_description(
                    'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (batch_idx, len(dataloader), eval_loss / (batch_idx + 1), 100. * correct / total, correct, total)
                )
            else:
                pbar.set_description(
                    'Batch Idx: (%d/%d) | Loss: %.3f' %
                    (batch_idx, len(dataloader), eval_loss / (batch_idx + 1))
                )

    clip_mse = []
    split_result = {}
    pred_agg = {}
    video_label = {}

    for video_name_batch, pred_batch, label_batch in all_preds:
        for i in range(len(video_name_batch)):
            v_name = video_name_batch[i]
            if v_name not in pred_agg:
                if args.num_long_term_classes > 0:
                    pred_agg[v_name] = softmax(pred_batch[i])
                else:
                    pred_agg[v_name] = [pred_batch[i]]
                video_label[v_name] = label_batch[i].cpu()
            else:
                if args.num_long_term_classes > 0:
                    pred_agg[v_name] += softmax(pred_batch[i])
                else:
                    pred_agg[v_name].append(pred_batch[i])

                assert video_label[v_name] == label_batch[i].cpu()

            if args.num_long_term_classes == -1:
                clip_mse.append(
                    (pred_batch[i] - label_batch[i]) ** 2.0
                )

    agg_sm_correct, agg_count = 0.0, 0.0
    mse = []

    for v_name in pred_agg.keys():
        if args.num_long_term_classes > 0:
            if pred_agg[v_name].argmax() == video_label[v_name]:
                agg_sm_correct += 1
        else:
            mse.append(
                (np.mean(pred_agg[v_name]) - video_label[v_name]) ** 2.0
            )
        agg_count += 1
        if args.num_long_term_classes > 0:
            acc = 100.0 * agg_sm_correct / agg_count
            split_result[split] = f'{acc} {agg_sm_correct} {agg_count}'
        else:
            split_result[split] = f'{np.mean(mse)} {len(mse)}'

    #print(split_result)

    # with open(args.output_eval_file, "a") as writer:
    #     if split == 'val':
    #         writer.write("Epoch trained %s\n" % str(epoch))
    #     for key in sorted(split_result.keys()):
    #         writer.write("%s = %s\n" % (key, str(split_result[key])))
    # writer.close()

    if args.num_long_term_classes > 0:
        return acc
    else:
        return np.mean(mse)

tasks = [('coin', 180)]

def main():
    args = parser.parse_args()
    for task, num_long_term_classes  in tasks:
        args.long_term_task = task
        args.num_long_term_classes = num_long_term_classes

        if args.num_long_term_classes > 0:
            args.d_output = args.num_long_term_classes
        else:
            args.d_output = 1

        if args.feature_type == 'temporal_mean_pooling':
            args.l_max = args.l_secs*49

        args.out_dir = f'outputs'
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        args.output_eval_file = f'{args.out_dir}/{args.long_term_task}.txt'

        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(args)

        print(f'==> Preparing {args.dataset} data..')

        trainset = CustomDataset(args=args, split='training')
        valset = CustomDataset(args=args, split='testing')

        # Dataloaders
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

        # Model
        print('==> Building model..')
        model = ViS4mer(
            d_input=args.d_input,
            l_max=args.l_max,
            d_output=args.d_output,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            prenorm=True,
        )

        model = model.to(args.device)

        if args.device == 'cuda':
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        if args.num_long_term_classes > 0:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        optimizer, scheduler = setup_optimizer(
            model, lr=args.lr, weight_decay=args.weight_decay, patience=args.patience
        )

        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        pbar = tqdm(range(start_epoch, start_epoch + 500))
        for epoch in pbar:
            if epoch == 0:
                pbar.set_description('Epoch: %d' % (epoch))
            else:
                pbar.set_description('Epoch: %d' % (epoch))
                # pbar.set_description('Epoch: %d | Val acc: %1.3f' % (epoch, val_acc))

            train(args=args, trainloader=trainloader, model=model, optimizer=optimizer, criterion=criterion)
            # for i, param_group in enumerate(optimizer.param_groups):
            #     print(f'learning rate param group {i}', param_group['lr'])

            val_acc = eval(args=args, dataloader=valloader, model=model,
                               epoch=epoch + 1, criterion=criterion, split='val')
            print('Epoch ', epoch + 1, 'acc :', val_acc)
                # eval(args=args, dataloader=testloader, model=model,
                #            epoch=epoch + 1, criterion=criterion, split='test')
                # with open(args.output_eval_file, "a") as writer:
                #     for i, param_group in enumerate(optimizer.param_groups):
                #         lr = param_group['lr']
                #         print(f'learning rate param group {i} : {lr}')
                #         writer.write(f'learning rate param group {i} : {lr}')
                #     writer.write('\n\n')
                # writer.close()
            with open(args.output_eval_file, "a") as writer:
                writer.write(f'acc epoch : {epoch} : {val_acc}\n')
            writer.close()
            scheduler.step(val_acc)

if __name__ == "__main__":
    main()
