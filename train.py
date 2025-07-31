import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from distill_model import TSmodel
from datasets import build_dataset
from samplers import RASampler

import utils
import os
# import math
import numpy as np
# from sklearn.metrics import roc_curve
import random


# Check GPU, connect to it if it is available 
device = '0'
if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA is available. GPU will be used for training.")
else:
    device = 'cpu'

# Set up arguments parser
def get_args_parser():
    parser = argparse.ArgumentParser('HFMC-FAS', add_help=False)

    parser.add_argument('--seed', default=42, type=int, 
                        help='global random seed for reproducibility')
    parser.add_argument('--epochs', default=300, type=int)

    parser.add_argument('--data-path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='FACE', choices=['FACE','FACE_TEST'],
                        type=str, help='dataset set')
    parser.add_argument('--save_model_path', default='', type=str,
                        help='model path')
    
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-beta', default=1.0, type=float)
    parser.add_argument('--distillation-tau', default=3.0, type=float, help="")

    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')

    parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--distributed', action='store_true')

    return parser

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    # Set global random seed
    seed = 42
    set_seed(seed)
    print(f"Set global random seed to: {seed}")

    # Preparing Data
    print("==> Prepairing data ...")

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # Create dataloader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=lambda worker_id: set_seed(args.seed + worker_id) # add seed to make workers different
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    length_train = len(dataset_train)
    length_validation = len(dataset_val)
    
    # import model
    model = TSmodel()

    # Pass model to GPU
    model = model.to(device)
    model.train()

    # Set up optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    criterion = torch.nn.CrossEntropyLoss()

    # Training
    max_accuracy = 0.0
    min_HTER = 1.0
    
    dict = {'Train Loss':[], 'Train Acc':[], 'Validation Loss':[], 'Validation Acc':[]}
    
    # Train the model
    for epoch in range(args.epochs):
        print("\nEpoch:", epoch+1, "/", args.epochs)

        running_loss = 0
        correct = 0
        
        for i, (images, target) in enumerate(data_loader_train):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            model.train()
            optimizer.zero_grad()

            teacher_output, teacher_features, student_output, student_features = model(images)
            
            feature_loss = []

            # Calculate FMCD loss
            for i in range(0, len(teacher_features), 3):
                teacher_features[i] = teacher_features[i].reshape(args.batch_size, 768, 196)
                teacher_features[i] = torch.transpose(teacher_features[i], 1, 2)

                batch = 0
                fea_sum_loss = 0

                for tea_fea, stu_fea in zip(teacher_features[i], student_features[i]):
                    tea_fea = F.normalize(tea_fea, dim=-1) # (N, 768)
                    stu_fea = F.normalize(stu_fea, dim=-1) # (N, 192)
                    corr_ts1 = torch.mm(tea_fea.transpose(0, 1),stu_fea) # (768*192)
                    rec_tea = torch.mm(tea_fea,corr_ts1) # (N, 192)
                    fea_loss1 = (rec_tea - stu_fea).pow(2).mean()
                    corr_ts2 = torch.mm(stu_fea.transpose(0, 1),tea_fea) # (192*768)
                    rec_stu = torch.mm(stu_fea,corr_ts2) # (N, 768)
                    fea_loss2 = (tea_fea - rec_stu).pow(2).mean()
                    fea_loss = fea_loss1 + fea_loss2
                    batch += 1
                    fea_sum_loss += fea_loss
                fea_sum_loss = fea_sum_loss/batch
                feature_loss.append(fea_sum_loss)
            fea_ts_loss = sum(feature_loss)/len(feature_loss)
            
            # Calculate CE loss
            classification_loss = (criterion(teacher_output, target) + criterion(student_output, target))/2
            
            # Calculate LLMD loss
            T = args.distillation_tau
            output_loss1 = F.kl_div(F.log_softmax(teacher_output / T, dim=1),F.log_softmax(student_output / T, dim=1),reduction='batchmean',log_target=True) * (T * T)
            output_loss2 = F.kl_div(F.log_softmax(student_output / T, dim=1),F.log_softmax(teacher_output / T, dim=1),reduction='batchmean',log_target=True) * (T * T)
            output_loss = (output_loss1 + output_loss2)/2
            
            # Calculate total loss
            total_loss = (args.distillation_alpha)*fea_ts_loss + (1-args.distillation_beta)*classification_loss + (args.distillation_beta)*output_loss
 
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += total_loss.item()

            train_output = (teacher_output + student_output)/2 
            _, pre_outputs = torch.max(train_output.data, 1)
            correct += (pre_outputs == target).sum().item()
        scheduler.step()

        Final_loss = running_loss/len(data_loader_train)
        Accuracy = 100*correct/length_train

        dict['Train Loss'].append(Final_loss)
        dict['Train Acc'].append(Accuracy)

        print('Train Loss: {:.2f}'.format(Final_loss))

        print('Train Accuracy: {:.2f}%'.format(Accuracy))
        
        # Val the model
        Final_loss, Accuracy = evaluate(data_loader_val, model, device,length_validation)

        dict['Validation Loss'].append(Final_loss)
        dict['Validation Acc'].append(Accuracy)

        print('Validation Loss: {:.2f}'.format(Final_loss))

        print('Validation Accuracy: {:.2f}%'.format(Accuracy))

        FAR, FRR, HTER = eval_hter(data_loader_val, model, device)
        print(f"* FAR {FAR*100:.2f}% FRR {FRR*100:.2f}%  HTER {HTER*100:.2f}%")
        
        # Save the model if you get the lowest HTER or EER on validation data
        if min_HTER > HTER:
            min_HTER = HTER
            print('Saving the model ...')
            model.eval()
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            
            base_path = args.save_model_path
            if base_path.endswith('.pth'):
                base_path = base_path[:-4]
            
            # Save the teacher and student models separately
            teacher_path = base_path + '_teacher.pth'
            torch.save(model.teacher.state_dict(), teacher_path)
            print(f'Saved teacher model to: {teacher_path}')
            
            student_path = base_path + '_student.pth'
            torch.save(model.student.state_dict(), student_path)
            print(f'Saved student model to: {student_path}')

        max_accuracy = max(max_accuracy, Accuracy)
        print(f'Max accuracy: {max_accuracy:.2f}%')
        print(f"Min HTER: {min_HTER*100:.2f}%")
        
    print("TRAINING IS FINISHED !!!")

    return

@torch.no_grad()
def evaluate(data_loader, model, device,length_validation):
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for images, target in (data_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            model.eval()
            teacher_output, teacher_features, student_output, student_features = model(images)
            val_output = (teacher_output+student_output)/2
            
            loss = criterion(val_output, target)
            total_loss += loss.item()

            _, pre_outputs = torch.max(val_output.data, 1)
            correct += (pre_outputs == target).sum().item()

        Final_loss = total_loss/len(data_loader)
        Accuracy = 100*correct/length_validation

    return Final_loss, Accuracy

@torch.no_grad()
def eval_hter(data_loader, model, device):

    # switch to evaluation mode
    model.eval()

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for images, target in (data_loader):

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            teacher_output, teacher_features, student_output, student_features = model(images)
        
        val_output = (teacher_output+student_output)/2
        
        y_p = []
        for i in val_output:
            if i[0] > i[1]:
                y_p.append(0)
            else:
                y_p.append(1)

        y_t = target.cpu().numpy()
        
        for i in range(len(y_t)):
            if y_p[i] == 0 and y_t[i] == 0:
                TN += 1
            elif y_p[i] == 0 and y_t[i] == 1:
                FN += 1
            elif y_p[i] == 1 and y_t[i] == 1:
                TP += 1
            elif y_p[i] == 1 and y_t[i] == 0:
                FP += 1

    FAR = FP/(FP+TN)
    FRR = FN/(FN+TP)
    HTER = (FAR + FRR)/2

    return FAR, FRR, HTER


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Mymodel training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)
    print(args)
    main(args)

