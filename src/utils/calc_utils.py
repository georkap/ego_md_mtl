# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:34:58 2018

calc_utils

@author: Γιώργος
"""

import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from src.utils.file_utils import print_and_save
from src.utils.epic_eval_utils import get_manyhot_classes
from src.constants import *

def update_per_dataset_metrics(metrics, outputs_for_dat, targets_for_dat, dataset_loss, partial_task_losses,
                               num_cls_tasks, dataset_batch_size, is_training, multioutput_loss, t_attn):
    if t_attn:
        cls_losses, gaze_coord_losses, hand_coord_losses, object_losses, obj_cat_losses, temporal_task_losses = partial_task_losses
    else:
        cls_losses, gaze_coord_losses, hand_coord_losses, object_losses, obj_cat_losses = partial_task_losses

    metrics['losses'].update(dataset_loss.item(), dataset_batch_size)
    for i in range(num_cls_tasks):
        if multioutput_loss:
            sum_outputs = torch.zeros_like(outputs_for_dat[i][0])
            for o in outputs_for_dat[i]:
                sum_outputs += o.softmax(-1)
            t1, t5 = accuracy(sum_outputs.detach().cpu(), targets_for_dat[i].detach().cpu().long(), topk=(1, 5))
        else:
            t1, t5 = accuracy(outputs_for_dat[i].detach().cpu(), targets_for_dat[i].detach().cpu().long(), topk=(1, 5))
        metrics['top1_meters'][i].update(t1.item(), dataset_batch_size)
        metrics['top5_meters'][i].update(t5.item(), dataset_batch_size)
        if is_training:
            if multioutput_loss:
                for j in range(multioutput_loss): # dfb size
                    metrics['cls_loss_meters'][i][j].update(cls_losses[i*multioutput_loss + j].item(), dataset_batch_size)
            else:
                metrics['cls_loss_meters'][i].update(cls_losses[i].item(), dataset_batch_size)
    if is_training and t_attn:
        metrics['losses_temporal'].update(sum(temporal_task_losses).item(), dataset_batch_size)
    if is_training:
        for i, gl in enumerate(gaze_coord_losses):
            metrics['losses_gaze'][i].update(gl.item(), dataset_batch_size)
        for i, hl in enumerate(hand_coord_losses):
            metrics['losses_hands'][i].update(hl.item(), dataset_batch_size)
        for i, ol in enumerate(object_losses):
            metrics['losses_objects'][i].update(ol.item(), dataset_batch_size)
        for i, ol in enumerate(obj_cat_losses):
            metrics['losses_obj_cat'][i].update(ol.item(), dataset_batch_size)


def init_test_metrics(tasks_per_dataset):
    dataset_metrics = list()
    for i, dat in enumerate(tasks_per_dataset):
        dataset_metrics.append(dict())
        num_cls_tasks = dat['num_cls_tasks']
        losses = AverageMeter()
        top1_meters = [AverageMeter() for _ in range(num_cls_tasks)]
        top5_meters = [AverageMeter() for _ in range(num_cls_tasks)]
        dataset_metrics[i]['losses'] = losses
        dataset_metrics[i]['top1_meters'] = top1_meters
        dataset_metrics[i]['top5_meters'] = top5_meters
    return dataset_metrics

def init_training_metrics(tasks_per_dataset, multioutput_loss, t_attn):
    batch_time = AverageMeter()
    full_losses = AverageMeter()
    dataset_metrics = list()
    for i, dataset in enumerate(tasks_per_dataset):
        dataset_metrics.append(dict())
        num_cls_tasks = dataset['num_cls_tasks']
        num_g_tasks = dataset['num_g_tasks']
        num_h_tasks = dataset['num_h_tasks']
        num_o_tasks = dataset['num_o_tasks']
        num_c_tasks = dataset['num_c_tasks']
        losses = AverageMeter()
        cls_loss_meters = []
        for _ in range(num_cls_tasks):
            cls_loss_meters.append([AverageMeter() for _ in range(multioutput_loss)] if multioutput_loss else AverageMeter()) # 4 is the dfb size, 3 is the lstm mtl -> to generalize
        losses_hands = [AverageMeter() for _ in range(num_h_tasks)]
        losses_gaze = [AverageMeter() for _ in range(num_g_tasks)]
        losses_objects = [AverageMeter() for _ in range(num_o_tasks)]
        losses_obj_cat = [AverageMeter() for _ in range(num_c_tasks)]
        top1_meters = [AverageMeter() for _ in range(num_cls_tasks)]
        top5_meters = [AverageMeter() for _ in range(num_cls_tasks)]
        dataset_metrics[i]['losses'] = losses
        dataset_metrics[i]['cls_loss_meters'] = cls_loss_meters
        dataset_metrics[i]['losses_hands'] = losses_hands
        dataset_metrics[i]['losses_gaze'] = losses_gaze
        dataset_metrics[i]['losses_objects'] = losses_objects
        dataset_metrics[i]['losses_obj_cat'] = losses_obj_cat
        dataset_metrics[i]['top1_meters'] = top1_meters
        dataset_metrics[i]['top5_meters'] = top5_meters
        if t_attn:
            losses_temporal = AverageMeter()
            dataset_metrics[i]['losses_temporal'] = losses_temporal
    return batch_time, full_losses, dataset_metrics


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    if maxk > output.shape[1]:
        maxk = output.shape[1]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def rec_prec_per_class(conf_matrix):
    # cm is inversed from the wikipedia example on 3/8/18
    tp = np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - tp
    fn = np.sum(conf_matrix, axis=1) - tp
    
    with np.errstate(divide='warn'):
        precision = np.nan_to_num(tp/(tp+fp))
        recall = np.nan_to_num(tp/(tp+fn))
    
    return np.around(100*recall, 2), np.around(100*precision, 2)

def analyze_preds_labels(preds, labels, all_class_indices):
    cf = confusion_matrix(labels, preds, all_class_indices).astype(int)
    recall, precision = rec_prec_per_class(cf)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    with np.errstate(divide='warn'):
        cls_acc = np.around(100*np.nan_to_num(cls_hit / cls_cnt), 2)
    mean_cls_acc = np.mean(cls_acc)
    top1_acc = np.around(100*(np.sum(cls_hit)/np.sum(cf)), 3)

    return cf, recall, precision, cls_acc, mean_cls_acc, top1_acc

def avg_rec_prec_trimmed(pred, labels, valid_class_indices, all_class_indices):
    cm = confusion_matrix(labels, pred, all_class_indices).astype(float)
    
    cm_trimmed_cols = cm[:, valid_class_indices]
    cm_trimmed_rows = cm_trimmed_cols[valid_class_indices, :]
    
    recall, precision = rec_prec_per_class(cm_trimmed_rows)
    
    return np.sum(precision)/len(precision), np.sum(recall)/len(recall), cm_trimmed_rows.astype(int)

def eval_final_print_mt(video_preds, video_labels, dataset, task_id, current_classes, log_file, annotations_path=None,
                        val_list=None, task_type='None', actions_file=None):
    cf, recall, precision, cls_acc, mean_cls_acc, top1_acc = analyze_preds_labels(video_preds, video_labels,
                                                                                  all_class_indices=
                                                                                  list(range(int(current_classes))))
    print_and_save("Task {}".format(task_id), log_file)
    print_and_save(cf, log_file)

    if dataset == 'epick' and annotations_path:
        split_type = os.path.basename(os.path.dirname(val_list)).split('_')[-2:]
        split_type = split_type[-2] if split_type[-1] == 'act' else split_type[-1]
        valid_action_indices, valid_verb_indices, verb_ids_sorted, valid_noun_indices, noun_ids_sorted = \
            get_manyhot_classes(annotations_path, val_list, split_type, 100, actions_file)
        if task_type == 'A':
            valid_indices = valid_action_indices
            all_indices = list(range(EPIC_CLASSES[0]))
        if task_type == 'V':
            valid_indices, ids_sorted = valid_verb_indices, verb_ids_sorted
            all_indices = list(range(EPIC_CLASSES[1]))
        elif task_type == 'N':
            valid_indices, ids_sorted = valid_noun_indices, noun_ids_sorted
            all_indices = list(range(EPIC_CLASSES[2]))
        ave_pre, ave_rec, _ = avg_rec_prec_trimmed(video_preds, video_labels, valid_indices, all_indices)
        print_and_save("{} > 100 instances at training:".format(task_type), log_file)
        print_and_save("Classes are {}".format(valid_indices), log_file)
        print_and_save("average precision {0:02f}%, average recall {1:02f}%".format(ave_pre, ave_rec), log_file)
        if task_type != 'A':
            print_and_save("Most common {} in training".format(task_type), log_file)
            print_and_save("15 {} rec {}".format(task_type, recall[ids_sorted[:15]]), log_file)
            print_and_save("15 {} pre {}".format(task_type, precision[ids_sorted[:15]]), log_file)

    print_and_save("Cls Rec {}".format(recall), log_file)
    print_and_save("Cls Pre {}".format(precision), log_file)
    print_and_save("Cls Acc {}".format(cls_acc), log_file)
    print_and_save("Mean Cls Acc {:.02f}%".format(mean_cls_acc), log_file)
    print_and_save("Dataset Acc {}".format(top1_acc), log_file)
    return mean_cls_acc, top1_acc


if __name__ == '__main__':
    _annotations_file = r"D:\Datasets\egocentric\EPIC_KITCHENS\EPIC_train_action_labels.csv"
    _splits_file = r"other\splits\epic_rgb_nd_brd_act\epic_rgb_train_1.txt"
    _split_type = os.path.basename(os.path.dirname(_splits_file)).split('_')[-2:]
    _split_type = _split_type[-2] if _split_type[-1] == 'act' else _split_type[-1]
    _num_instances = 100
    _actions_file = r"D:\Datasets\egocentric\EPIC_KITCHENS\EPIC_action_classes.csv"
    _valid_action_indices, _valid_verb_indices, _verb_ids_sorted, _valid_noun_indices, _noun_ids_sorted = \
        get_manyhot_classes(_annotations_file, _splits_file, _split_type, _num_instances, _actions_file)
    print(_valid_action_indices)
    print(_valid_verb_indices)
    print(_verb_ids_sorted)
    print(_valid_noun_indices)
    print(_noun_ids_sorted)

