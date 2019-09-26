# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:34:58 2018

calc_utils

@author: Γιώργος
"""

import os
import numpy as np
from sklearn.metrics import confusion_matrix
from src.utils.file_utils import print_and_save
from src.utils.epic_eval_utils import get_manyhot_classes
from src.constants import *


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
        if task_type == 'V': # 'Verbs': error prone if I ever train nouns on their own
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

