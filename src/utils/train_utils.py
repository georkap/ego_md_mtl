# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:53:37 2018

Functions that are used in training the network.

1) Cyclic learning rate scheduler from:
    https://github.com/thomasjpfan/pytorch/blob/master/torch/optim/lr_scheduler.py


@author: Γιώργος
"""
import time
import torch

from src.utils.calc_utils import AverageMeter, accuracy, init_training_metrics, init_test_metrics, update_per_dataset_metrics
from src.utils.learning_rates import CyclicLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.losses.mtl_losses import get_mtl_losses, multiobjective_gradient_optimization
from src.utils.eval_utils import *

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda(device=0)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    pred = pred.to(torch.device("cuda:{}".format(0)))
    y_a = y_a.to(torch.device("cuda:{}".format(0)))
    y_b = y_b.to(torch.device("cuda:{}".format(0)))
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def init_inputs(data, use_flow, base_gpu):
    if use_flow:
        rgb, flow, targets, masks, dataset_ids = data
        rgb = rgb.cuda(base_gpu)
        flow = flow.cuda(base_gpu)
        inputs = [rgb, flow]
    else:
        inputs, targets, masks, dataset_ids = data
        inputs = inputs.cuda(base_gpu)
    return inputs, targets, masks, dataset_ids

def init_batch(tasks_per_dataset, dataset_ids):
    # identify the dataset that the samples of the batch belong to
    batch_ids_per_dataset = []
    for dat in range(len(tasks_per_dataset)): # num_datasets = len(tasks_per_dataset)
        batch_ids = []
        batch_ids_per_dataset.append(batch_ids)
    for batch_ind, dataset_id in enumerate(dataset_ids):
        batch_ids_per_dataset[dataset_id].append(batch_ind)
    return batch_ids_per_dataset

def init_inputs_batch(data, tasks_per_dataset, use_flow, base_gpu):
    inputs, targets, masks, dataset_ids = init_inputs(data, use_flow, base_gpu)
    batch_ids_per_dataset = init_batch(tasks_per_dataset, dataset_ids)
    return inputs, targets, masks, dataset_ids, batch_ids_per_dataset

# def update_metrics_per_dataset(dataset_metrics, outputs_per_dataset, targets_per_dataset, loss_per_dataset,
#                                partial_loss_per_dataset, batch_ids_per_dataset, tasks_per_dataset, is_training, dfb):
#     # update metrics
#     for dataset_id in range(len(tasks_per_dataset)):
#         num_cls_tasks = tasks_per_dataset[dataset_id]['num_cls_tasks']
#         loss = loss_per_dataset[dataset_id]
#         partial_losses = partial_loss_per_dataset[dataset_id]
#         dataset_batch_size = len(batch_ids_per_dataset[dataset_id])
#         update_per_dataset_metrics(dataset_metrics[dataset_id], outputs_per_dataset[dataset_id],
#                                    targets_per_dataset[dataset_id], loss, partial_losses, num_cls_tasks,
#                                    dataset_batch_size, is_training, dfb)

def calc_losses_per_dataset(network_output, targets, masks, tasks_per_dataset, batch_ids_per_dataset, one_obj_layer,
                            is_training, base_gpu, dataset_metrics, multioutput_loss=0):
    (outputs, coords, heatmaps, probabilities, objects, obj_cat) = network_output

    full_loss = []
    partial_loss = []
    outputs_per_dataset = []
    targets_per_dataset = []
    masks_per_dataset = []
    objects_per_dataset = []
    obj_cat_per_dataset = []
    global_task_id = 0  # assign indices to keep track of the tasks because the model outputs come in a sequence
    global_coord_id = 0
    global_object_id = 0
    global_obj_cat_id = 0
    # calculate losses for the tasks of each dataset individually
    for dataset_id in range(len(tasks_per_dataset)):
        num_cls_tasks = tasks_per_dataset[dataset_id]['num_cls_tasks']
        num_g_tasks = tasks_per_dataset[dataset_id]['num_g_tasks']
        num_h_tasks = tasks_per_dataset[dataset_id]['num_h_tasks']
        num_o_tasks = tasks_per_dataset[dataset_id]['num_o_tasks']
        num_c_tasks = tasks_per_dataset[dataset_id]['num_c_tasks']
        num_coord_tasks = num_g_tasks + 2 * num_h_tasks
        # needs transpose to get the first dim to be the task and the second dim to be the batch
        tmp_targets = targets[batch_ids_per_dataset[dataset_id]].cuda(base_gpu).transpose(0, 1)
        tmp_masks = masks[batch_ids_per_dataset[dataset_id]].cuda(base_gpu)  # the masks are for the coordinate tasks
        targets_per_dataset.append(tmp_targets)
        masks_per_dataset.append(tmp_masks)
        outputs_per_dataset.append([])
        objects_per_dataset.append([])
        obj_cat_per_dataset.append([])
        # when a dataset does not have any representative samples in a batch
        if not len(tmp_targets[0]) > 0:
            global_task_id += num_cls_tasks
            global_coord_id += num_coord_tasks
            global_object_id += num_o_tasks
            global_obj_cat_id += num_c_tasks
            continue
        # get model's outputs for the classification tasks for the current dataset
        for task_id in range(num_cls_tasks):
            # increase first dim of tmp_outputs to accomodate dfb model,
            # also added dimension to the non-dfb models to facilite code in this stage
            if multioutput_loss:
                tmp_outputs = []
                for o in outputs:
                    tmp_outputs.append(o[global_task_id + task_id][batch_ids_per_dataset[dataset_id]])
            else:
                tmp_outputs = outputs[global_task_id + task_id][batch_ids_per_dataset[dataset_id]]
            outputs_per_dataset[dataset_id].append(tmp_outputs)
        # get model's outputs for the coord tasks of the current dataset
        coo, hea, pro = None, None, None
        if coords is not None:
            coo = coords[batch_ids_per_dataset[dataset_id], :, global_coord_id:global_coord_id + num_coord_tasks, :]
            hea = heatmaps[batch_ids_per_dataset[dataset_id], :, global_coord_id:global_coord_id + num_coord_tasks, :]
            # pro = probabilities[batch_ids_per_dataset[dataset_ids], :]
        # get model's outputs for the object classification tasks of the current dataset
        counts = [None, None]
        object_outputs = None
        if num_o_tasks > 0:
            if one_obj_layer:
                object_outputs = objects[global_object_id][batch_ids_per_dataset[dataset_id]]
                counts[0] = object_outputs.shape[1]
            else:
                for no in range(len(objects[global_object_id])):
                    objects_per_dataset[dataset_id].append(
                        objects[global_object_id][no][batch_ids_per_dataset[dataset_id], :])
                object_outputs = objects_per_dataset[dataset_id]
                counts[0] = len(object_outputs)
        obj_cat_outputs = None
        if num_c_tasks > 0:
            if one_obj_layer:
                obj_cat_outputs = obj_cat[global_obj_cat_id][batch_ids_per_dataset[dataset_id]]
                counts[1] = obj_cat_outputs.shape[1]
            else:
                for no in range(len(obj_cat[global_obj_cat_id])):
                    obj_cat_per_dataset[dataset_id].append(
                        obj_cat[global_obj_cat_id][no][batch_ids_per_dataset[dataset_id], :])
                obj_cat_outputs = obj_cat_per_dataset[dataset_id]
                counts[1] = len(obj_cat_outputs)

        task_outputs = (outputs_per_dataset[dataset_id], coo, hea, pro, object_outputs, obj_cat_outputs)
        task_sizes = (num_cls_tasks, num_g_tasks, num_h_tasks, num_o_tasks, num_c_tasks)

        loss, partial_losses = get_mtl_losses(targets_per_dataset[dataset_id], masks_per_dataset[dataset_id],
                                              task_outputs, task_sizes, one_obj_layer, counts, is_training=is_training,
                                              multioutput_loss=multioutput_loss)

        global_task_id += num_cls_tasks
        global_coord_id += num_coord_tasks
        global_object_id += num_o_tasks
        global_obj_cat_id += num_c_tasks
        full_loss.append(loss)
        partial_loss.append(partial_losses)

        # update metrics
        dataset_batch_size = len(batch_ids_per_dataset[dataset_id])
        update_per_dataset_metrics(dataset_metrics[dataset_id], outputs_per_dataset[dataset_id],
                                   targets_per_dataset[dataset_id], loss, partial_losses, num_cls_tasks,
                                   dataset_batch_size, is_training, multioutput_loss)

    return sum(full_loss)
    # return full_loss, partial_loss, outputs_per_dataset, targets_per_dataset

def make_to_print(to_print, log_file, tasks_per_dataset, dataset_metrics, is_training,
                  full_losses=None, dataset_ids=None, full_loss=None, multioutput_loss=0):
    num_datasets = len(tasks_per_dataset)
    if is_training:
        if num_datasets > 1:
            batch_size = dataset_ids.shape[0]
            full_losses.update(full_loss.item(), batch_size)
            to_print += '[F_Loss {:.4f}[avg:{:.4f}]\n\t'.format(full_losses.val, full_losses.avg)

    for dataset_id in range(num_datasets):
        num_cls_tasks = tasks_per_dataset[dataset_id]['num_cls_tasks']
        top1_meters = dataset_metrics[dataset_id]['top1_meters']
        top5_meters = dataset_metrics[dataset_id]['top5_meters']

        if is_training:
            num_g_tasks = tasks_per_dataset[dataset_id]['num_g_tasks']
            num_h_tasks = tasks_per_dataset[dataset_id]['num_h_tasks']
            num_o_tasks = tasks_per_dataset[dataset_id]['num_o_tasks']
            num_c_tasks = tasks_per_dataset[dataset_id]['num_c_tasks']
            losses = dataset_metrics[dataset_id]['losses']
            losses_gaze = dataset_metrics[dataset_id]['losses_gaze']
            losses_hands = dataset_metrics[dataset_id]['losses_hands']
            losses_objects = dataset_metrics[dataset_id]['losses_objects']
            losses_obj_cat = dataset_metrics[dataset_id]['losses_obj_cat']
            cls_loss_meters = dataset_metrics[dataset_id]['cls_loss_meters']
            to_print += '[Losses {:.4f}[avg:{:.4f}], '.format(losses.val, losses.avg)
            for ind in range(num_cls_tasks):
                if multioutput_loss:
                    to_print += '\n\tT{}::loss '.format(ind)
                    if multioutput_loss == 4:
                        loss_names = ['avg', 'ch', 'max', 'sum']
                    elif multioutput_loss == 3:
                        loss_names = ['lstm', 'avg', 'sum']
                    else:
                        loss_names = []
                    for j, l_name in enumerate(loss_names):
                        to_print += ' -({}) {:.4f}[avg:{:.4f}] '.format(l_name, cls_loss_meters[ind][j].val,
                                                                        cls_loss_meters[ind][j].avg)
                else:
                    to_print += 'T{}::loss {:.4f}[avg:{:.4f}], '.format(ind, cls_loss_meters[ind].val,
                                                                        cls_loss_meters[ind].avg)
            if multioutput_loss:
                to_print += '\n\t'
            for ind in range(num_h_tasks):
                to_print += '[l_hcoo_{} {:.4f}[avg:{:.4f}], '.format(ind, losses_hands[ind].val, losses_hands[ind].avg)
            for ind in range(num_g_tasks):
                to_print += '[l_gcoo_{} {:.4f}[avg:{:.4f}], '.format(ind, losses_gaze[ind].val, losses_gaze[ind].avg)
            for ind in range(num_o_tasks):
                to_print += '[l_obj_{} {:.4f}[avg:{:.4f}], '.format(ind, losses_objects[ind].val,
                                                                    losses_objects[ind].avg)
            for ind in range(num_c_tasks):
                to_print += '[l_obj_cat_{} {:.4f}[avg:{:.4f}], '.format(ind, losses_obj_cat[ind].val,
                                                                        losses_obj_cat[ind].avg)
            for ind in range(num_cls_tasks):
                if ind == 0:
                    to_print += '\n\t\t'
                to_print += 'T{}::Top1 {:.3f}[avg:{:.3f}],Top5 {:.3f}[avg:{:.3f}],'.format(
                    ind, top1_meters[ind].val, top1_meters[ind].avg, top5_meters[ind].val, top5_meters[ind].avg)
            if dataset_id + 1 < num_datasets:
                to_print += "\n\t"
        else: # not training
            to_print += '\n\t'
            to_print = append_to_print_cls_results(to_print, num_cls_tasks, top1_meters, top5_meters)

    print_and_save(to_print, log_file)

def append_to_print_cls_results(to_print, num_cls_tasks, top1_meters, top5_meters):
    for ind in range(num_cls_tasks):
        to_print += '[T{}::Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]],'.format(
            ind, top1_meters[ind].val, top1_meters[ind].avg, top5_meters[ind].val, top5_meters[ind].avg)
    return to_print

def make_final_test_print(tasks_per_dataset, dataset_metrics, dataset_type, log_file):
    for dataset_id in range(len(tasks_per_dataset)):
        num_cls_tasks = tasks_per_dataset[dataset_id]['num_cls_tasks']
        losses = dataset_metrics[dataset_id]['losses']
        top1_meters = dataset_metrics[dataset_id]['top1_meters']
        top5_meters = dataset_metrics[dataset_id]['top5_meters']
        final_print = '{} Results: Loss {:.3f},'.format(dataset_type, losses.avg)
        for ind in range(num_cls_tasks):
            final_print += 'T{}::Top1 {:.3f}, Top5 {:.3f}, '.format(ind, top1_meters[ind].avg, top5_meters[ind].avg)
        print_and_save(final_print, log_file)


def train_mfnet_mo(model, optimizer, train_iterator, tasks_per_dataset, cur_epoch, log_file, gpus, lr_scheduler=None,
                   moo=False, use_flow=False, one_obj_layer=False, grad_acc_batches=None, multioutput_loss=0):
    batch_time, full_losses_metric, dataset_metrics = init_training_metrics(tasks_per_dataset, multioutput_loss)

    optimizer.zero_grad()
    model.train()
    is_training = True
    if not isinstance(lr_scheduler, CyclicLR) and not isinstance(lr_scheduler, ReduceLROnPlateau):
        lr_scheduler.step()

    print_and_save('*********', log_file)
    print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()

    if grad_acc_batches is not None:
        num_aggregated_batches = 0
        real_batch_idx = 0

    for batch_idx, data in enumerate(train_iterator):
        if isinstance(lr_scheduler, CyclicLR):
            lr_scheduler.step()

        inputs, targets, masks, dataset_ids, batch_ids_per_dataset = init_inputs_batch(data, tasks_per_dataset,
                                                                                       use_flow, gpus[0])

        if grad_acc_batches is not None:
            if num_aggregated_batches == 0:
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        network_output = model(inputs)
        full_loss = calc_losses_per_dataset(
            network_output, targets, masks, tasks_per_dataset, batch_ids_per_dataset, one_obj_layer, is_training,
            gpus[0], dataset_metrics, multioutput_loss)

        # implement grad accumulation
        if grad_acc_batches is not None:
            grad_acc_loss = full_loss / grad_acc_batches
            grad_acc_loss.backward()
            num_aggregated_batches += 1
            if num_aggregated_batches == grad_acc_batches:
                optimizer.step()
                batch_time.update(time.time() - t0)
                t0 = time.time()
                # print results
                real_batch_idx += 1
                to_print = '[Epoch:{}, Batch {}/{} in {:.3f} s, LR {:.6f}]'.format(
                    cur_epoch, real_batch_idx, len(train_iterator)//num_aggregated_batches, batch_time.val,
                    lr_scheduler.get_lr()[0])
                make_to_print(to_print, log_file, tasks_per_dataset, dataset_metrics, is_training, full_losses_metric,
                              dataset_ids, grad_acc_loss, multioutput_loss)
                num_aggregated_batches = 0
        else:
            full_loss.backward()
            optimizer.step()
            batch_time.update(time.time() - t0)
            t0 = time.time()
            # print results
            to_print = '[Epoch:{}, Batch {}/{} in {:.3f} s, LR {:.6f}]'.format(
                cur_epoch, batch_idx, len(train_iterator), batch_time.val, lr_scheduler.get_lr()[0])
            make_to_print(to_print, log_file, tasks_per_dataset, dataset_metrics, is_training, full_losses_metric,
                          dataset_ids, full_loss, multioutput_loss)

    print_and_save("Epoch train time: {}".format(batch_time.sum), log_file)

def test_mfnet_mo(model, test_iterator, tasks_per_dataset, cur_epoch, dataset_type, log_file, gpus, use_flow=False,
                  one_obj_layer=False, multioutput_loss=0):
    is_training = False
    dataset_metrics = init_test_metrics(tasks_per_dataset)

    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset_type), log_file)
        for batch_idx, data in enumerate(test_iterator):
            inputs, targets, masks, dataset_ids, batch_ids_per_dataset = init_inputs_batch(
                data, tasks_per_dataset, use_flow, gpus[0])

            network_output = model(inputs)
            _ = calc_losses_per_dataset(network_output, targets, masks, tasks_per_dataset, batch_ids_per_dataset,
                                        one_obj_layer, is_training, gpus[0], dataset_metrics, multioutput_loss)

            # print results
            to_print = '[Epoch:{}, Batch {}/{}]'.format(cur_epoch, batch_idx, len(test_iterator))
            make_to_print(to_print, log_file, tasks_per_dataset, dataset_metrics, is_training, multioutput_loss)

        make_final_test_print(tasks_per_dataset, dataset_metrics, dataset_type, log_file)

        task_top1s = list()
        for dataset_id in range(len(tasks_per_dataset)):
            top1_meters = dataset_metrics[dataset_id]['top1_meters']
            for tasktop1 in top1_meters:
                task_top1s.append(tasktop1.avg)

    return task_top1s

def validate_mfnet_mo(model, test_iterator, task_sizes, cur_epoch, dataset, log_file, use_flow=False, one_obj_layer=False,
                      multioutput_loss=0, eval_branch=None, eval_ensemble=None):
    num_cls_tasks, num_g_tasks, num_h_tasks, num_o_tasks, num_c_tasks = task_sizes
    losses = AverageMeter()
    top1_meters = [AverageMeter() for _ in range(num_cls_tasks)]
    top5_meters = [AverageMeter() for _ in range(num_cls_tasks)]
    task_outputs = [[] for _ in range(num_cls_tasks)]

    print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
    with torch.no_grad():
        model.eval()
        for batch_idx, data in enumerate(test_iterator):

            inputs, targets, masks, video_names = init_inputs(data=data, use_flow=use_flow, base_gpu=None)
            batch_size = targets.shape[0]
            targets = targets.cuda().transpose(0, 1)
            masks = masks.cuda()

            # outputs, coords, heatmaps, probabilities, objects, obj_cat = model(inputs)
            network_output = model(inputs)
            outputs, coords, heatmaps, probabilities, objects, obj_cat = network_output

            if eval_ensemble is not None: # only compatible for multioutput_loss ==False for now
                assert not multioutput_loss
                full_outputs, ens_outputs = outputs
                found = [[0 for __ in range(num_cls_tasks)] for _ in range(batch_size)]
                for task_id, task_out in enumerate(full_outputs):
                    for j, unbatched_outs in enumerate(task_out):
                        res = np.argmax(unbatched_outs.detach().cpu().numpy())
                        label = targets[task_id][j].detach().cpu().numpy()
                        if res == label: # found tp in outputs
                            found[j][task_id] = 1
                ensemble_outputs = full_outputs.copy()
                ens_found = [[0 for __ in range(num_cls_tasks)] for _ in range(batch_size)]
                connected = [['n' for __ in range(num_cls_tasks)] for _ in range(batch_size)]
                for ens_out in ens_outputs:
                    for task_id, task_out in enumerate(ens_out):
                        for j, unbatched_outs in enumerate(task_out):
                            res = np.argmax(unbatched_outs.detach().cpu().numpy())
                            label = targets[task_id][j].detach().cpu().numpy()
                            if res == label: # found tp in ensemble
                                ensemble_outputs[task_id][j] = unbatched_outs
                                ens_found[j][task_id] += 1
                                if connected[j][task_id] == 'n': # if 'n'ot connected set 'c'onnected to True
                                    connected[j][task_id] = 'c'
                                elif connected[j][task_id] == 'd': # 1 connection 'd'ropped and found new correct ensemble
                                    connected[j][task_id] = 'r' # 'r'econnected
                            else: # if an ensemble part is not correct
                                if connected[j][task_id] == 'c': # check if connected and set connection dropped
                                    connected[j][task_id] = 'd'

                outputs = ensemble_outputs

            if multioutput_loss:
                temp_outputs = []
                for task_id in range(num_cls_tasks):
                    tmp_outputs = []
                    for o in outputs:
                        tmp_outputs.append(o[task_id])
                    temp_outputs.append(tmp_outputs)
                outputs = temp_outputs

            counts = [0, 0]
            if objects is not None:
                objects = objects[0]
                counts[0] = objects.shape[1] if one_obj_layer else len(objects) # no dataset id, just use 0 to simulate
            if obj_cat is not None:
                obj_cat = obj_cat[0]
                counts[1] = obj_cat.shape[1] if one_obj_layer else len(obj_cat)

            per_task_outputs = (outputs, coords, heatmaps, probabilities, objects, obj_cat)
            loss, partial_losses = get_mtl_losses(targets, masks, per_task_outputs, task_sizes, one_obj_layer, counts,
                                                  is_training=False, multioutput_loss=multioutput_loss)

            # save predictions for evaluation afterwards
            batch_preds = []
            for j in range(batch_size):
                txt_batch_preds = "{}".format(video_names[j])
                for task_id in range(num_cls_tasks):
                    txt_batch_preds += ", "
                    if multioutput_loss:
                        sum_outputs = torch.zeros_like(outputs[task_id][0][j])
                        if eval_branch is not None:
                            sum_outputs += outputs[task_id][eval_branch][j].softmax(-1)
                        else:
                            for o in outputs[task_id]:
                                sum_outputs += o[j].softmax(-1)
                        res = np.argmax(sum_outputs.detach().cpu().numpy())
                    else:
                        res = np.argmax(outputs[task_id][j].detach().cpu().numpy())
                    label = targets[task_id][j].detach().cpu().numpy()
                    task_outputs[task_id].append([res, label])
                    txt_batch_preds += "T{} P-L:{}-{}".format(task_id, res, label)
                    if eval_ensemble:
                        txt_batch_preds += ": {}:{} : {}".format(ens_found[j][task_id], connected[j][task_id], found[j][task_id])
                batch_preds.append(txt_batch_preds)

            losses.update(loss.item(), batch_size)
            for task_id in range(num_cls_tasks):
                if multioutput_loss:
                    sum_outputs = torch.zeros_like(outputs[task_id][0])
                    if eval_branch is not None:
                        sum_outputs += outputs[task_id][eval_branch].softmax(-1)
                    else:
                        for o in outputs[task_id]:
                            sum_outputs += o.softmax(-1)
                    t1, t5 = accuracy(sum_outputs.detach().cpu(), targets[task_id].detach().cpu().long(), topk=(1, 5))
                else:
                    t1, t5 = accuracy(outputs[task_id].detach().cpu(), targets[task_id].detach().cpu().long(), topk=(1, 5))
                top1_meters[task_id].update(t1.item(), batch_size)
                top5_meters[task_id].update(t5.item(), batch_size)

            to_print = '[Batch {}/{}]'.format(batch_idx, len(test_iterator))
            to_print = append_to_print_cls_results(to_print, num_cls_tasks, top1_meters, top5_meters)
            to_print += '\n\t{}'.format(batch_preds)
            print_and_save(to_print, log_file)

        to_print = '{} Results: Loss {:.3f}'.format(dataset, losses.avg)
        for task_id in range(num_cls_tasks):
            to_print += ', T{}::Top1 {:.3f}, Top5 {:.3f}'.format(task_id, top1_meters[task_id].avg, top5_meters[task_id].avg)
        print_and_save(to_print, log_file)
    return [tasktop1.avg for tasktop1 in top1_meters], task_outputs


def validate_mfnet_mo_gaze(model, test_iterator, task_sizes, cur_epoch, dataset, log_file, use_flow=False, **kwargs):
    num_cls_tasks, num_g_tasks, num_h_tasks, num_o_tasks, num_c_tasks = task_sizes
    auc_frame, auc_temporal = AverageMeter(), AverageMeter()
    aae_frame, aae_temporal = AverageMeter(), AverageMeter()
    print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)

    with torch.no_grad():
        model.eval()
        frame_counter = 0
        actual_frame_counter = 0
        video_counter = 0
        for batch_idx, data in enumerate(test_iterator):
            inputs, targets, orig_gaze, video_names = init_inputs(data=data, use_flow=use_flow, base_gpu=None)
            batch_size = targets.shape[0]
            targets = targets.cuda().transpose(0, 1)
            orig_gaze = orig_gaze.cuda().transpose(0, 1)

            video_counter += 1
            to_print = '[Batch {}/{}]'.format(batch_idx, len(test_iterator))

            double_temporal_size = inputs.shape[2]
            temporal_size = double_temporal_size // 2

            cls_targets = targets[:num_cls_tasks, :].long()
            assert len(cls_targets) == num_cls_tasks

            gaze_targets = targets[num_cls_tasks:num_cls_tasks + 2*temporal_size, :].transpose(1, 0).reshape(-1, temporal_size, 1, 2)
            gaze_targets.squeeze_(2)
            gaze_targets = unnorm_gaze_coords(gaze_targets).cpu().numpy()

            orig_targets = orig_gaze[:2*temporal_size, :].transpose(1, 0).reshape(-1, temporal_size, 1, 2)
            orig_targets.squeeze_(2)

            # batch over the blocks of 16 frames for mfnet
            mf_blocks = double_temporal_size//16
            mf_remaining = double_temporal_size % 16
            for mf_i in range(mf_blocks):
                mf_inputs = inputs[:, :, mf_i*16:(mf_i+1)*16, :, :]
                mf_targets = gaze_targets[:, mf_i*8:(mf_i+1)*8]
                or_targets = orig_targets[:, mf_i*8:(mf_i+1)*8]

                auc_frame, auc_temporal, aae_frame, aae_temporal, frame_counter, actual_frame_counter = \
                    inner_batch_calc(model, mf_inputs, mf_targets, or_targets, frame_counter, actual_frame_counter,
                                     aae_frame, auc_frame, aae_temporal, auc_temporal, to_print, log_file)
            if mf_remaining > 0:
                mf_inputs = inputs[:, :, double_temporal_size-16:, :, :]
                mf_targets = gaze_targets[:, temporal_size-8:]
                or_targets = orig_targets[:, temporal_size-8:]

                auc_frame, auc_temporal, aae_frame, aae_temporal, frame_counter, actual_frame_counter = \
                    inner_batch_calc(model, mf_inputs, mf_targets, or_targets, frame_counter, actual_frame_counter,
                                     aae_frame, auc_frame, aae_temporal, auc_temporal, to_print, log_file,
                                     mf_remaining//2)

        to_print = 'Evaluated in total {}/{} frames in {} video segments.'.format(frame_counter, actual_frame_counter,
                                                                                  video_counter)
        print_and_save(to_print, log_file)

        return None, None

def validate_mfnet_mo_json(model, test_iterator, dataset, action_file):
    # TODO: add updated code for flow
    json_outputs = dict()
    json_outputs['version'] = '0.1'
    json_outputs['challenge'] = 'action_recognition'
    json_outputs['results'] = dict()

    import pandas
    all_action_ids = pandas.read_csv(action_file)

    print("Running on test set {} of EPIC".format(dataset))
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets, uid) in enumerate(test_iterator):
            inputs = inputs.cuda()
            outputs, _, _ = model(inputs)

            batch_size = outputs[0].size(0)

            for j in range(batch_size):
                json_outputs['results'][str(uid[j].item())] = {'verb': {}, 'noun': {}, 'action': {}}

                if len(outputs) >= 1: # if there is action prediction
                    action = outputs[0][j].detach().cpu().numpy().astype(np.float64)
                if len(outputs) >= 2: # if there is verb prediction
                    verb = outputs[1][j].detach().cpu().numpy().astype(np.float64)
                    for i, v in enumerate(verb):
                        json_outputs['results'][str(uid[j].item())]['verb'][str(i)] = v
                else:
                    for i in range(125):
                        json_outputs['results'][str(uid[j].item())]['verb'][str(i)] = round(float(0.0), ndigits=3)
                if len(outputs) >= 3: # if there is noun prediction
                    noun = outputs[2][j].detach().cpu().numpy().astype(np.float64)
                    for i, n in enumerate(noun):
                        json_outputs['results'][str(uid[j].item())]['noun'][str(i)] = n
                else:
                    for i in range(352):
                        json_outputs['results'][str(uid[j].item())]['noun'][str(i)] = round(float(0.0), ndigits=3)

                action_sort_ids = np.argsort(action)[::-1]
                for i, a_id in enumerate(action_sort_ids[:100]):
                    a = action[a_id]
                    class_key = all_action_ids[all_action_ids.action_id == a_id].class_key.item()
                    json_outputs['results'][str(uid[j].item())]['action'][class_key.replace('_', ',')] = round(a, ndigits=3)
            print('\r[Batch {}/{}]'.format(batch_idx, len(test_iterator)), end='')

    return json_outputs

