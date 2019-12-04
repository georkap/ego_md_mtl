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

from src.utils.calc_utils import AverageMeter, accuracy
from src.utils.learning_rates import CyclicLR
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


def train_mfnet_mo(model, optimizer, criterion, train_iterator, tasks_per_dataset, cur_epoch, log_file, gpus,
                   lr_scheduler=None, moo=False, use_flow=False):
    batch_time = AverageMeter()
    full_losses = AverageMeter()
    dataset_metrics = list()
    for i, dataset in enumerate(tasks_per_dataset):
        dataset_metrics.append(dict())
        num_cls_tasks = dataset['num_cls_tasks']
        num_g_tasks = dataset['num_g_tasks']
        num_h_tasks = dataset['num_h_tasks']
        num_o_tasks = dataset['num_o_tasks']
        losses = AverageMeter()
        cls_loss_meters = [AverageMeter() for _ in range(num_cls_tasks)]
        losses_hands = [AverageMeter() for _ in range(num_h_tasks)]
        losses_gaze = [AverageMeter() for _ in range(num_g_tasks)]
        losses_objects = [AverageMeter() for _ in range(num_o_tasks)]
        top1_meters = [AverageMeter() for _ in range(num_cls_tasks)]
        top5_meters = [AverageMeter() for _ in range(num_cls_tasks)]
        dataset_metrics[i]['losses'] = losses
        dataset_metrics[i]['cls_loss_meters'] = cls_loss_meters
        dataset_metrics[i]['losses_hands'] = losses_hands
        dataset_metrics[i]['losses_gaze'] = losses_gaze
        dataset_metrics[i]['losses_objects'] = losses_objects
        dataset_metrics[i]['top1_meters'] = top1_meters
        dataset_metrics[i]['top5_meters'] = top5_meters

    model.train()

    if not isinstance(lr_scheduler, CyclicLR):
        lr_scheduler.step()

    print_and_save('*********', log_file)
    print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()
    for batch_idx, data in enumerate(train_iterator):
        if isinstance(lr_scheduler, CyclicLR):
            lr_scheduler.step()

        if use_flow:
            rgb, flow, targets, masks, dataset_ids = data
            rgb = rgb.cuda(gpus[0])
            flow = flow.cuda(gpus[0])
            inputs = [rgb, flow]
        else:
            inputs, targets, masks, dataset_ids = data
            inputs = inputs.cuda(gpus[0])
        optimizer.zero_grad()
        batch_size = dataset_ids.shape[0]

        # identify the dataset that the samples of the batch belong to
        num_datasets = len(tasks_per_dataset)
        batch_ids_per_dataset = []
        for dat in range(num_datasets):
            batch_ids = []
            batch_ids_per_dataset.append(batch_ids)
        for batch_ind, dataset_id in enumerate(dataset_ids):
            batch_ids_per_dataset[dataset_id].append(batch_ind)

        # For single dataset only
        # if moo:
        #     loss, cls_losses, gaze_coord_losses, hand_coord_losses, outputs = \
        #         multiobjective_gradient_optimization(model, optimizer, inputs, targets, num_outputs, tasks_per_dataset,
        #                                              criterion)
        # else:
        #     outputs, coords, heatmaps = model(inputs)
        #     loss, cls_losses, gaze_coord_losses, hand_coord_losses = get_mtl_losses(targets, outputs,
        #                                                                             coords, heatmaps,
        #                                                                             num_outputs, tasks_per_dataset,
        #                                                                             criterion)

        outputs, coords, heatmaps, probabilities, objects = model(inputs)

        outputs_per_dataset = []
        targets_per_dataset = []
        masks_per_dataset = []
        objects_per_dataset = []
        global_task_id = 0 # assign indices to keep track of the tasks because the model outputs come in a sequence
        global_coord_id = 0
        global_object_id = 0
        full_loss = []
# calculate losses for the tasks of each dataset individually
        for dataset_id in range(num_datasets):
            num_cls_tasks = tasks_per_dataset[dataset_id]['num_cls_tasks']
            num_g_tasks = tasks_per_dataset[dataset_id]['num_g_tasks']
            num_h_tasks = tasks_per_dataset[dataset_id]['num_h_tasks']
            num_o_tasks = tasks_per_dataset[dataset_id]['num_o_tasks']
            num_coord_tasks = num_g_tasks + 2 * num_h_tasks
# needs transpose to get the first dim to be the task and the second dim to be the batch
            tmp_targets = targets[batch_ids_per_dataset[dataset_id]].cuda(gpus[0]).transpose(0, 1)
            tmp_masks = masks[batch_ids_per_dataset[dataset_id]].cuda(gpus[0]) # the masks are for the coordinate tasks
            targets_per_dataset.append(tmp_targets)
            masks_per_dataset.append(tmp_masks)
            outputs_per_dataset.append([])
            objects_per_dataset.append([])
# when a dataset does not have any representative samples in a batch
            if not len(tmp_targets[0]) > 0:
                global_task_id += num_cls_tasks
                global_coord_id += num_coord_tasks
                global_object_id += num_o_tasks
                continue
# get model's outputs for the classification tasks for the current dataset
            for task_id in range(num_cls_tasks):
                tmp_outputs = outputs[global_task_id+task_id][batch_ids_per_dataset[dataset_id]]
                outputs_per_dataset[dataset_id].append(tmp_outputs)
# get model's outputs for the coord tasks of the current dataset
            coo, hea, pro = None, None, None
            if coords is not None:
                coo = coords[batch_ids_per_dataset[dataset_id], :, global_coord_id:global_coord_id+num_coord_tasks, :]
                hea = heatmaps[batch_ids_per_dataset[dataset_id], :, global_coord_id:global_coord_id+num_coord_tasks, :]
                # pro = probabilities[batch_ids_per_dataset[dataset_ids], :]
# get model's outputs for the object classification tasks of the current dataset
            if objects is not None:
                for no in range(len(objects[global_object_id])):
                    objects_per_dataset[dataset_id].append(objects[global_object_id][no][batch_ids_per_dataset[dataset_id], :])
            loss, cls_losses, gaze_coord_losses, hand_coord_losses, object_losses = get_mtl_losses(
                targets_per_dataset[dataset_id], masks_per_dataset[dataset_id], outputs_per_dataset[dataset_id],
                coo, hea, pro, objects_per_dataset[dataset_id], (num_cls_tasks, num_g_tasks, num_h_tasks, num_o_tasks),
                criterion)
            global_task_id += num_cls_tasks
            global_coord_id += num_coord_tasks
            global_object_id += num_o_tasks
            full_loss.append(loss)

            # loss.backward()

            # update metrics
            dataset_batch_size = len(batch_ids_per_dataset[dataset_id])
            dataset_metrics[dataset_id]['losses'].update(loss.item(), dataset_batch_size)
            for ind in range(num_cls_tasks):
                t1, t5 = accuracy(outputs_per_dataset[dataset_id][ind].detach().cpu(),
                                  targets_per_dataset[dataset_id][ind].detach().cpu().long(), topk=(1, 5))
                dataset_metrics[dataset_id]['top1_meters'][ind].update(t1.item(), dataset_batch_size)
                dataset_metrics[dataset_id]['top5_meters'][ind].update(t5.item(), dataset_batch_size)
                dataset_metrics[dataset_id]['cls_loss_meters'][ind].update(cls_losses[ind].item(), dataset_batch_size)
            for i, gl in enumerate(gaze_coord_losses):
                dataset_metrics[dataset_id]['losses_gaze'][i].update(gl.item(), dataset_batch_size)
            for i, hl in enumerate(hand_coord_losses):
                dataset_metrics[dataset_id]['losses_hands'][i].update(hl.item(), dataset_batch_size)
            for i, ol in enumerate(object_losses):
                dataset_metrics[dataset_id]['losses_objects'][i].update(ol.item(), dataset_batch_size)

        # compute gradients and backward
        full_loss = sum(full_loss)
        full_loss.backward()

        # print results
        batch_time.update(time.time() - t0)
        t0 = time.time()
        to_print = '[Epoch:{}, Batch {}/{} in {:.3f} s, LR {:.6f}]'.format(cur_epoch, batch_idx, len(train_iterator),
                                                                           batch_time.val, lr_scheduler.get_lr()[0])
        if num_datasets > 1:
            full_losses.update(full_loss.item(), batch_size)
            to_print += '[F_Loss {:.4f}[avg:{:.4f}]\n\t'.format(full_losses.val, full_losses.avg)
        for dataset_id in range(num_datasets):
            num_cls_tasks = tasks_per_dataset[dataset_id]['num_cls_tasks']
            num_g_tasks = tasks_per_dataset[dataset_id]['num_g_tasks']
            num_h_tasks = tasks_per_dataset[dataset_id]['num_h_tasks']
            num_o_tasks = tasks_per_dataset[dataset_id]['num_o_tasks']

            losses = dataset_metrics[dataset_id]['losses']
            losses_gaze = dataset_metrics[dataset_id]['losses_gaze']
            losses_hands = dataset_metrics[dataset_id]['losses_hands']
            losses_objects = dataset_metrics[dataset_id]['losses_objects']
            cls_loss_meters = dataset_metrics[dataset_id]['cls_loss_meters']
            top1_meters = dataset_metrics[dataset_id]['top1_meters']
            top5_meters = dataset_metrics[dataset_id]['top5_meters']
            to_print += '[Losses {:.4f}[avg:{:.4f}], '.format(losses.val, losses.avg)
            for ind in range(num_cls_tasks):
                to_print += 'T{}::loss {:.4f}[avg:{:.4f}], '.format(ind, cls_loss_meters[ind].val, cls_loss_meters[ind].avg)
            for ind in range(num_h_tasks):
                to_print += '[l_hcoo_{} {:.4f}[avg:{:.4f}], '.format(ind, losses_hands[ind].val, losses_hands[ind].avg)
            for ind in range(num_g_tasks):
                to_print += '[l_gcoo_{} {:.4f}[avg:{:.4f}], '.format(ind, losses_gaze[ind].val, losses_gaze[ind].avg)
            for ind in range(num_o_tasks):
                to_print += '[l_obj_{} {:.4f}[avg:{:.4f}], '.format(ind, losses_objects[ind].val, losses_objects[ind].avg)
            for ind in range(num_cls_tasks):
                if ind == 0:
                    to_print += '\n\t\t'
                to_print += 'T{}::Top1 {:.3f}[avg:{:.3f}],Top5 {:.3f}[avg:{:.3f}],'.format(
                    ind, top1_meters[ind].val, top1_meters[ind].avg, top5_meters[ind].val, top5_meters[ind].avg)
            if dataset_id+1 < num_datasets:
                to_print += "\n\t"
        print_and_save(to_print, log_file)

        optimizer.step()

    print_and_save("Epoch train time: {}".format(batch_time.sum), log_file)

def test_mfnet_mo(model, criterion, test_iterator, tasks_per_dataset, cur_epoch, dataset, log_file, gpus, use_flow=False):
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

    num_datasets = len(tasks_per_dataset)
    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, data in enumerate(test_iterator):
            if use_flow:
                rgb, flow, targets, masks, dataset_ids = data
                rgb = rgb.cuda(gpus[0])
                flow = flow.cuda(gpus[0])
                inputs = [rgb, flow]
            else:
                inputs, targets, masks, dataset_ids = data
                inputs = inputs.cuda(gpus[0])
            batch_size = dataset_ids.shape[0]

            batch_ids_per_dataset = []
            for dat in range(num_datasets):
                batch_ids = []
                batch_ids_per_dataset.append(batch_ids)
            for batch_ind, dataset_id in enumerate(dataset_ids):
                batch_ids_per_dataset[dataset_id].append(batch_ind)

            outputs, coords, heatmaps, probabilities, objects = model(inputs)

            outputs_per_dataset = []
            targets_per_dataset = []
            masks_per_dataset = []
            objects_per_dataset = []
            global_task_id = 0
            global_coord_id = 0
            global_object_id = 0
            for dataset_id in range(num_datasets):
                num_cls_tasks = tasks_per_dataset[dataset_id]['num_cls_tasks']
                num_g_tasks = tasks_per_dataset[dataset_id]['num_g_tasks']
                num_h_tasks = tasks_per_dataset[dataset_id]['num_h_tasks']
                num_o_tasks = tasks_per_dataset[dataset_id]['num_o_tasks']
                num_coord_tasks = num_g_tasks + 2 * num_h_tasks
                tmp_targets = targets[batch_ids_per_dataset[dataset_id]].cuda(gpus[0]).transpose(0, 1)
                tmp_masks = masks[batch_ids_per_dataset[dataset_id]].cuda(gpus[0])
                targets_per_dataset.append(tmp_targets)
                masks_per_dataset.append(tmp_masks)
                outputs_per_dataset.append([])
                objects_per_dataset.append([])
                if not len(tmp_targets[0]) > 0:
                    global_task_id += num_cls_tasks
                    global_coord_id += num_coord_tasks
                    global_object_id += num_o_tasks
                    continue
                for task_id in range(num_cls_tasks):
                    tmp_outputs = outputs[global_task_id + task_id][batch_ids_per_dataset[dataset_id]]
                    outputs_per_dataset[dataset_id].append(tmp_outputs)
                coo, hea, pro = None, None, None
                if coords is not None:
                    coo = coords[batch_ids_per_dataset[dataset_id], :, global_coord_id:global_coord_id + num_coord_tasks, :]
                    hea = heatmaps[batch_ids_per_dataset[dataset_id], :, global_coord_id:global_coord_id + num_coord_tasks, :]
                    # pro = probabilities[batch_ids_per_dataset[dataset_ids], :]
                if objects is not None:
                    for no in range(len(objects[global_object_id])):
                        objects_per_dataset[dataset_id].append(
                            objects[global_object_id][no][batch_ids_per_dataset[dataset_id], :])
                loss, cls_losses, gaze_coord_losses, hand_coord_losses, object_losses = get_mtl_losses(
                    targets_per_dataset[dataset_id], masks_per_dataset[dataset_id], outputs_per_dataset[dataset_id],
                    coo, hea, pro, objects_per_dataset[dataset_id], (num_cls_tasks, num_g_tasks, num_h_tasks, num_o_tasks), criterion)

                global_task_id += num_cls_tasks
                global_coord_id += num_coord_tasks
                global_object_id += num_o_tasks

                # update metrics
                dataset_batch_size = len(batch_ids_per_dataset[dataset_id])
                dataset_metrics[dataset_id]['losses'].update(loss.item(), dataset_batch_size)
                for ind in range(num_cls_tasks):
                    t1, t5 = accuracy(outputs_per_dataset[dataset_id][ind].detach().cpu(),
                                      targets_per_dataset[dataset_id][ind].detach().cpu().long(), topk=(1, 5))
                    dataset_metrics[dataset_id]['top1_meters'][ind].update(t1.item(), dataset_batch_size)
                    dataset_metrics[dataset_id]['top5_meters'][ind].update(t5.item(), dataset_batch_size)

            # print results
            to_print = '[Epoch:{}, Batch {}/{}]'.format(cur_epoch, batch_idx, len(test_iterator))
            for dataset_id in range(num_datasets):
                to_print += '\n\t'
                num_cls_tasks = tasks_per_dataset[dataset_id]['num_cls_tasks']
                top1_meters = dataset_metrics[dataset_id]['top1_meters']
                top5_meters = dataset_metrics[dataset_id]['top5_meters']
                for ind in range(num_cls_tasks):
                    to_print += ' T{}::Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}],'.format(
                        ind, top1_meters[ind].val, top1_meters[ind].avg, top5_meters[ind].val, top5_meters[ind].avg)

            print_and_save(to_print, log_file)

        for dataset_id in range(num_datasets):
            num_cls_tasks = tasks_per_dataset[dataset_id]['num_cls_tasks']
            losses = dataset_metrics[dataset_id]['losses']
            top1_meters = dataset_metrics[dataset_id]['top1_meters']
            top5_meters = dataset_metrics[dataset_id]['top5_meters']
            final_print = '{} Results: Loss {:.3f},'.format(dataset, losses.avg)
            for ind in range(num_cls_tasks):
                final_print += 'T{}::Top1 {:.3f}, Top5 {:.3f}, '.format(ind, top1_meters[ind].avg, top5_meters[ind].avg)
            print_and_save(final_print, log_file)

        task_top1s = list()
        for dataset_id in range(num_datasets):
            top1_meters = dataset_metrics[dataset_id]['top1_meters']
            for tasktop1 in top1_meters:
                task_top1s.append(tasktop1.avg)
    return task_top1s

def validate_mfnet_mo_gaze(model, test_iterator, num_outputs, use_gaze, use_hands, cur_epoch, dataset, log_file):
    # TODO: add updated code for flow
    auc_frame, auc_temporal = AverageMeter(), AverageMeter()
    aae_frame, aae_temporal = AverageMeter(), AverageMeter()
    print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)

    with torch.no_grad():
        model.eval()
        frame_counter = 0
        actual_frame_counter = 0
        video_counter = 0
        for batch_idx, (inputs, targets, orig_gaze, video_names) in enumerate(test_iterator):
            video_counter += 1
            to_print = '[Batch {}/{}]'.format(batch_idx, len(test_iterator))

            inputs = inputs.cuda()
            targets = targets.cuda().transpose(0, 1)
            orig_gaze = orig_gaze.cuda().transpose(0, 1)

            double_temporal_size = inputs.shape[2]
            temporal_size = double_temporal_size // 2

            if use_gaze or use_hands:
                cls_targets = targets[:num_outputs, :].long()
            else:
                cls_targets = targets
            assert len(cls_targets) == num_outputs

            gaze_targets = targets[num_outputs:num_outputs + 2*temporal_size, :].transpose(1, 0).reshape(-1, temporal_size, 1, 2)
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

def validate_mfnet_mo(model, criterion, test_iterator, num_outputs, cur_epoch, dataset, log_file, use_flow=False):
    num_cls_outputs, num_g_outputs, num_h_outputs, num_o_outputs = num_outputs
    losses = AverageMeter()
    top1_meters = [AverageMeter() for _ in range(num_cls_outputs)]
    top5_meters = [AverageMeter() for _ in range(num_cls_outputs)]
    task_outputs = [[] for _ in range(num_cls_outputs)]

    print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
    with torch.no_grad():
        model.eval()
        for batch_idx, data in enumerate(test_iterator):

            if use_flow:
                rgb, flow, targets, masks, video_names = data
                rgb = rgb.cuda()
                flow = flow.cuda()
                inputs = [rgb, flow]
            else:
                inputs, targets, masks, video_names = data
                inputs = inputs.cuda()

            outputs, coords, heatmaps, probabilities, objects = model(inputs)
            targets = targets.cuda().transpose(0, 1)
            masks = masks.cuda()

            loss, cls_losses, gaze_coord_losses, hand_coord_losses, object_losses = get_mtl_losses(
                targets, masks, outputs, coords, heatmaps, probabilities, objects, num_outputs, criterion)

            batch_size = outputs[0].size(0)

            batch_preds = []
            for j in range(batch_size):
                txt_batch_preds = "{}".format(video_names[j])
                for ind in range(num_cls_outputs):
                    txt_batch_preds += ", "
                    res = np.argmax(outputs[ind][j].detach().cpu().numpy())
                    label = targets[ind][j].detach().cpu().numpy()
                    task_outputs[ind].append([res, label])
                    txt_batch_preds += "T{} P-L:{}-{}".format(ind, res, label)
                batch_preds.append(txt_batch_preds)

            losses.update(loss.item(), batch_size)
            for ind in range(num_cls_outputs):
                t1, t5 = accuracy(outputs[ind].detach().cpu(), targets[ind].detach().cpu().long(), topk=(1, 5))
                top1_meters[ind].update(t1.item(), batch_size)
                top5_meters[ind].update(t5.item(), batch_size)

            to_print = '[Batch {}/{}]'.format(batch_idx, len(test_iterator))
            for ind in range(num_cls_outputs):
                to_print += '[T{}::Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]],'.format(ind, top1_meters[ind].val,
                                                                                              top1_meters[ind].avg,
                                                                                              top5_meters[ind].val,
                                                                                              top5_meters[ind].avg)
            to_print += '\n\t{}'.format(batch_preds)
            print_and_save(to_print, log_file)

        to_print = '{} Results: Loss {:.3f}'.format(dataset, losses.avg)
        for ind in range(num_cls_outputs):
            to_print += ', T{}::Top1 {:.3f}, Top5 {:.3f}'.format(ind, top1_meters[ind].avg, top5_meters[ind].avg)
        print_and_save(to_print, log_file)
    return [tasktop1.avg for tasktop1 in top1_meters], task_outputs
