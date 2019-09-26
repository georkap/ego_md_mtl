# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:53:37 2018

Functions that are used in training the network.

1) Cyclic learning rate scheduler from:
    https://github.com/thomasjpfan/pytorch/blob/master/torch/optim/lr_scheduler.py


@author: Γιώργος
"""
import sys
import time
import torch

from torch.optim.lr_scheduler import StepLR, MultiStepLR
from src.utils.calc_utils import AverageMeter, accuracy
from src.utils.learning_rates import CyclicLR, GroupMultistep
from src.losses.mtl_losses import get_mtl_losses
from src.utils.eval_utils import *


def load_lr_scheduler(lr_type, lr_steps, optimizer, train_iterator_length):
    if lr_type == 'step':
        lr_scheduler = StepLR(optimizer, step_size=int(lr_steps[0]), gamma=float(lr_steps[1]))
    elif lr_type == 'multistep':
        lr_scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in lr_steps[:-1]], gamma=float(lr_steps[-1]))
    elif lr_type == 'clr':
        lr_scheduler = CyclicLR(optimizer, base_lr=float(lr_steps[0]),
                                max_lr=float(lr_steps[1]), step_size_up=int(lr_steps[2])*train_iterator_length,
                                step_size_down=int(lr_steps[3])*train_iterator_length, mode=str(lr_steps[4]),
                                gamma=float(lr_steps[5]))
    elif lr_type == 'groupmultistep':
        lr_scheduler = GroupMultistep(optimizer,
                                      milestones=[int(x) for x in lr_steps[:-1]],
                                      gamma=float(lr_steps[-1]))
    else:
        sys.exit("Unsupported lr type")
    return lr_scheduler


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


def train_mfnet_mo(model, optimizer, criterion, train_iterator, num_outputs, tasks_per_dataset, cur_epoch, log_file,
                   gpus, lr_scheduler=None):
    num_cls_outputs, num_g_outputs, num_h_outputs = num_outputs
    batch_time = AverageMeter()
    losses = AverageMeter()
    loss_meters = [AverageMeter() for _ in range(num_cls_outputs)]
    losses_hands = [AverageMeter() for _ in range(num_h_outputs)]
    losses_gaze = [AverageMeter() for _ in range(num_g_outputs)]
    top1_meters = [AverageMeter() for _ in range(num_cls_outputs)]
    top5_meters = [AverageMeter() for _ in range(num_cls_outputs)]

    model.train()

    if not isinstance(lr_scheduler, CyclicLR):
        lr_scheduler.step()

    print_and_save('*********', log_file)
    print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_iterator):
        if isinstance(lr_scheduler, CyclicLR):
            lr_scheduler.step()

        inputs = inputs.cuda(gpus[0])
        outputs, coords, heatmaps = model(inputs)
        # needs transpose to get the first dim to be the task and the second dim to be the batch
        targets = targets.cuda(gpus[0]).transpose(0, 1)

        loss, cls_losses, gaze_coord_losses, hand_coord_losses = get_mtl_losses(targets, outputs, coords, heatmaps,
                                                                                num_outputs, tasks_per_dataset,
                                                                                criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metrics
        batch_size = outputs[0].size(0)
        losses.update(loss.item(), batch_size)
        for ind in range(num_cls_outputs):
            t1, t5 = accuracy(outputs[ind].detach().cpu(), targets[ind].detach().cpu().long(), topk=(1, 5))
            top1_meters[ind].update(t1.item(), batch_size)
            top5_meters[ind].update(t5.item(), batch_size)
            loss_meters[ind].update(cls_losses[ind].item(), batch_size)

        for i, gl in enumerate(gaze_coord_losses):
            losses_gaze[i].update(gl.item(), batch_size)
        for i, hl in enumerate(hand_coord_losses):
            losses_hands[i].update(hl.item(), batch_size)

        batch_time.update(time.time() - t0)
        t0 = time.time()
        to_print = '[Epoch:{}, Batch {}/{} in {:.3f} s]'.format(cur_epoch, batch_idx, len(train_iterator),
                                                                batch_time.val)
        to_print += '[Losses {:.4f}[avg:{:.4f}], '.format(losses.val, losses.avg)
        for ind in range(num_g_outputs):
            to_print += '[l_gcoo_{} {:.4f}[avg:{:.4f}], '.format(ind, losses_gaze[ind].val, losses_gaze[ind].avg)
        for ind in range(num_h_outputs):
            to_print += '[l_hcoo_{} {:.4f}[avg:{:.4f}], '.format(ind, losses_hands[ind].val, losses_hands[ind].avg)
        for ind in range(num_cls_outputs):
            to_print += 'T{}::loss {:.4f}[avg:{:.4f}], '.format(ind, loss_meters[ind].val, loss_meters[ind].avg)
        for ind in range(num_cls_outputs):
            to_print += 'T{}::Top1 {:.3f}[avg:{:.3f}],Top5 {:.3f}[avg:{:.3f}],'.format(
                ind, top1_meters[ind].val, top1_meters[ind].avg, top5_meters[ind].val, top5_meters[ind].avg)
        to_print += 'LR {:.6f}'.format(lr_scheduler.get_lr()[0])
        print_and_save(to_print, log_file)
    print_and_save("Epoch train time: {}".format(batch_time.sum), log_file)


def test_mfnet_mo(model, criterion, test_iterator, num_outputs, tasks_per_dataset, cur_epoch, dataset, log_file, gpus):
    num_cls_outputs, num_g_outputs, num_h_outputs = num_outputs
    losses = AverageMeter()
    top1_meters = [AverageMeter() for _ in range(num_cls_outputs)]
    top5_meters = [AverageMeter() for _ in range(num_cls_outputs)]

    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, targets) in enumerate(test_iterator):
            inputs = inputs.cuda(gpus[0])
            outputs, coords, heatmaps = model(inputs)
            targets = targets.cuda(gpus[0]).transpose(0, 1)

            loss, cls_losses, gaze_coord_losses, hand_coord_losses = get_mtl_losses(targets, outputs, coords, heatmaps,
                                                                                    num_outputs, tasks_per_dataset,
                                                                                    criterion)

            # update metrics
            batch_size = outputs[0].size(0)
            losses.update(loss.item(), batch_size)
            for ind in range(num_cls_outputs):
                t1, t5 = accuracy(outputs[ind].detach().cpu(), targets[ind].detach().cpu().long(), topk=(1, 5))
                top1_meters[ind].update(t1.item(), batch_size)
                top5_meters[ind].update(t5.item(), batch_size)

            to_print = '[Epoch:{}, Batch {}/{}] '.format(cur_epoch, batch_idx, len(test_iterator))
            for ind in range(num_cls_outputs):
                to_print += 'T{}::Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}],'.format(
                    ind, top1_meters[ind].val, top1_meters[ind].avg, top5_meters[ind].val, top5_meters[ind].avg)

            print_and_save(to_print, log_file)

        final_print = '{} Results: Loss {:.3f},'.format(dataset, losses.avg)
        for ind in range(num_cls_outputs):
            final_print += 'T{}::Top1 {:.3f}, Top5 {:.3f},'.format(ind, top1_meters[ind].avg, top5_meters[ind].avg)
        print_and_save(final_print, log_file)
    return [tasktop1.avg for tasktop1 in top1_meters]


def validate_mfnet_mo_gaze(model, test_iterator, num_outputs, use_gaze, use_hands, cur_epoch, dataset, log_file):
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
                action = outputs[0][j].detach().cpu().numpy().astype(np.float64)
                verb = outputs[1][j].detach().cpu().numpy().astype(np.float64)
                noun = outputs[2][j].detach().cpu().numpy().astype(np.float64)

                json_outputs['results'][str(uid[j].item())] = {'verb': {}, 'noun': {}, 'action': {}}
                for i, v in enumerate(verb):
                    json_outputs['results'][str(uid[j].item())]['verb'][str(i)] = v
                for i, n in enumerate(noun):
                    json_outputs['results'][str(uid[j].item())]['noun'][str(i)] = n
                # for i in range(322, 352):
                #     json_outputs['results'][str(uid[j].item())]['noun'][str(i)]=0.0
                action_sort_ids = np.argsort(action)[::-1]
                for i, a_id in enumerate(action_sort_ids[:100]):
                    a = action[a_id]
                    class_key = all_action_ids[all_action_ids.action_id == a_id].class_key.item()
                    json_outputs['results'][str(uid[j].item())]['action'][class_key.replace('_', ',')] = a
            print('\r[Batch {}/{}]'.format(batch_idx, len(test_iterator)), end='')

    return json_outputs


def validate_mfnet_mo(model, criterion, test_iterator, num_outputs, tasks_per_dataset, cur_epoch, dataset, log_file):
    num_cls_outputs, num_g_outputs, num_h_outputs = num_outputs
    losses = AverageMeter()
    top1_meters = [AverageMeter() for _ in range(num_outputs)]
    top5_meters = [AverageMeter() for _ in range(num_outputs)]
    task_outputs = [[] for _ in range(num_outputs)]

    print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets, video_names) in enumerate(test_iterator):
            inputs = inputs.cuda()
            outputs, coords, heatmaps = model(inputs)
            targets = targets.cuda().transpose(0, 1)

            loss, cls_losses, gaze_coord_losses, hand_coord_losses = get_mtl_losses(targets, outputs, coords, heatmaps,
                                                                                    num_outputs, tasks_per_dataset,
                                                                                    criterion)

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
                t1, t5 = accuracy(outputs[ind].detach().cpu(), targets[ind].detach().cpu(), topk=(1, 5))
                top1_meters[ind].update(t1.item(), batch_size)
                top5_meters[ind].update(t5.item(), batch_size)

            to_print = '[Batch {}/{}]'.format(batch_idx, len(test_iterator))
            for ind in range(num_outputs):
                to_print += '[T{}::Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]],'.format(ind, top1_meters[ind].val,
                                                                                              top1_meters[ind].avg,
                                                                                              top5_meters[ind].val,
                                                                                              top5_meters[ind].avg)
            to_print += '\n\t{}'.format(batch_preds)
            print_and_save(to_print, log_file)

        to_print = '{} Results: Loss {:.3f}'.format(dataset, losses.avg)
        for ind in range(num_outputs):
            to_print += ', T{}::Top1 {:.3f}, Top5 {:.3f}'.format(ind, top1_meters[ind].avg, top5_meters[ind].avg)
        print_and_save(to_print, log_file)
    return [tasktop1.avg for tasktop1 in top1_meters], task_outputs
