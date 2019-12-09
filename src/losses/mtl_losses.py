import torch
import torch.nn.functional as F
from src.losses.coord_loss import gaze_loss, hand_loss
from src.losses.object_loss import object_loss
from src.losses.min_norm_solvers import MinNormSolver, gradient_normalizers


def get_mtl_losses(targets, masks, task_outputs, task_sizes, one_obj_layer, counts, is_training=False):
    outputs, coords, heatmaps, probabilities, objects, obj_cat = task_outputs
    num_cls_outputs, num_g_outputs, num_h_outputs, num_o_outputs, num_c_outputs = task_sizes
    targets_starting_point = num_cls_outputs
    masks_starting_point = 0
    slice_from = 0
    cls_targets = targets[:num_cls_outputs, :].long()
    cls_losses = []
    assert len(cls_targets) == num_cls_outputs
    for output, target in zip(outputs, cls_targets):
        loss_for_task = F.cross_entropy(output, target)
        cls_losses.append(loss_for_task)
    loss = sum(cls_losses)
    # finished with classification losses for any dataset

    gaze_coord_losses, hand_coord_losses = [], []
    if num_g_outputs > 0:
        gaze_coord_loss = gaze_loss(targets, masks, targets_start_from=targets_starting_point,
                                    masks_start_from=masks_starting_point, coords=coords, heatmaps=heatmaps,
                                    probabilities=probabilities, slice_ind=slice_from)
        targets_starting_point += 16
        masks_starting_point += 8
        slice_from += 1
        loss = loss + gaze_coord_loss
        gaze_coord_losses.append(gaze_coord_loss)
    if num_h_outputs > 0:
        hand_coord_loss = hand_loss(targets, masks, targets_start_from=targets_starting_point,
                                    masks_start_from=masks_starting_point, coords=coords, heatmaps=heatmaps,
                                    probabilities=probabilities, slice_from=slice_from)
        targets_starting_point += 32
        masks_starting_point += 16
        slice_from += 2
        loss = loss + hand_coord_loss
        hand_coord_losses.append(hand_coord_loss)
    object_losses, obj_cat_losses = [], []
    if num_o_outputs > 0:
        num_objects = counts[0]
        object_vector_loss = object_loss(targets, masks, objects, num_objects, targets_start_from=targets_starting_point,
                                         masks_start_from=masks_starting_point, single_object_layer=one_obj_layer,
                                         is_training=is_training)
        targets_starting_point += num_objects
        loss = loss + object_vector_loss
        object_losses.append(object_vector_loss)
    if num_c_outputs > 0:
        num_c_obj = counts[1]
        obj_cat_vector_loss = object_loss(targets, masks, obj_cat, num_c_obj, targets_start_from=targets_starting_point,
                                          masks_start_from=masks_starting_point, single_object_layer=one_obj_layer,
                                          is_training=is_training)
        targets_starting_point += num_c_obj
        loss = loss + obj_cat_vector_loss
        obj_cat_losses.append(obj_cat_vector_loss)
    if num_o_outputs > 0 or num_c_outputs > 0:
        masks_starting_point += 1

    partial_losses = cls_losses, gaze_coord_losses, hand_coord_losses, object_losses, obj_cat_losses
    return loss, partial_losses

def _get_mtl_losses(targets, dataset_ids, outputs, coords, heatmaps, num_outputs, tasks_per_dataset, criterion):
    num_cls_outputs, num_g_outputs, num_h_outputs = num_outputs
    targets_starting_point = num_cls_outputs
    num_datasets = len(tasks_per_dataset)
    slice_from = 0

    # the structure here will depend on how the labels are given from the dataset loader
    # (which is not yet done for multidataset).
    # The design will be: all classification labels first, and then it will be GH per dataset
    cls_targets = targets[:num_cls_outputs, :].long()
    cls_losses = []
    if num_datasets == 1:
        assert len(cls_targets) == num_cls_outputs
        for output, target in zip(outputs, cls_targets):
            loss_for_task = criterion(output, target)
            cls_losses.append(loss_for_task)
    else:
        # find which part of the batch belongs in which dataset
        batch_ids_per_dataset = []
        for dat in range(num_datasets):
            batch_ids = []
            batch_ids_per_dataset.append(batch_ids)
        for batch_ind, dataset_id in enumerate(dataset_ids):
            batch_ids_per_dataset[dataset_id].append(batch_ind)

        global_task_id = 0
        for dataset_id in range(num_datasets):
            # for task_id, output in enumerate(outputs):
            num_cls_tasks = tasks_per_dataset[dataset_id]['num_cls_tasks']
            for task_id in range(num_cls_tasks):
                t_output = outputs[global_task_id+task_id][batch_ids_per_dataset[dataset_id]]
                t_target = cls_targets[:, batch_ids_per_dataset[dataset_id]][task_id]
                if len(t_output) > 0:
                    loss_for_task = criterion(t_output, t_target)
                    cls_losses.append(loss_for_task)
            global_task_id += num_cls_tasks

    loss = sum(cls_losses)
    # finished with classification losses for any dataset

    gaze_coord_losses, hand_coord_losses = [], []
    for td in tasks_per_dataset:
        if 'G' in td:
            gaze_coord_loss = gaze_loss(targets, start_from=targets_starting_point, coords=coords, heatmaps=heatmaps,
                                        slice_ind=slice_from)
            targets_starting_point += 16
            slice_from += 1
            loss = loss + gaze_coord_loss
            gaze_coord_losses.append(gaze_coord_loss)
        if 'H' in td:
            hand_coord_loss = hand_loss(targets, start_from=targets_starting_point, coords=coords, heatmaps=heatmaps,
                                        slice_from=slice_from)
            targets_starting_point += 32
            slice_from += 2
            loss = loss + hand_coord_loss
            hand_coord_losses.append(hand_coord_loss)
    return loss, cls_losses, gaze_coord_losses, hand_coord_losses


def multiobjective_gradient_optimization(model, optimizer, inputs, targets, num_outputs, tasks_per_dataset,
                                         cls_criterion):
    num_cls_outputs, num_g_outputs, num_h_outputs = num_outputs
    targets_starting_point = num_cls_outputs
    slice_from = 0

    tasks = []
    grads = {}
    losses = {}
    scale = {}
    optimizer.zero_grad()
    with torch.no_grad():
        # compute activations of the shared network block, right before task branchess, without computing gradients
        # sh_block_output, sh_block_tail = model(inputs, upto='shared')
        sh_block_output, sh_block_tail = model.module.forward_shared_block(inputs)
    sh_block_output_variable = torch.tensor(sh_block_output.data.clone(), requires_grad=True)
    sh_block_tail_variable = torch.tensor(sh_block_tail.data.clone(), requires_grad=True)

    cls_targets = targets[:num_cls_outputs, :].long()
    assert len(cls_targets) == num_cls_outputs

    # compute gradients for the classification tasks
    for i, (cls_task, cls_target) in enumerate(zip(model.module.classifier_list.classifier_list, cls_targets)):
        optimizer.zero_grad()
        cls_task_output = cls_task(sh_block_output_variable)
        task_loss = cls_criterion(cls_task_output, cls_target)
        losses[i] = task_loss.item()
        task_loss.backward()
        grads[i] = [torch.tensor(sh_block_output_variable.grad.clone(), requires_grad=False)]
        sh_block_output_variable.grad.zero_()
        tasks.append(i)

    # compute gradients for the coordinate regression tasks
    # start with one dataset tasks for simplicity for now
    coords, heatmaps = model.module.forward_coord_layers(sh_block_tail_variable)
    if num_g_outputs > 0:
        optimizer.zero_grad()
        coord_loss = gaze_loss(targets, start_from=targets_starting_point, coords=coords, heatmaps=heatmaps,
                               slice_ind=slice_from)
        losses['G'] = coord_loss.data.item()
        coord_loss.backward()
        grads_g = torch.tensor(sh_block_tail_variable.grad.clone(), requires_grad=False)
        grads['G'] = [F.avg_pool3d(grads_g, kernel_size=(8, 7, 7), stride=(1, 1, 1)).squeeze_()]
        sh_block_output_variable.grad.data.zero_()
        tasks.append('G')
        targets_starting_point += 16
        slice_from += 1
    if num_h_outputs > 0:
        optimizer.zero_grad()
        coord_loss = hand_loss(targets, start_from=targets_starting_point, coords=coords, heatmaps=heatmaps,
                               slice_from=slice_from)
        losses['H'] = coord_loss.data.item()
        coord_loss.backward()
        grads_h = torch.tensor(sh_block_tail_variable.grad.clone(), requires_grad=False)
        grads['H'] = [F.avg_pool3d(grads_h, kernel_size=(8, 7, 7), stride=(1, 1, 1)).squeeze_()]
        sh_block_output_variable.grad.data.zero_()
        tasks.append('H')
        targets_starting_point += 32
        slice_from += 2

    # Normalize all gradients, this is optional and not included in the paper. See the notebook for details
    gn = gradient_normalizers(grads, losses, 'loss+')
    for t in tasks:
        for gr_i in range(len(grads[t])):
            grads[t][gr_i] = grads[t][gr_i] / gn[t]

    # Frank-Wolfe iteration to compute scales.
    sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
    for i, t in enumerate(tasks):
        scale[t] = float(sol[i])

    # Scaled back-propagation
    optimizer.zero_grad()
    outputs, coords, heatmaps = model(inputs)
    cls_losses = []
    for i, (output, target) in enumerate(zip(outputs, cls_targets)):
        loss_for_task = cls_criterion(output, target)
        cls_losses.append(loss_for_task * scale[i])
    loss = sum(cls_losses)

    targets_starting_point = num_cls_outputs
    slice_from = 0
    gaze_coord_losses, hand_coord_losses = [], []
    for td in tasks_per_dataset:
        if 'G' in td:
            gaze_coord_loss = gaze_loss(targets, start_from=targets_starting_point, coords=coords, heatmaps=heatmaps,
                                        slice_ind=slice_from)
            targets_starting_point += 16
            slice_from += 1
            loss = loss + gaze_coord_loss * scale['G']
            gaze_coord_losses.append(gaze_coord_loss)
        if 'H' in td:
            hand_coord_loss = hand_loss(targets, start_from=targets_starting_point, coords=coords, heatmaps=heatmaps,
                                        slice_from=slice_from)
            targets_starting_point += 32
            slice_from += 2
            loss = loss + hand_coord_loss * scale['H']
            hand_coord_losses.append(hand_coord_loss)

    return loss, cls_losses, gaze_coord_losses, hand_coord_losses, outputs
