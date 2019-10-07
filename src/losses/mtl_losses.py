import torch
import torch.nn.functional as F
from src.losses.coord_loss import gaze_loss, hand_loss
from src.losses.min_norm_solvers import MinNormSolver, gradient_normalizers


def get_mtl_losses(targets, outputs, coords, heatmaps, num_outputs, tasks_per_dataset, criterion):
    num_cls_outputs, num_g_outputs, num_h_outputs = num_outputs
    targets_starting_point = num_cls_outputs
    slice_from = 0

    # the structure here will depend on how the labels are given from the dataset loader
    # (which is not yet done for multidataset).
    # The design will be: all classification labels first, and then it will be GH per dataset
    cls_targets = targets[:num_cls_outputs, :].long()
    assert len(cls_targets) == num_cls_outputs

    cls_losses = []
    for output, target in zip(outputs, cls_targets):
        loss_for_task = criterion(output, target)
        cls_losses.append(loss_for_task)
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
