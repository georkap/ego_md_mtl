from src.losses.coord_loss import gaze_loss, hand_loss


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

