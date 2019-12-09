from torch.functional import F

def object_loss(targets, masks, objects, num_objects, targets_start_from, masks_start_from, single_object_layer=False,
                is_training=False):
    object_targets = targets[targets_start_from:targets_start_from + num_objects, :]
    object_mask = masks[:, masks_start_from:masks_start_from + 1]

    criterion = F.binary_cross_entropy_with_logits if is_training else F.binary_cross_entropy
    if single_object_layer:
        return single_object_loss(objects, object_targets.transpose(1, 0), object_mask, criterion)
    else:
        return multiple_object_loss(objects, object_targets, object_mask, criterion)


def single_object_loss(objects, object_targets, object_mask, criterion):
    loss = criterion(objects, object_targets.type(objects.dtype), weight=object_mask, reduction='sum')
    return loss

def multiple_object_loss(objects, object_targets, object_mask, criterion):
    obj_losses = []
    for obj, tar in zip(objects, object_targets): # for every 1-unit output layer
        obj_losses.append(criterion(obj, tar.unsqueeze(1).type(obj.dtype), weight=object_mask))
    return sum(obj_losses)
