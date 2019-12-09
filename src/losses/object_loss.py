from torch.functional import F

def object_loss(targets, objects, num_objects, targets_start_from, single_object_layer=False):
    object_targets = targets[targets_start_from:targets_start_from + num_objects, :] #.reshape(-1, num_objects)

    if single_object_layer:
        return single_object_loss(objects, object_targets.transpose(1, 0))
    else:
        return multiple_object_loss(objects, object_targets)


def single_object_loss(objects, object_targets):
    loss = F.binary_cross_entropy_with_logits(objects, object_targets.type(objects.dtype), reduction='sum')
    return loss

def multiple_object_loss(objects, object_targets):
    obj_losses = []
    for obj, tar in zip(objects, object_targets):
        obj_losses.append(F.binary_cross_entropy_with_logits(obj, tar.unsqueeze(1).type(obj.dtype)))
    return sum(obj_losses)
