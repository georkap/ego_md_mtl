import torch
import dsntnn


def gaze_loss(targets, masks, targets_start_from, masks_start_from, coords, heatmaps, probabilities, slice_ind):
    gaze_targets = targets[targets_start_from:targets_start_from + 16, :].transpose(1, 0).reshape(-1, 8, 1, 2)
    gaze_masks = masks[:, masks_start_from:masks_start_from+8].squeeze()
    gaze_coords = coords[:, :, slice_ind, :]
    gaze_coords.unsqueeze_(2)
    gaze_heatmaps = heatmaps[:, :, slice_ind, :]
    gaze_heatmaps.unsqueeze_(2)
    gaze_coord_loss = calc_coord_loss(gaze_coords, gaze_heatmaps, gaze_targets, gaze_masks)
    return gaze_coord_loss


def hand_loss(targets, masks, targets_start_from, masks_start_from, coords, heatmaps, probabilities, slice_from):
    hand_targets = targets[targets_start_from:targets_start_from + 32, :].transpose(1, 0).reshape(-1, 8, 2, 2)
    hand_masks = masks[:, masks_start_from:masks_start_from+16].reshape(-1, 8, 2).squeeze()
    # for hands slice the last two elements, first is left, second is right hand
    hand_coords = coords[:, :, slice_from:slice_from + 2, :]
    hand_heatmaps = heatmaps[:, :, slice_from:slice_from + 2, :]
    hand_coord_loss = calc_coord_loss(hand_coords, hand_heatmaps, hand_targets, hand_masks)
    return hand_coord_loss


def calc_coord_loss(coords, heatmaps, target_var, masks):
    # Per-location euclidean losses
    euc_losses = dsntnn.euclidean_losses(coords, target_var)  # shape:[B, D, L, 2] batch, depth, locations, feature
    # Per-location regularization losses

    reg_losses = []
    for i in range(heatmaps.shape[1]):
        hms = heatmaps[:, i]
        target = target_var[:, i]
        reg_loss = dsntnn.js_reg_losses(hms, target, sigma_t=1.0)
        reg_losses.append(reg_loss)
    reg_losses = torch.stack(reg_losses, 1)
    # reg_losses = dsntnn.js_reg_losses(heatmaps, target_var, sigma_t=1.0) # shape: [B, D, L, 7, 7]
    # Combine losses into an overall loss
    coord_loss = dsntnn.average_loss((euc_losses + reg_losses).squeeze(), mask=masks)
    return coord_loss
