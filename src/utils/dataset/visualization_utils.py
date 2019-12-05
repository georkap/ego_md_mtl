import cv2
import numpy as np
from src.utils.dataset.file_loading_utils import load_pickle


def vis_with_circle(img, left_point, right_point, winname):
    k = cv2.circle(img.copy(), (int(left_point[0]), int(left_point[1])), 10, (255, 0, 0), 4)
    k = cv2.circle(k, (int(right_point[0]), int(right_point[1])), 10, (0, 0, 255), 4)
    cv2.imshow(winname, k)


def vis_with_circle_gaze(img, gaze_point, winname):
    k = cv2.circle(img.copy(), (int(gaze_point[0]), int(gaze_point[1])), 10, (0, 255, 0), 4)  # green is gaze
    cv2.imshow(winname, k)


def visualize_item(sampled_frames, clip_input, track_idxs, use_hands, hand_path, left_track_vis, right_track_vis,
                   use_gaze, gaze_path, gaze_track_vis, or_w, or_h, norm_val, use_flow, sampled_flow, clip_flow):
    if use_hands:
        hand_tracks = load_pickle(hand_path)
        orig_left = np.array(hand_tracks['left'], dtype=np.float32)
        orig_left = orig_left[track_idxs]
        orig_right = np.array(hand_tracks['right'], dtype=np.float32)
        orig_right = orig_right[track_idxs]
    if use_gaze:
        gaze_data = load_pickle(gaze_path)
        orig_gaze = np.array([[value[0], value[1]] for key, value in gaze_data.items()], dtype=np.float32)[track_idxs]

    for i in range(len(sampled_frames)):
        cv2.imshow('orig_img', cv2.cvtColor(sampled_frames[i], cv2.COLOR_RGB2BGR))
        trans_img = clip_input[:, i, :, :].numpy().transpose(1, 2, 0)
        trans_img -= trans_img.min()
        trans_img /= trans_img.max()
        cv2.imshow('transform', cv2.cvtColor(trans_img, cv2.COLOR_RGB2BGR))
        if use_hands:
            vis_with_circle(sampled_frames[i], orig_left[i], orig_right[i], 'hands no aug')
            vis_with_circle(clip_input[:, i, :, :].numpy().transpose(1, 2, 0), left_track_vis[i], right_track_vis[i], 'hands transformed')
            vis_with_circle(clip_input[:, i, :, :].numpy().transpose(1, 2, 0), orig_left[i], orig_right[i], 'hands trans. img not coords')
        if use_gaze:
            vis_with_circle_gaze(sampled_frames[i], orig_gaze[i]*[or_w, or_h], 'gaze no aug')
            vis_with_circle_gaze(clip_input[:, i, :, :].numpy().transpose(1, 2, 0), gaze_track_vis[i], 'gaze transformed')
            vis_with_circle_gaze(clip_input[:, i, :, :].numpy().transpose(1, 2, 0), orig_gaze[i]*norm_val, 'gaze trans. img not coords')
        if use_flow:
            cv2.imshow('u', sampled_flow[i][:, :, 0])
            cv2.imshow('u_t', clip_flow[0, i, :, :].numpy())
            cv2.imshow('v', sampled_flow[i][:, :, 1])
            cv2.imshow('v_t', clip_flow[1, i, :, :].numpy())
        cv2.waitKey(0)
