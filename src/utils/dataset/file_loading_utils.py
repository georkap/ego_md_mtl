import os
import pickle
import cv2
import numpy as np


def load_rgb_clip(path, sampled_idxs, dataset_info):
    sampled_rgb_frames = load_images(path, sampled_idxs, dataset_info.img_tmpl)
    clip_input = np.concatenate(sampled_rgb_frames, axis=2)
    return clip_input, sampled_rgb_frames

# from PIL import Image
def load_images(data_path, frame_indices, image_tmpl):
    images = []
    # images = np.zeros((len(frame_indices), 640, 480, 3))
    for f_ind in frame_indices:
        im_name = os.path.join(data_path, image_tmpl.format(f_ind))
        # next_image = np.array(Image.open(im_name).convert('RGB'))
        next_image = cv2.imread(im_name, cv2.IMREAD_COLOR)
        next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)
        images.append(next_image)
        # images[i] = next_image
    return images

def load_images2(data_path, frame_indices, image_tmpl, vis_data):
    image = cv2.imread(os.path.join(data_path, image_tmpl.format(frame_indices[0])))
    h, w, c = image.shape
    images = np.zeros((c * len(frame_indices), h, w), dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images[:c, :, :] = image.transpose((2, 0, 1))
    for i, f_ind in enumerate(frame_indices[1:]):
        im_name = os.path.join(data_path, image_tmpl.format(f_ind))
        image = cv2.imread(im_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images[(i+1)*c:(i+2)*c, :, :] = image.transpose((2, 0, 1))

    images = images.transpose((1, 2, 0))
    if vis_data:
        return images, np.split(images, len(frame_indices), axis=2)
    else:
        return images, None

def load_flow_clip(path, sampled_idxs, dataset_info, only_flow):
    sub_with_flow = dataset_info.sub_with_flow
    flow_path = path.replace(sub_with_flow, 'flow\\')
    # if only flow input do not subsample from 16 to 8 flow image pairs
    if only_flow:
        flow_idxs = (np.array(sampled_idxs) // 2)
    else:
        flow_idxs = (np.array(sampled_idxs) // 2)[::2]
    # for each flow channel load half the images of the rgb channel
    sampled_flow_frames = load_flow_images(flow_path, flow_idxs, dataset_info.img_tmpl)
    clip_flow = np.concatenate(sampled_flow_frames, axis=2)
    return clip_flow, sampled_flow_frames

def load_flow_images(data_path, frame_indices, image_tmpl):
    flow = [] # the will go uvuv...
    for f_ind in frame_indices:
        f_ind = f_ind + 1 if f_ind == 0 else f_ind
        u_name = os.path.join(data_path, 'u', image_tmpl.format(f_ind))
        v_name = os.path.join(data_path, 'v', image_tmpl.format(f_ind))
        _u = cv2.imread(u_name, cv2.IMREAD_GRAYSCALE)
        _v = cv2.imread(v_name, cv2.IMREAD_GRAYSCALE)
        flow.append(np.concatenate((_u[:,:,np.newaxis], _v[:,:,np.newaxis]), axis=2))
    return flow

def load_hand_tracks(hand_path, track_idxs, use_hands):
    if not use_hands:
        return None, None
    hand_tracks = load_pickle(hand_path)
    left_track = np.array(hand_tracks['left'], dtype=np.float32)[track_idxs]
    right_track = np.array(hand_tracks['right'], dtype=np.float32)[track_idxs]
    return left_track, right_track

def load_gaze_track(gaze_path, track_idxs, norm_val, eval_gaze, use_gaze):
    if not use_gaze:
        return None, None
    gaze_data = load_pickle(gaze_path)
    gaze_track = np.array([[value[0], value[1]] for key, value in gaze_data.items()], dtype=np.float32)
    if not eval_gaze: # if 'DoubleFullSampling' not in self.sampler.__repr__():
        gaze_track = gaze_track[track_idxs]
        gaze_track = gaze_track[::2]
    gaze_mask = gaze_track != [0, 0]
    gaze_mask = gaze_mask[:, 0] & gaze_mask[:, 1]
    gaze_track *= norm_val # take from [0,1] to [0, or_w] and [0, or_h] to apply transformations later
    return gaze_track, gaze_mask


def load_pickle(tracks_path):
    with open(tracks_path, 'rb') as f:
        tracks = pickle.load(f)
    return tracks

def substitute_prefix(tracks_path, secondary_prefix):
    obj_path = secondary_prefix
    for p in tracks_path.split('\\')[1:]:
        obj_path = os.path.join(obj_path, p)
    return obj_path

def load_two_pickle(tracks_path, secondary_prefix):
    obj_path = substitute_prefix(tracks_path, secondary_prefix)
    return load_pickle(tracks_path), load_pickle(obj_path)



