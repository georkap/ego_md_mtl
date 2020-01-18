# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 20:55:53 2018

Image dataset loader for a .txt file with a sample per line in the format
'path of image start_frame verb_id noun_id'

@author: Γιώργος
"""

import sys
import numpy as np
import torch

from src.utils.dataset.DatasetInfo import DatasetInfo
from src.utils.dataset.file_loading_utils import load_rgb_clip, load_flow_clip, load_hand_tracks, load_gaze_track, \
    load_pickle, load_images2, load_images3
from src.utils.dataset.dataset_loader_transforms import get_scale_and_shift
from src.utils.dataset.dataset_lines import *
from src.utils.dataset.visualization_utils import visualize_item
from src.constants import *


def read_samples_list(list_file, datatype):
    return [datatype(row) for row in open(list_file)]

def read_samples_list_vid_level(list_file, datatype, video_splits):
    video_list, sampled_centers = [], []
    for row in open(list_file):
        data_line = datatype(row)
        centers_for_video = np.linspace(data_line.start_frame + 16, data_line.start_frame + data_line.num_frames - 16,
                                        video_splits).astype(np.int)
        for center in centers_for_video:
            video_list.append(data_line)
            sampled_centers.append(center)
    return video_list, sampled_centers

def apply_transform_to_track(track, scale_x, scale_y, tl_x, tl_y, norm_val, is_flipped):
    track *= [scale_x, scale_y]
    track -= [tl_x, tl_y]
    if is_flipped:
        track[:, 0] = norm_val[0] - track[:, 0]  # norm_val[0] = img width after transform
    return track


def make_dsnt_track(track, norm_val):
    track = (track * 2 + 1) / norm_val - 1
    return track


def mask_coord_bounds(track):
    mask_lower = track >= [-1, -1]
    mask_lower = mask_lower[:, 0] & mask_lower[:, 1]
    mask_upper = track <= [1, 1]
    mask_upper = mask_upper[:, 0] & mask_upper[:, 1]
    mask = mask_lower & mask_upper
    return mask


def object_list_to_bpv(detections, num_obj_classes):
    bpv = np.zeros(num_obj_classes, dtype=np.float32)
    for dets_per_frame in detections:
        bpv[dets_per_frame] += 1
        # for detection in dets_per_frame:
        #     bpv[detection] = 1  # set 1 to the objects that are seen in any of the frames, not really important when
    return bpv/(np.max(bpv) if np.max(bpv) != 0 else 1)


def object_list_to_ocpv(detections, num_obj_classes, categories):
    opcv = np.zeros(num_obj_classes, dtype=np.float32)
    for dets_per_frame in detections:
        opcv[categories[dets_per_frame]] += 1
    return opcv/(np.max(opcv) if np.max(opcv) != 0 else 1)

class MultitaskDatasetLoaderVideoLevel(torch.utils.data.Dataset):
    def __init__(self, sampler, split_files, dataset_names, tasks_per_dataset, batch_transform, video_splits=25,
                 validation=False, vis_data=False):
        self.sampler = sampler
        self.video_list = list()
        self.dataset_infos = dict()
        self.sampled_centers = list()
        for i, (dataset_name, split_file, td) in enumerate(zip(dataset_names, split_files, tasks_per_dataset)):
            # charegoDataLineConstructor(dataset_name, 'video')
            data_line = CHAREGO1DataLineVideo if dataset_name == 'charego1' else CHAREGO3DataLineVideo
            img_tmpl = '{}-{:06d}.jpg'
            cls_tasks = CHARADES_CLS_TASKS
            max_num_classes = LABELS_CHARADES
            sub_with_flow = 'clips_frames\\'
            video_list, sampled_centers = read_samples_list_vid_level(split_file, data_line, video_splits)
            self.video_list += video_list
            self.sampled_centers += sampled_centers

            dat_info = DatasetInfo(i, dataset_name, data_line, img_tmpl, td, cls_tasks, max_num_classes,
                                   None, None, None, None, video_list, sub_with_flow)
            # self.maximum_target_size = tasks_for_dataset['max_target_size']
            self.dataset_infos[dataset_name] = dat_info

        self.transform = batch_transform
        self.validation = validation
        self.vis_data = vis_data

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        data_line = self.video_list[index]
        if isinstance(data_line, CHAREGO1DataLineVideo):
            dataset_name = 'charego1'
        elif isinstance(data_line, CHAREGO3DataLineVideo):
            dataset_name = 'charego3'
        else:
            # unknown type
            sys.exit("Wrong data_line in dataloader.__getitem__ for mAP. Exit code -3")

        dataset_info = self.dataset_infos[dataset_name]
        (path, uid), (start_frame, frame_count), use_stuff, data_paths = data_line.parse(dataset_info)
        use_hands, use_gaze, use_objects, use_obj_categories = use_stuff
        hand_path, gaze_path, obj_path = data_paths

        center = self.sampled_centers[index]
        sampled_idxs = self.sampler.sampling(range_max=32, v_id=index, start_frame=center-16)

        # leave the if clause just in case I ever enhance this for other datasets
        if dataset_name in ['charego1', 'charego3']:
            image_paths = [os.path.join(path, dataset_info.img_tmpl.format(uid, idx)) for idx in sampled_idxs]
        else:
            image_paths = [os.path.join(path, dataset_info.img_tmpl.format(idx)) for idx in sampled_idxs]

        clip_rgb, sampled_frames = load_images3(image_paths, self.vis_data)
        or_h, or_w, _ = clip_rgb.shape

        orig_norm_val = [or_w, or_h]

        if self.transform is not None:
            clip_rgb = self.transform(clip_rgb)

        if self.vis_data:
            track_idxs = (np.array(sampled_idxs) - start_frame).astype(np.int)
            visualize_item(sampled_frames, clip_rgb, track_idxs, use_hands, hand_path, None, None,
                           use_gaze, gaze_path, None, or_w, or_h, orig_norm_val, False, None, None)

        labels = list()
        all_cls_tasks = self.dataset_infos[dataset_name].cls_tasks
        all_cls_task_sizes = self.dataset_infos[dataset_name].max_num_classes.values()
        all_cls_tasks_names = list(self.dataset_infos[dataset_name].max_num_classes.keys())
        tasks_for_dataset = self.dataset_infos[dataset_name].td
        # mappings = self.dataset_infos[dataset_name].mappings
        for i, (cls_task, task_size) in enumerate(zip(all_cls_tasks, all_cls_task_sizes)):
            if cls_task in tasks_for_dataset:
                if cls_task == 'L' and dataset_name == 'adl':  # hack only for ADL location labels
                    label_num = data_line.label_location(sampled_idxs)
                else:
                    label_num = getattr(data_line, all_cls_tasks_names[i])
                mh = np.zeros(task_size)
                mh[np.array(label_num)] = 1
                labels = np.concatenate((labels, mh))

        labels_dtype = np.int64
        labels = np.array(labels, dtype=labels_dtype)
        if self.validation:
            to_return = (clip_rgb, labels, uid)
        else:
            dataset_id = self.dataset_infos[dataset_name].dataset_id
            # putting labels twice as a hack to use init_inputs_batch function in train_utils
            to_return = (clip_rgb, labels, labels, dataset_id)
        return to_return

class MultitaskDatasetLoader(torch.utils.data.Dataset):
    def __init__(self, sampler, split_files, dataset_names, tasks_per_dataset, batch_transform,
                 gaze_list_prefix, hand_list_prefix, object_list_prefix, object_categories,
                 validation=False, eval_gaze=False, vis_data=False, use_flow=False, flow_transforms=None,
                 only_flow=False):
        self.sampler = sampler
        # 1-1 association between dataset name, split file and resp tasks
        assert len(dataset_names) == len(tasks_per_dataset)
        self.video_list = list()
        self.dataset_infos = dict()
        self.maximum_target_size = 0
        for i, (dataset_name, split_file, td) in enumerate(zip(dataset_names, split_files, tasks_per_dataset)):
            glp = gaze_list_prefix.pop(0) if 'G' in td else None
            hlp = hand_list_prefix.pop(0) if 'H' in td else None
            olp = object_list_prefix.pop(0) if 'O' in td or 'C' in td else None
            ocp = object_categories.pop(0) if 'C' in td else None
            if dataset_name == 'epick':
                data_line = EPICDataLine
                img_tmpl = 'frame_{:010d}.jpg'
                cls_tasks = EPIC_CLS_TASKS
                max_num_classes = LABELS_EPIC
                sub_with_flow = 'rgb\\'
            elif dataset_name == 'egtea':
                data_line = GTEADataLine
                img_tmpl = 'img_{:05d}.jpg'
                cls_tasks = GTEA_CLS_TASKS
                max_num_classes = LABELS_GTEA
                sub_with_flow = 'clips_frames\\'
            elif dataset_name == 'somv1':
                data_line = SOMETHINGV1DataLine
                img_tmpl = '{:05d}.jpg'
                cls_tasks = SOMV1_CLS_TASKS
                max_num_classes = LABELS_SOMV1
                sub_with_flow = 'clips_frames\\'
            elif dataset_name == 'adl':
                data_line = ADLDataLine
                img_tmpl = '{:06d}.jpg'
                cls_tasks = ADL_CLS_TASKS
                max_num_classes = LABELS_ADL
                sub_with_flow = 'ADL_frames\\'
            elif dataset_name in ['charego1', 'charego3']:
                # charegoDataLineConstructor(dataset_name, 'sample')
                data_line = CHAREGO1DataLineSample if dataset_name == 'charego1' else CHAREGO3DataLineSample
                img_tmpl = '{}-{:06d}.jpg'
                cls_tasks = CHARADES_CLS_TASKS
                max_num_classes = LABELS_CHARADES
                sub_with_flow = 'clips_frames\\'
            else:
                # undeveloped dataset yet e.g. something something v2 or whatever
                sys.exit("Unknown dataset")
            video_list = read_samples_list(split_file, data_line)
            dat_info = DatasetInfo(i, dataset_name, data_line, img_tmpl, td, cls_tasks, max_num_classes, glp, hlp, olp,
                                   ocp, video_list, sub_with_flow)
            self.maximum_target_size = td['max_target_size'] if td['max_target_size'] > self.maximum_target_size else self.maximum_target_size
            self.dataset_infos[dataset_name] = dat_info
            self.video_list += video_list

        self.use_flow = use_flow
        self.transform = batch_transform
        self.flow_transforms = flow_transforms
        self.do_transforms = self.transform or self.flow_transforms
        self.validation = validation
        self.eval_gaze = eval_gaze
        self.vis_data = vis_data
        self.only_flow = only_flow
        if self.only_flow:
            self.use_flow = False

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        data_line = self.video_list[index]
        if isinstance(data_line, EPICDataLine):  # parse EPIC line
            dataset_name = 'epick'
        elif isinstance(data_line, GTEADataLine):  # parse EGTEA line
            dataset_name = 'egtea'
        elif isinstance(data_line, SOMETHINGV1DataLine):
            dataset_name = 'somv1'
        elif isinstance(data_line, ADLDataLine):
            dataset_name = 'adl'
        elif isinstance(data_line, CHAREGO1DataLineSample):
            dataset_name = 'charego1'
        elif isinstance(data_line, CHAREGO3DataLineSample):
            dataset_name = 'charego3'
        else:
            # unknown type
            sys.exit("Wrong data_line in dataloader.__getitem__. Exit code -2")
        # global sample_time, indices_time, rgb_time, rgb_transforms_time, hand_track_time, gaze_track_time, tracks_transform_time, bpv_time, ocpv_time, transforms_time, fin_tracks_time, labels_time

        # t = time.time()
        # retrieve sample specific information and data
        dataset_info = self.dataset_infos[dataset_name]
        (path, uid), (start_frame, frame_count), use_stuff, data_paths = data_line.parse(dataset_info)
        use_hands, use_gaze, use_objects, use_obj_categories = use_stuff
        hand_path, gaze_path, obj_path = data_paths
        # sample_time += (time.time() - t)

        # t = time.time()
        # get indices for the images to be loaded
        sampled_idxs = self.sampler.sampling(range_max=frame_count, v_id=index, start_frame=start_frame)
        # indices_time += (time.time() - t)

        if dataset_name in ['charego1', 'charego3']:
            image_paths = [os.path.join(path, dataset_info.img_tmpl.format(uid.split('_')[0], idx)) for idx in sampled_idxs]
        else:
            image_paths = [os.path.join(path, dataset_info.img_tmpl.format(idx)) for idx in sampled_idxs]

        # t = time.time()
        # load rgb images
        sampled_frames, clip_rgb = None, None
        if not self.only_flow:
            # clip_rgb, sampled_frames = load_rgb_clip(path, sampled_idxs, dataset_info)
            # clip_rgb, sampled_frames = load_images2(path, sampled_idxs, dataset_info.img_tmpl, self.vis_data)
            clip_rgb, sampled_frames = load_images3(image_paths, self.vis_data)
            or_h, or_w, _ = clip_rgb.shape
        # rgb_time += (time.time() - t)

        # load flow images
        sampled_flow, clip_flow = None, None
        if self.use_flow or self.only_flow:
            # if ever I use flow again this needs fixing for charades frame name convention
            clip_flow, sampled_flow = load_flow_clip(path, sampled_idxs, dataset_info, self.only_flow)
            if self.only_flow:
                or_h, or_w, _ = clip_flow.shape

        orig_norm_val = [or_w, or_h]  # apparently if I do not load any rgb or flow images I get a warning

        # t = time.time()
        track_idxs = (np.array(sampled_idxs) - start_frame).astype(np.int)
        # hands points is the final output, hand tracks is pickle, left and right track are intermediate versions
        left_track, right_track = load_hand_tracks(hand_path, track_idxs, use_hands)
        # hand_track_time += (time.time() - t)

        # t = time.time()
        # gaze points is the final output, gaze data is the pickle data, gaze track is intermediate versions
        gaze_track, gaze_mask = load_gaze_track(gaze_path, track_idxs, orig_norm_val, self.eval_gaze, use_gaze)
        # gaze_track_time += (time.time() - t)

        # t = time.time()
        bpv = None
        if use_objects:
            object_detections = load_pickle(obj_path)
            bpv = object_list_to_bpv(object_detections, 352)
        # bpv_time += (time.time() - t)

        # t = time.time()
        ocpv = None
        if use_obj_categories:
            object_detections = load_pickle(obj_path)
            categories = dataset_info.categories_per_noun
            ocpv = object_list_to_ocpv(object_detections, 20, categories)
        # ocpv_time += (time.time() - t)

        if bpv is not None:
            object_mask = [1] if bpv.max() != 0 else [0]
        elif ocpv is not None:
            object_mask = [1] if ocpv.max() != 0 else [0]
        else:
            object_mask = None

        # t = time.time()
        trns_norm_val = orig_norm_val
        if self.do_transforms:
            # tr = time.time()
            # transform the rgb frames
            if not self.only_flow:  # if only flow there is no rgb clip
                clip_rgb = self.transform(clip_rgb)
                _, _, max_h, max_w = clip_rgb.shape
            # rgb_transforms_time += (time.time() - tr)

            # apply transforms to tracks and flow
            if use_hands or use_gaze or self.use_flow or self.only_flow:
                # normal work-flow for rgb only or multibranch
                if not self.only_flow:
                    # There is rgb stream so get the tranform details for the other modalities from here
                    scale_x, scale_y, tl_x, tl_y, is_flipped, is_training = get_scale_and_shift(self.transform,
                                                                                                orig_norm_val)
                    # set predetermined flipping transform only if training for multi-branch network
                    if self.use_flow and is_training:
                        self.flow_transforms.transforms[2].set_flip(is_flipped)

                # if evaluating multi-branch or train/eval single-branch flow just apply the transforms
                if self.use_flow or self.only_flow:
                    clip_flow = self.flow_transforms(clip_flow)
                    if self.only_flow:
                        # There is only the flow branch, so get the transforms from here
                        scale_x, scale_y, tl_x, tl_y, is_flipped, is_training = get_scale_and_shift(
                            self.flow_transforms, orig_norm_val)
                        _, _, max_h, max_w = clip_flow.shape

                # tt = time.time()
                trns_norm_val = [max_w, max_h]
                if use_hands:  # apply transforms to hand tracks
                    left_track = apply_transform_to_track(left_track, scale_x, scale_y, tl_x, tl_y, trns_norm_val,
                                                          is_flipped)
                    right_track = apply_transform_to_track(right_track, scale_x, scale_y, tl_x, tl_y, trns_norm_val,
                                                           is_flipped)
                if use_gaze:  # apply transforms to gaze tracks
                    gaze_track = apply_transform_to_track(gaze_track, scale_x, scale_y, tl_x, tl_y, trns_norm_val,
                                                          is_flipped)
                # tracks_transform_time += (time.time() - tt)
        # transforms_time += (time.time() - t)

        left_track_vis, right_track_vis, gaze_track_vis = None, None, None
        if self.vis_data:
            if use_hands:
                left_track_vis = left_track
                right_track_vis = right_track
            if use_gaze:
                gaze_data = load_pickle(gaze_path)
                gaze_track_vis = np.array([[value[0], value[1]] for key, value in gaze_data.items()], dtype=np.float32)
                if 'DoubleFullSampling' not in self.sampler.__repr__():
                    gaze_track_vis = gaze_track_vis[track_idxs]
                gaze_track_vis *= orig_norm_val
                gaze_track_vis = apply_transform_to_track(gaze_track_vis, scale_x, scale_y, tl_x, tl_y, trns_norm_val,
                                                          is_flipped)

        # t = time.time()
        # regardless of the transforms the tracks should be normalized to [-1,1] for the dsnt layer
        norm_val = trns_norm_val if self.do_transforms else orig_norm_val
        left_hand_mask, right_hand_mask = None, None
        if use_hands:
            # subsample the hand tracks to reach the output size
            left_track = left_track[::2]
            right_track = right_track[::2]
            left_track = make_dsnt_track(left_track, norm_val)
            right_track = make_dsnt_track(right_track, norm_val)
            left_hand_mask = mask_coord_bounds(left_track)
            right_hand_mask = mask_coord_bounds(right_track)
        if use_gaze:
            gaze_track = make_dsnt_track(gaze_track, norm_val)
            gaze_mask = gaze_mask & mask_coord_bounds(gaze_track)
        # fin_tracks_time += (time.time() - t)

        if self.vis_data:
            visualize_item(sampled_frames, clip_rgb, track_idxs, use_hands, hand_path, left_track_vis, right_track_vis,
                           use_gaze, gaze_path, gaze_track_vis, or_w, or_h, norm_val, self.use_flow, sampled_flow,
                           clip_flow)

        # t = time.time()
        # get the classification task labels
        labels = list()
        masks = list()
        all_cls_tasks = self.dataset_infos[dataset_name].cls_tasks
        all_cls_tasks_names = list(self.dataset_infos[dataset_name].max_num_classes.keys())
        tasks_for_dataset = self.dataset_infos[dataset_name].td
        mappings = self.dataset_infos[dataset_name].mappings
        for i, cls_task in enumerate(all_cls_tasks):
            if cls_task in tasks_for_dataset:
                if cls_task == 'L' and dataset_name == 'adl': # hack only for ADL location labels
                    label_num = data_line.label_location(sampled_idxs)
                else:
                    label_num = getattr(data_line, all_cls_tasks_names[i])
                label_num = mappings[i][label_num] if mappings[i] is not None else label_num
                labels.append(label_num)

        labels_dtype = np.float32 if use_gaze or use_hands else np.int64
        labels = np.array(labels, dtype=labels_dtype)

        # get the coordinate task labels
        if use_gaze:
            gaze_points = gaze_track.astype(np.float32).flatten()
            labels = np.concatenate((labels, gaze_points))
            masks = np.concatenate((masks, gaze_mask)).astype(np.bool)
        if use_hands:
            hand_points = np.concatenate((left_track[:, np.newaxis, :], right_track[:, np.newaxis, :]), axis=1).astype(
                np.float32)
            hand_points = hand_points.flatten()
            labels = np.concatenate((labels, hand_points))
            masks = np.concatenate((masks, left_hand_mask, right_hand_mask)).astype(np.bool)
        if use_objects:
            bpv = bpv.flatten()
            labels = np.concatenate((labels, bpv))
        if use_obj_categories:
            ocpv = ocpv.flatten()
            labels = np.concatenate((labels, ocpv))
        if use_objects or use_obj_categories:
            masks = np.concatenate((masks, object_mask)).astype(np.bool)

        # this is for the dataloader only, to avoid having uneven sizes in the label/mask dimension of the batch
        if len(labels) < self.maximum_target_size:
            labels = np.concatenate((labels, [0.0] * (self.maximum_target_size - len(labels)))).astype(np.float32)
        if len(masks) < self.maximum_target_size:
            masks = np.concatenate((masks, [False] * (self.maximum_target_size - len(masks)))).astype(np.float32)
        # labels_time += (time.time() - t)

        # Validation is always dataset specific so no dataset_id is required,
        # however the uid is needed to log the results
        dataset_id = self.dataset_infos[dataset_name].dataset_id
        if self.validation:
            if self.use_flow:
                to_return = (clip_rgb, clip_flow, labels, masks, uid)
            elif self.only_flow:
                to_return = (clip_flow, labels, masks, uid)
            elif self.eval_gaze and use_gaze:  # this is not refined code
                gaze_data = load_pickle(gaze_path)
                orig_gaze = np.array([[value[0], value[1]] for key, value in gaze_data.items()],
                                     dtype=np.float32).flatten()
                to_return = (clip_rgb, clip_flow, labels, orig_gaze, uid) if self.use_flow else (clip_rgb, labels, orig_gaze, uid)
            else:
                to_return = (clip_rgb, labels, masks, uid)
        elif self.only_flow:
            to_return = (clip_flow, labels, masks, dataset_id)
        else:
            to_return = (clip_rgb, clip_flow, labels, masks, dataset_id) if self.use_flow else (
                clip_rgb, labels, masks, dataset_id)

        return to_return


# sample_time, indices_time, rgb_time, hand_track_time, gaze_track_time, bpv_time, ocpv_time = 0,0,0,0,0,0,0
# transforms_time, rgb_transforms_time, tracks_transform_time, fin_tracks_time, labels_time = 0,0,0,0,0
if __name__ == '__main__':
    # video_list_file = r"D:\Code\hand_track_classification\splits\epic_rgb_select2_56_nd\epic_rgb_train_1.txt"
    # video_list_file = r"D:\Code\hand_track_classification\splits\epic_rgb_brd\epic_rgb_train_1.txt"
    # video_list_file = r"D:\Code\hand_track_classification\splits\epic_rgb_select2_56_nd_brd\epic_rgb_train_1.txt"
    # video_list_file = r"D:\Code\hand_track_classification\vis_utils\21247.txt"

    # video_list_file = r"D:\Code\hand_track_classification\splits\gtea_rgb\fake_split2.txt"
    # video_list_file = r"splits\gtea_rgb_frames\fake_split3.txt"

    import torchvision.transforms as transforms
    from src.utils.dataset.dataset_loader_transforms import RandomScale, RandomCrop, RandomHorizontalFlip, RandomHLS_2, \
        ToTensorVid, Normalize, Resize, CenterCrop, PredefinedHorizontalFlip
    from src.utils.video_sampler import RandomSampling, MiddleSampling
    from src.utils.argparse_utils import parse_tasks_str

    mean_3d = [124 / 255, 117 / 255, 104 / 255]
    std_3d = [0.229, 0.224, 0.225]
    mean_1d = [0.5]
    std_1d = [0.226]

    seed = 0
    train_transforms = transforms.Compose([
        RandomScale(make_square=True, aspect_ratio=[0.8, 1. / 0.8], slen=[224, 288], seed=seed),
        RandomCrop((224, 224), seed=seed), RandomHorizontalFlip(seed=seed),
        RandomHLS_2(vars=[15, 35, 25]),
        ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
    train_flow_transforms = transforms.Compose([
        RandomScale(make_square=True, aspect_ratio=[0.8, 1. / 0.8], slen=[224, 288], seed=seed),
        RandomCrop((224, 224), seed=seed), PredefinedHorizontalFlip(),
        ToTensorVid(dim=2), Normalize(mean=mean_1d, std=std_1d)])

    test_transforms = transforms.Compose([Resize((256, 256), False), CenterCrop((224, 224)), ToTensorVid(),
                                          Normalize(mean=mean_3d, std=std_3d)])

    # test_sampler = MiddleSampling(num=16)
    # test_sampler = FullSampling()
    test_sampler = RandomSampling(num=16, interval=2, speed=[1.0, 1.0], seed=seed)

    #### tests
    # Note: before running tests put working directory as the "main" files
    # 1 test dataloader for epic kitchens
    # task_str = 'N352C20'  # "V125N352" # "A2513N352H" ok # "A2513V125N352H" ok # "V125N351" ok # "V125H" ok # "A2513V125N351H" ok  # "A2513V125N351GH" crashes due to 'G'
    # datasets = ['epick']
    # video_list_file = [r"other\splits\EPIC_KITCHENS\epic_rgb_new_nd_val_act\epic_rgb_new_val_1.txt"]
    # _hlp = ['hand_detection_tracks_lr005_new']
    # _glp = ['gaze_tracks']
    # _olp = ['tracked_noun_bpv_oh_new']
    # _oclp = [r"D:\Datasets\egocentric\EPIC_KITCHENS\EPIC_categories.csv"]

    # 2 test dataloader for egtea
    # task_str = "A106V19N53GH" # "N53GH" ok # "A106V19N53GH" ok
    # datasets = ['egtea']
    # video_list_file = [r"other\splits\EGTEA\fake_split3.txt"]
    # _hlp = ['hand_detection_tracks_lr005']
    # _glp = ['gaze_tracks']
    # _olp = [None]

    # 3 test dataloader for epic + gtea
    # task_str = "A2513V125N352HO352+A106V19N53GH"
    # datasets = ['epick', 'egtea']
    # video_list_file = [r"other\splits\EPIC_KITCHENS\epic_rgb_new_nd_val_act\epic_rgb_new_val_1_fake.txt", r"other\splits\EGTEA\fake_split3.txt"]
    # _hlp = ['hand_detection_tracks_lr005', 'hand_detection_tracks_lr005']
    # _glp = ['gaze_tracks']
    # _olp = ['noun_bpv_oh']

    # 4 test dataloader for gtea + epic
    # task_str = "A106V19N53GH+A2513V125N352HO352"
    # datasets = ['egtea', 'epick']
    # video_list_file = [r"other\splits\EGTEA\fake_split1.txt", r"other\splits\EPIC_KITCHENS\epic_rgb_new_nd_val_act\epic_rgb_new_val_1_fake.txt"]
    # _hlp = ['hand_detection_tracks_lr005', 'hand_detection_tracks_lr005']
    # _glp = ['gaze_tracks']
    # _olp = ['noun_bpv_oh']

    # 5 test dataloader for something something v1
    # task_str = "A174"
    # datasets = ['somv1']
    # video_list_file = [r"other\splits\SOMETHINGV1\fake_list.txt"]
    # _hlp = [None]
    # _glp = [None]
    # _olp = [None]

    # 6 test dataloader for charego1
    # task_str = "A157V33N38L16"
    # datasets = ['charego1']
    # video_list_file = [r"D:\Code\mtl_advanced\other\splits\CHARADES_EGO\fake1st_list.txt"]
    # _hlp = [None]
    # _glp = [None]
    # _olp = [None]
    # _oclp = [None]

    # 7 test dataloader for charego3
    # task_str = "A157V33N38L16"
    # datasets = ['charego3']
    # video_list_file = [r"D:\Code\mtl_advanced\other\splits\CHARADES_EGO\fake3rd_list.txt"]
    # _hlp = [None]
    # _glp = [None]
    # _olp = [None]
    # _oclp = [None]

    # 8 test dataloader for charego1+charego3
    # task_str = "A157V33N38L16+A157V33N38L16"
    # datasets = ['charego1', 'charego3']
    # video_list_file = [r"D:\Code\mtl_advanced\other\splits\CHARADES_EGO\fake1st_list.txt",
    #                    r"D:\Code\mtl_advanced\other\splits\CHARADES_EGO\fake3rd_list.txt"]
    # _hlp = [None]
    # _glp = [None]
    # _olp = [None]
    # _oclp = [None]

    # tpd = parse_tasks_str(task_str, datasets)
    # loader = MultitaskDatasetLoader(test_sampler, video_list_file, datasets, tasks_per_dataset=tpd,
    #                                 batch_transform=train_transforms, gaze_list_prefix=_glp, hand_list_prefix=_hlp,
    #                                 object_list_prefix=_olp, object_categories=_oclp,
    #                                 validation=False, eval_gaze=False, vis_data=True, use_flow=False,
    #                                 flow_transforms=train_flow_transforms, only_flow=False)



    # 9 test dataloader for charego1 video level
    # task_str = "A157V33N38L16"
    # datasets = ['charego1']
    # video_list_file = [r"D:\Code\mtl_advanced\other\splits\CHARADES_EGO\fake1st_listvl.txt"]
    # _hlp = [None]
    # _glp = [None]
    # _olp = [None]
    # _oclp = [None]

    # 10 test dataloader for charego1+charego3 video level
    task_str = "A157V33N38L16+A157V33N38L16"
    datasets = ['charego1', 'charego3']
    video_list_file = [r"D:\Code\mtl_advanced\other\splits\CHARADES_EGO\fake1st_listvl.txt",
                       r"D:\Code\mtl_advanced\other\splits\CHARADES_EGO\fake3rd_listvl.txt"]
    _hlp = [None]
    _glp = [None]
    _olp = [None]
    _oclp = [None]

    tpd = parse_tasks_str(task_str, datasets)
    loader = MultitaskDatasetLoaderVideoLevel(MiddleSampling(16), video_list_file, datasets, tpd,
                                              batch_transform=train_transforms, vis_data=True)

    # for ind in range(len(loader)):
    #     data_point = loader.__getitem__(ind)
    #     if loader.use_flow:
    #         _clip_input, _clip_flow, _labels, _dataset_id, _validation_id = data_point
    #     elif loader.only_flow:
    #         _clip_flow, _labels, _dataset_id, _validation_id = data_point
    #     else:
    #         _clip_input, _labels, _dataset_id, _validation_id = data_point
    #     print("\rItem {}: {}: {}".format(ind, _validation_id, _labels))

    import time

    t0 = time.time()
    for i in range(len(loader)):
        data_point = loader.__getitem__(i)
        print(data_point[1:])
    t1 = time.time()

    # print('time for rgb2hls', loader.transform.transforms[3].time_rgb2hls)
    # print('time for augmentations', loader.transform.transforms[3].time_aug)
    # print('time for hls2rgb', loader.transform.transforms[3].time_hls2rgb)
    print('total time', t1 - t0)
    # print('sample time', sample_time)
    # print('rgb time', rgb_time)
    # print('indices time', indices_time)
    # print('hand tracks time', hand_track_time)
    # print('gaze tracks time', gaze_track_time)
    # print('bpv time', bpv_time)
    # print('ocpv time', ocpv_time)
    # print('transforms time', transforms_time)
    # print('... rgb time', rgb_transforms_time)
    # print('... tracks time', tracks_transform_time)
    # print('fin tracks time', fin_tracks_time)
    # print('labels time', labels_time)
