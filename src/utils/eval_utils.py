import math
import numpy as np
from scipy import ndimage
from src.utils.file_utils import print_and_save

def unnorm_gaze_coords(_coords):  # expecting values in [-1, 1]
    return ((_coords + 1) * 224 - 1) / 2

def calc_aae(pred, gt):
    # input should be [2] with modalities=1
    d = 112/math.tan(math.pi/6)
    pred = pred - 112
    gt = gt - 112
    r1 = np.array([pred[0], pred[1], d])  # x, y are inverted in numpy but it doesn't change results
    r2 = np.array([gt[0], gt[1], d])
    # angles needs to be of dimension batch*temporal*modalities*1
    angles = math.atan2(np.linalg.norm(np.cross(r1, r2)), np.dot(r1, r2))

    # angles_deg = math.degrees(angles)
    angles_deg = np.rad2deg(angles)
    return angles_deg

def calc_auc(pred, gt):
    z = np.zeros((224, 224))
    z[int(pred[0])][int(pred[1])] = 1
    z = ndimage.filters.gaussian_filter(z, 14)
    z = z - np.min(z)
    z = z / np.max(z)
    atgt = z[int(gt[0])][int(gt[1])]  # z[i][j]
    fpbool = z > atgt
    auc = (1 - float(fpbool.sum()) / (224 * 224))
    return auc

def inner_batch_calc(_model, _inputs, _gaze_targets, _or_targets, _frame_counter, _actual_frame_counter, _aae_frame,
                     _auc_frame, _aae_temporal, _auc_temporal, _to_print, _log_file, _mf_remaining=8):

    # _outputs, _coords, _heatmaps = _model(_inputs)
    network_output = _model(_inputs)
    _outputs, _coords, _heatmaps, _probabilities, _objects, _obj_cat = network_output

    _gaze_coords = _coords[:, :, 0, :]
    _gaze_coords = unnorm_gaze_coords(_gaze_coords).cpu().numpy()

    _batch_size, _temporal_size, _ = _gaze_targets.shape
    for _b in range(_batch_size): # this will always be one, otherwise torch.stack complains for variable temporal dim.
        _aae_temp = []
        _auc_temp = []
        for _t in range(_temporal_size-_mf_remaining, _temporal_size):
            # after transforms target gaze might be off the image. this is not evaluated
            _actual_frame_counter += 1
            if _gaze_targets[_b, _t][0] < 0 or _gaze_targets[_b, _t][0] >= 224 or _gaze_targets[_b, _t][1] < 0 or \
                    _gaze_targets[_b, _t][1] >= 224: # gt out of evaluated area after cropping
                continue
            if _or_targets[_b, _t][0] == 0 and _or_targets[_b, _t][1] == 0: # bad ground truth
                continue
            _frame_counter += 1
            _angle_deg = calc_aae(_gaze_coords[_b, _t], _gaze_targets[_b, _t])
            _aae_temp.append(_angle_deg)
            _aae_frame.update(_angle_deg)  # per frame

            _auc_once = calc_auc(_gaze_coords[_b, _t], _gaze_targets[_b, _t])
            _auc_temp.append(_auc_once)
            _auc_frame.update(_auc_once)
        if len(_aae_temp) > 0:
            _aae_temporal.update(np.mean(_aae_temp))  # per video segment
        if len(_auc_temp) > 0:
            _auc_temporal.update(np.mean(_auc_temp))

    _to_print += '[Gaze::aae_frame {:.3f}[avg:{:.3f}], aae_temporal {:.3f}[avg:{:.3f}],'.format(_aae_frame.val,
                                                                                                _aae_frame.avg,
                                                                                                _aae_temporal.val,
                                                                                                _aae_temporal.avg)
    _to_print += '::auc_frame {:.3f}[avg:{:.3f}], auc_temporal {:.3f}[avg:{:.3f}]]'.format(_auc_frame.val,
                                                                                           _auc_frame.avg,
                                                                                           _auc_temporal.val,
                                                                                           _auc_temporal.avg)
    print_and_save(_to_print, _log_file)
    return _auc_frame, _auc_temporal, _aae_frame, _aae_temporal, _frame_counter, _actual_frame_counter
