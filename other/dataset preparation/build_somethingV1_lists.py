import os
import pandas
import argparse

def parse_labels(labels_csv_path):
    labels_csv = pandas.read_csv(labels_csv_path, sep=';', header=None)
    labels_dict = labels_csv.to_dict('index')
    _labels = dict()
    for key, value in labels_dict.items():
        _labels[value[0]] = key
    return _labels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', type=str)
    parser.add_argument('labels_csv', type=str)
    parser.add_argument('file_name_with_path', type=str)
    return parser.parse_args()


args = parse_args()
data_csv = pandas.read_csv(args.input_csv, sep=';')
labels = parse_labels(args.labels_csv)
f = open(args.file_name_with_path, 'a')

base_dir = "data\\SOMETHINGV1\\clips_frames"
for ind, val in data_csv.iterrows():
    vid_id = val[0]
    text_label = val[1]
    numeric_label = labels[text_label]
    action_vid_dir = os.path.join(base_dir, "{}".format(vid_id))
    # num_frames = os.listdir(action_vid_dir)
    line = "{} {} {}\n".format(action_vid_dir, numeric_label, text_label)
    f.write(line)
f.close()
