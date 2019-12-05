import numpy as np


class DatasetInfo(object):
    def __init__(self, dataset_id, dataset_name, data_line, img_tmpl, tasks_for_dataset, cls_tasks, max_num_classes,
                 gaze_list_prefix, hand_list_prefix, object_list_prefix, video_list, sub_with_flow):
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.data_line = data_line
        self.img_tmpl = img_tmpl
        self.gaze_list_prefix = gaze_list_prefix
        self.hand_list_prefix = hand_list_prefix
        self.object_list_prefix = object_list_prefix
        self.td = tasks_for_dataset
        self.cls_tasks = cls_tasks
        self.max_num_classes = max_num_classes
        self.sub_with_flow = sub_with_flow

        self.num_classes = list()
        self.mappings = list()

        for task_short, task_long in zip(cls_tasks, list(max_num_classes.keys())): # works in python 3.7 because dicts are ordered
            if task_short in tasks_for_dataset:
                value = tasks_for_dataset[task_short]
                self.num_classes.append(value)
                if value != max_num_classes[task_long]:
                    self.mappings.append(make_class_mapping_generic(video_list, task_long))
                else:
                    self.mappings.append(None)
            else:
                self.mappings.append(None)


def make_class_mapping_generic(samples_list, attribute):
    classes = []
    for sample in samples_list:
        label = getattr(sample, attribute)
        if label not in classes:
            classes.append(label)
    classes = np.sort(classes)
    mapping_dict = {}
    for i, c in enumerate(classes):
        mapping_dict[c] = i
    return mapping_dict
