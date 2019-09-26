import os
import numpy as np
import pandas

def init_annot_and_splits(annotations_file, split_type='val'):
    # init annotation file and splits
    annotations = pandas.read_csv(annotations_file)

    # get number of verb and noun classes for the full "train" dataset as discussed in the paper
    verb_classes = len(np.unique(annotations.verb_class.values))
    noun_classes = len(np.unique(annotations.noun_class.values))
    print("Classes in original train dataset, verbs:{}, nouns:{}".format(verb_classes, noun_classes))

    unavailable = [9, 11, 18]
    available_pids = ["P{:02d}".format(i) for i in range(1, 32) if i not in unavailable]

    if split_type == "75":  # val PID = [25, 26, 27, 28, 29, 30, 31]
        split_size = 21
    elif split_type == "brd":  # val PID = [26, 27, 28, 29, 30, 31]
        split_size = 22
    else:  # elif args.split_type == "val":  # val PID = [30, 31]
        split_size = 26

    split_1 = {}
    for i in range(28):
        split_1[available_pids[i]] = "train" if i < split_size else "val"
    split_dicts = [split_1]

    return annotations, split_dicts


def get_manyhot_classes(annotations_file, split_path, split_type, num_instances, actions_file):
    annotations, split_dicts = init_annot_and_splits(annotations_file, split_type=split_type)
    # export action classes from verbs and nouns with more than 100 instances and that exist at least once in training
    split_id = int(os.path.basename(split_path).split('.')[0].split('_')[-1]) - 1
    split = split_dicts[split_id]
    train = [k for (k, i) in split.items() if i == 'train']
    # val = [k for (k, i) in split.items() if i == 'val']

    verbs_t_un, verbs_t_count = np.unique(annotations.loc[annotations['participant_id'].isin(train)].
                                          verb_class.values, return_counts=True)
    nouns_t_un, nouns_t_count = np.unique(annotations.loc[annotations['participant_id'].isin(train)].
                                          noun_class.values, return_counts=True)

    actions_t_all = annotations[annotations['participant_id'].isin(train)][['verb_class', 'noun_class']]

    # find action combinations
    verbs_training = dict(zip(verbs_t_un, verbs_t_count))
    nouns_training = dict(zip(nouns_t_un, nouns_t_count))

    verbs_t_instances, nouns_t_instances = {}, {}
    action_t_combinations = []
    for key, item in verbs_training.items():
        if int(item) >= num_instances:
            verbs_t_instances[key] = item
    for key, item in nouns_training.items():
        if int(item) >= num_instances:
            nouns_t_instances[key] = item
    for key_verb, _ in verbs_t_instances.items():
        for key_noun, __ in nouns_t_instances.items():
            # to avoid the warning from the line below and make sure it works properly I split it in two steps
            temp = actions_t_all[actions_t_all.verb_class == int(key_verb)]
            temp = temp[temp.noun_class == int(key_noun)]
            if len(temp) > 0:
                action_t_combinations.append("{}_{}".format(key_verb, key_noun))

    all_actions = pandas.read_csv(actions_file)
    action_t_classes = dict()
    for key in action_t_combinations:
        action_id = all_actions[all_actions.class_key == key].action_id.item()
        action_t_classes[action_id] = True

    action_t_classes = list(action_t_classes.keys())
    action_t_classes.sort()

    verb_ids_sorted = list(reversed(np.argsort(verbs_t_count)))
    noun_ids_sorted = list(reversed(np.argsort(nouns_t_count)))

    return action_t_classes, list(verbs_t_instances.keys()), verb_ids_sorted, list(nouns_t_instances.keys()), noun_ids_sorted
