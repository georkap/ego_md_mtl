import os, csv, argparse

locations = {'Basement (A room below the ground floor)': 0,
             'Bathroom': 1,
             'Bedroom': 2,
             'Closet / Walk-in closet / Spear closet': 3,
             'Dining room': 4,
             'Entryway (A hall that is generally located at the entrance of a house)': 5,
             'Garage': 6,
             'Hallway': 7,
             'Home Office / Study (A room in a house used for work)': 8,
             'Kitchen': 9,
             'Laundry room': 10,
             'Living room': 11,
             'Other': 12,
             'Pantry': 13,
             'Recreation room / Man cave': 14,
             'Stairs': 15}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir_path", type=str)
    parser.add_argument("charades_annot_file", type=str)
    parser.add_argument("charades_frame_counts_path", type=str)
    parser.add_argument("charades_mapping_path", type=str)
    parser.add_argument('out_file_name_with_path', type=str)
    parser.add_argument('--video_level', default=False, action='store_true')
    return parser.parse_args()

# used to create charades_counts.txt
# def parse_actual_dataset_frame_nums(dataset_dir):
#     out_file = open('charades_counts.txt', 'a')
#     dataset_ids = os.listdir(dataset_dir)
#     dataset_id_paths = [os.path.join(dataset_dir, x) for x in dataset_ids]
#     for i, dip in enumerate(dataset_id_paths):
#         num_frames = len(os.listdir(dip))
#         # id_counts[dataset_ids[i]] = num_frames
#         out_file.write('{} {}\n'.format(dataset_ids[i], num_frames))
#
#     out_file.close()

def parse_charades_counts(frame_counts_path):
    id_counts = {}
    with open(frame_counts_path, 'r') as f:
        lines = f.readlines()
    for l in lines:
        id, counts = l.strip().split()
        id_counts[id] = int(counts)
    return id_counts

def parse_charades_mappings(charades_mapping_path):
    mappings = {}
    with open(charades_mapping_file, 'r') as f:
        lines = f.readlines()
    for l in lines:
        act, noun, verb = l.strip().split()
        mappings[act] = (verb, noun)
    return mappings

def cls2int(x):
    return int(x[1:])

def parse_charades_csv(filename):
    labels = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            actions = row['actions']
            if actions == '':
                actions = []
                length = float(row['length'])
                scene = row['scene']
                quality = row['quality']
                relevance = row['relevance']
                verified = '1' if row['verified'] == 'Yes' else '0'
                print('skipped', vid, actions, length, scene, quality, relevance, verified)
                continue
            else:
                actions = [a.split(' ') for a in actions.split(';')]
                actions = [{'class': x, 'start': float(y), 'end': float(z)} for x, y, z in actions]
                length = float(row['length'])
                scene = row['scene']
                quality = row['quality']
                relevance = row['relevance']
                verified = '1' if row['verified'] == 'Yes' else '0'
            labels[vid] = (actions, length, scene, quality, relevance, verified)
    return labels


FPS = 24
args = parse_args()
data_dir = args.data_dir_path
charades_annot_file = args.charades_annot_file
charades_counts_file = args.charades_frame_counts_path
charades_mapping_file = args.charades_mapping_path
outfile_path = args.out_file_name_with_path
outfile_dir = os.path.dirname(outfile_path)
video_level = args.video_level
if not os.path.exists(outfile_dir):
    os.makedirs(outfile_dir)

id_counts = parse_charades_counts(charades_counts_file)
class_mappings = parse_charades_mappings(charades_mapping_file)
# annotations = pandas.read_csv(charades_annot_file)
annotations = parse_charades_csv(charades_annot_file)

outfile = open(outfile_path, 'a')

for key, val in annotations.items():
    actions, length, scene, quality, relevance, verified = val
    segment_dir = os.path.join(data_dir, key)
    location_class = locations[scene]
    num_video_frames = id_counts[key]
    l_actions, l_verbs, l_nouns = [], [], []
    for act_ind, act in enumerate(actions):
        label_action = act['class']
        label_verb, label_noun = class_mappings[label_action]
        action_class, verb_class, noun_class = cls2int(label_action), cls2int(label_verb), cls2int(label_noun) # labels are 0-based
        if not video_level:
            uid = "{}_{}".format(key, act_ind)
            start_time = act['start']
            end_time = act['end']
            if end_time <= start_time: # (!)
                continue
            if end_time > length:
                # crop end times to the clip length if needed following https://github.com/gsig/charades-algorithms/issues/5
                end_time = length
            start_frame = int(start_time / length * num_video_frames)
            end_frame = int(end_time / length * num_video_frames)
            assert end_frame <= num_video_frames
            num_segment_frames = end_frame - start_frame # end frame should be exclusive so I do not add +1 for the 0-based start_frames
            if num_segment_frames < 4:
                continue

            out_line = "{} {} {} {} {} {} {} {}\n".format(segment_dir, num_segment_frames, verb_class, noun_class,
                                                          location_class, uid, start_frame, action_class)
            outfile.write(out_line)
        else:
            l_actions.append(action_class)
            l_verbs.append(verb_class)
            l_nouns.append(noun_class)
    if video_level:
        uid = "{}".format(key)
        start_frame = 0
        action_class = "-".join("{}".format(aa) for aa in l_actions)
        verb_class = "-".join("{}".format(vv) for vv in l_verbs)
        noun_class = "-".join("{}".format(nn) for nn in l_nouns)

        out_line = "{} {} {} {} {} {} {} {}\n".format(segment_dir, num_video_frames, verb_class, noun_class,
                                                      location_class, uid, start_frame, action_class)
        outfile.write(out_line)
    # break

outfile.close()
