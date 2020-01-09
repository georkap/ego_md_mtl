"""
Takes a folder with frame ranges and action annotations for the ADL dataset and produces a split file suitable for
mtl advanced
"""

import os, argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('annots_dir_path')
    parser.add_argument('locations_dir_path')
    parser.add_argument('out_file_name_with_path', type=str)
    return parser.parse_args()

def search_for_locations(locations_lines, start_a, end_a):
    loc_annots = "-"
    start_a, end_a = int(start_a), int(end_a)
    for l in locations_lines:
        start_l, end_l, loc = l.strip().split()
        start_l, end_l = int(start_l), int(end_l)
        if start_a > end_l: # no overlap action starts after location -> continue
            continue
        elif end_a < start_l: # action ends before location -> break
            break
        else:
            # start of action inside location range
            # ->location valid until end of action or location, whichever comes first
            if start_l <= start_a <= end_l:
                overlap = min(end_l, end_a) - start_a + 1
                loc_annots += "{}:{}-".format(int(loc)-1, overlap) # turn location into 0-based
            # end of action inside location range but not its start
            # ->location is valid from start of location until end of action
            elif start_l <= end_a <= end_l:
                overlap = end_a - start_l + 1
                loc_annots += "{}:{}-".format(int(loc)-1, overlap)
    return loc_annots


args = parse_args()
annots_dir = args.annots_dir_path
locations_dir = args.locations_dir_path
outfile_path = args.out_file_name_with_path
outfile_dir = os.path.dirname(outfile_path)
if not os.path.exists(outfile_dir):
    os.makedirs(outfile_dir)

annot_names = os.listdir(annots_dir)
annot_files = [os.path.join(annots_dir, x) for x in annot_names]
outfile = open(outfile_path, 'a')

uid = -1
base_dir = "data\\ADL\\ADL_frames"
for i, annot_file in enumerate(annot_files):
    video_id = str(annot_names[i].split('.')[0].split('_')[1])
    with open(annot_file) as af:
        lines = af.readlines()

    location_file = os.path.join(locations_dir, "L{}.txt".format(video_id))
    with open(location_file) as lf:
        loc_lines = lf.readlines()

    annot_name = "P_" + video_id
    for l in lines:
        data = l.strip().split()
        start_frame = data[0]
        end_frame = data[1]
        action = int(data[2]) - 1 # turn action into 0-based
        if action < 0: # to avoid a weirdly labeled action out of the action set
            continue
        locations = search_for_locations(loc_lines, start_frame, end_frame)
        segment_dir = os.path.join(base_dir, annot_name)
        num_frames = int(end_frame) - int(start_frame) + 1 # assume inclusive of end frame
        uid += 1 # assign manual uids
        output_line = "{} {} {} {} {} {}\n".format(segment_dir, num_frames, uid, start_frame, action, locations)
        outfile.write(output_line)
outfile.close()






