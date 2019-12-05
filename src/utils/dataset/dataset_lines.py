import os

class DataLine(object):
    def __init__(self, row):
        self.data = row.strip().split(' ')

    @property
    def data_path(self):
        return self.data[0]


class EPICDataLine(DataLine):
    def __init__(self, row):
        super(EPICDataLine, self).__init__(row)

    @property
    def num_frames(self):  # sto palio format ayto einai to start_frame
        return int(self.data[1])

    @property
    def label_verb(self):
        return int(self.data[2])

    @property
    def label_noun(self):
        return int(self.data[3])

    @property
    def uid(self):
        return int(self.data[4] if len(self.data) > 4 else -1)

    @property
    def start_frame(self):
        return int(self.data[5] if len(self.data) > 5 else -1)

    @property
    def label_action(self):
        return int(self.data[6] if len(self.data) > 6 else -1)

    def parse(self, dataset_info):
        uid = self.uid
        frame_count = self.num_frames
        start_frame = self.start_frame if self.start_frame != -1 else 0
        use_hands, use_gaze, use_objects = False, False, False
        hand_track_path, gaze_track_path, obj_track_path = None, None, None
        if 'H' in dataset_info.td:
            use_hands = True
            path_d, path_ds, a, b, c, pid, vid_id = self.data_path.split("\\")
            hand_track_path = os.path.join(path_d, path_ds, dataset_info.hand_list_prefix, pid, vid_id,
                                           "{}_{}_{}.pkl".format(start_frame, self.label_verb, self.label_noun))
        if 'O' in dataset_info.td:
            use_objects = True
            path_d, path_ds, a, b, c, pid, vid_id = self.data_path.split("\\")
            obj_track_path = os.path.join(path_d, path_ds, dataset_info.object_list_prefix, pid, vid_id,
                                          "{}_{}_{}.pkl".format(start_frame, self.label_verb, self.label_noun))
        return (self.data_path, uid), (start_frame, frame_count), (use_hands, use_gaze, use_objects), (hand_track_path, gaze_track_path, obj_track_path)


class GTEADataLine(DataLine):
    def __init__(self, row):
        super(GTEADataLine, self).__init__(row)
        self.data_len = len(row)

    @property
    def frames_path(self):
        path_parts = os.path.normpath(self.data[0]).split(os.sep)
        session_parts = path_parts[-1].split('-')
        session = session_parts[0] + '-' + session_parts[1] + '-' + session_parts[2]
        return os.path.join(path_parts[-4], path_parts[-3]), os.path.join(path_parts[-4], path_parts[-3], path_parts[-2], session, path_parts[-1])

    @property
    def instance_name(self):
        return os.path.normpath(self.data[0]).split(os.sep)[-1]

    @property
    def label_action(self): # to zero based labels
        return int(self.data[1]) - 1

    @property
    def label_verb(self):
        return int(self.data[2]) - 1

    @property
    def label_noun(self):
        return int(self.data[3]) - 1

    @property
    def extra_nouns(self):
        extra_nouns = list()
        if self.data_len > 4:
            for noun in self.data[4:]:
                extra_nouns.append(int(noun) - 1)
        return extra_nouns

    def parse(self, dataset_info):
        base_path, path = self.frames_path
        instance_name = self.instance_name
        uid = instance_name
        frame_count = len(os.listdir(path))
        assert frame_count > 0
        use_hands, use_gaze, use_objects = False, False, False
        hand_track_path, gaze_track_path, obj_track_path = None, None, None
        start_frame = 0
        if 'H' in dataset_info.td:
            use_hands = True
            hand_track_path = os.path.join(base_path, dataset_info.hand_list_prefix, instance_name + '.pkl')
        if 'G' in dataset_info.td:
            use_gaze = True
            gaze_track_path = os.path.join(base_path, dataset_info.gaze_list_prefix, instance_name + '.pkl')
        return (path, uid), (start_frame, frame_count), (use_hands, use_gaze, use_objects), (hand_track_path, gaze_track_path, obj_track_path)

class SOMETHINGV1DataLine(DataLine):
    def __init__(self, row):
        super(SOMETHINGV1DataLine, self).__init__(row)

    @property
    def uid(self): # return as str not int
        return self.data[0].split("\\")[-1]

    @property
    def label_action(self):
        return self.data[1]

    @property
    def label_action_str(self):
        action_str = ""
        for d in self.data[1:]:
            action_str += "{} ".format(d)
        return action_str[:-1]

    def parse(self, dataset_info):
        start_frame = 1
        frame_count = len(os.listdir(self.data_path))
        assert frame_count > 0
        use_hands, use_gaze, use_objects = False, False, False
        hand_track_path, gaze_track_path, obj_track_path = None, None, None
        if 'H' in dataset_info.td:
            use_hands = True
            hand_track_path = os.path.join(self.data_path.replace("clips_frames", dataset_info.hand_list_prefix), '.pkl')

        return (self.data_path, self.uid), (start_frame, frame_count), (use_hands, use_gaze, use_objects), (hand_track_path, gaze_track_path, obj_track_path)
