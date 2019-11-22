mean_3d = [124 / 255, 117 / 255, 104 / 255]
std_3d = [0.229, 0.224, 0.225]
mean_1d = [0.5]
std_1d = [0.226]

EPIC_CLASSES = [2513, 125, 352]
LABELS_EPIC = {'label_action': EPIC_CLASSES[0], 'label_verb': EPIC_CLASSES[1], 'label_noun': EPIC_CLASSES[2]}
EPIC_CLS_TASKS = ['A', 'V', 'N']

GTEA_CLASSES = [106, 19, 53]
LABELS_GTEA = {'label_action': GTEA_CLASSES[0], 'label_verb': GTEA_CLASSES[1], 'label_noun': GTEA_CLASSES[2]}
GTEA_CLS_TASKS = ['A', 'V', 'N']