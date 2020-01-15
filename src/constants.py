mean_3d = [124 / 255, 117 / 255, 104 / 255]
std_3d = [0.229, 0.224, 0.225]
mean_1d = [0.5]
std_1d = [0.226]

EPIC_CLASSES = [2513, 125, 352] # AVN
LABELS_EPIC = {'label_action': EPIC_CLASSES[0], 'label_verb': EPIC_CLASSES[1], 'label_noun': EPIC_CLASSES[2]}
EPIC_CLS_TASKS = ['A', 'V', 'N']
EPIC_OBJ_TASKS = ['O', 'C'] # O for object classes, C for object categories
EPIC_OBJECTS = [352]
EPIC_OBJ_CAT = [20]

GTEA_CLASSES = [106, 19, 53] # AVN
LABELS_GTEA = {'label_action': GTEA_CLASSES[0], 'label_verb': GTEA_CLASSES[1], 'label_noun': GTEA_CLASSES[2]}
GTEA_CLS_TASKS = ['A', 'V', 'N']

SOMETHINGV1_CLASSES = [174] # A
LABELS_SOMV1 = {'label_action': SOMETHINGV1_CLASSES[0]}
SOMV1_CLS_TASKS = ['A']

ADL_CLASSES = [31, 8] # AL
LABELS_ADL = {'label_action': ADL_CLASSES[0], 'label_location': ADL_CLASSES[1]}
ADL_CLS_TASKS = ['A', 'L']

CHARADES_CLASSES = [157, 33, 38, 16] # AVNL
LABELS_CHARADES = {'label_action': CHARADES_CLASSES[0], 'label_verb': CHARADES_CLASSES[1],
                   'label_noun': CHARADES_CLASSES[2], 'label_location': CHARADES_CLASSES[3]}
CHARADES_CLS_TASKS = ['A', 'V', 'N', 'L']
