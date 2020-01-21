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

gtea_mapped_verbs = {0: 50, 1: 2, 2: 0, 3: 5, 4: 12, 5: 1, 6: 45, 7: 3, 8: 9, 9: 4,
                     10: 36, 11: 15, 12: 43, 13: 4, 14: 6, 15: 7, 16: 23, 17: 31, 18: 21}

gtea_mapped_nouns = { 0: 307,  1: 10,  2: 75,  3: 30,  4: 3,    5: 8,   6: 77,  7: 94,  8: 47,  9: 40,
                     10: 13,  11: 9,  12: 4,  13: 6,  14: 108, 15: 20, 16: 46, 17: 22, 18: 66, 19: 133,
                     20: 148, 21: 52, 22: 1,  23: 73, 24: 207, 25: 15, 26: 9,  27: 12, 28: 77, 29: 23,
                     30: 78,  31: 20, 32: 19, 33: 21, 34: 49,  35: 77, 36: 60, 37: 64, 38: 77, 39: 51,
                     40: 24,  41: 77, 42: 83, 43: 54, 44: 35,  45: 20, 46: 31, 47: 50, 48: 17, 49: 206,
                     50: 70,  51: 55, 52: 132}
