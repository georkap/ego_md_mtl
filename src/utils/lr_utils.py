import sys

from torch.optim.lr_scheduler import StepLR, MultiStepLR
from src.utils.learning_rates import CyclicLR, GroupMultistep

def load_lr_scheduler(lr_type, lr_steps, optimizer, train_iterator_length):
    if lr_type == 'step':
        lr_scheduler = StepLR(optimizer, step_size=int(lr_steps[0]), gamma=float(lr_steps[1]))
    elif lr_type == 'multistep':
        lr_scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in lr_steps[:-1]], gamma=float(lr_steps[-1]))
    elif lr_type == 'clr':
        lr_scheduler = CyclicLR(optimizer, base_lr=float(lr_steps[0]),
                                max_lr=float(lr_steps[1]), step_size_up=int(lr_steps[2])*train_iterator_length,
                                step_size_down=int(lr_steps[3])*train_iterator_length, mode=str(lr_steps[4]),
                                gamma=float(lr_steps[5]))
    elif lr_type == 'groupmultistep':
        lr_scheduler = GroupMultistep(optimizer,
                                      milestones=[int(x) for x in lr_steps[:-1]],
                                      gamma=float(lr_steps[-1]))
    else:
        sys.exit("Unsupported lr type")
    return lr_scheduler
