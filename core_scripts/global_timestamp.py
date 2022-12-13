def init(global_step, current_epoch):
    global CURRENT_TRAIN_STEP
    global CURRENT_EPOCH
    CURRENT_TRAIN_STEP = global_step
    CURRENT_EPOCH = current_epoch