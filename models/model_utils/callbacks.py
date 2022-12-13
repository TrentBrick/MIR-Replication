
from composer import Callback, Algorithm, Event
import torch 
import torch.nn.utils.prune as prune
from torch.nn import functional as F
from core_scripts import global_timestamp

class SaveOutHParams(Callback):
    def after_load(self, state, logger):
        torch.save(vars(state.model.params), f"{state.model.params.output_directory}{state.model.params.test_name}/hyperparams.pt")
        
class SyncGlobalTimeStamp(Callback):
    def fit_start(self, state, logger):
        global_timestamp.init(state.timestamp.batch._value, state.timestamp.epoch._value)

    def batch_end(self, state, logger):
        global_timestamp.CURRENT_TRAIN_STEP += 1

    def epoch_end(self, state, logger):
        global_timestamp.CURRENT_EPOCH += 1
