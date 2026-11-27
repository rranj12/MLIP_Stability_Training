"""Keep StABlE's EMA export and save a matched raw model/optimizer at that step."""

import runpy
import sys
from pathlib import Path

import torch


def preserve_best_training_state(trainer_class):
    original_save = trainer_class.save

    def save(self, metrics=None, checkpoint_file="checkpoint.pt", training_state=True):
        original_save(self, metrics, checkpoint_file, training_state)
        if checkpoint_file != "best_checkpoint.pt" or self.is_debug:
            return
        from mdsim.common import distutils

        if not distutils.is_master():
            return
        # original_save has restored the raw parameters after exporting EMA weights.
        names = {id(p): n.removeprefix("module.") for n, p in self.model.named_parameters()}
        torch.save({
            "state_dict": {n.removeprefix("module."): p for n, p in self.model.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "optimizer_class": type(self.optimizer).__name__,
            "optimizer_param_names": [[names[id(p)] for p in g["params"]]
                                      for g in self.optimizer.param_groups],
            "step": self.step,
        }, Path(self.config["cmd"]["checkpoint_dir"]) / "best_training_checkpoint.pt")

    trainer_class.save = save


if __name__ == "__main__":
    stable = Path(__file__).resolve().parents[1] / "third_party/StABlE-Training"
    sys.path.insert(0, str(stable))
    from mdsim.trainers.trainer import Trainer

    preserve_best_training_state(Trainer)
    runpy.run_path(str(stable / "pretraining.py"), run_name="__main__")
