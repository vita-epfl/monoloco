import os
import sys

# Python does not consider the current directory to be a package
sys.path.insert(0, os.path.join('..', 'monoloco'))

from monoloco.train import Trainer


def test_trainer():
    joints = 'tests/joints_sample.json'
    trainer = Trainer(joints=joints, epochs=150)
    _ = trainer.train()
    dic_err, model = trainer.evaluate()
    assert dic_err['val']['all']['mean'] < 2.2
