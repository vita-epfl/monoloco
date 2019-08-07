"""Test if the main modules of the package run correctly"""

import os
import sys
import json

# Python does not consider the current directory to be a package
sys.path.insert(0, os.path.join('..', 'monoloco'))

from monoloco.train import Trainer
from monoloco.network import MonoLoco
from monoloco.network.process import preprocess_pifpaf, factory_for_gt


def test_package():

    # Training test
    joints = 'tests/joints_sample.json'
    trainer = Trainer(joints=joints, epochs=150, lr=0.01)
    _ = trainer.train()
    dic_err, model = trainer.evaluate()
    assert dic_err['val']['all']['mean'] < 2

    # Prediction test
    path_keypoints = 'tests/002282.png.pifpaf.json'
    with open(path_keypoints, 'r') as f:
        pifpaf_out = json.load(f)

    kk, _ = factory_for_gt(im_size=[1240, 340])

    # Preprocess pifpaf outputs and run monoloco
    boxes, keypoints = preprocess_pifpaf(pifpaf_out)
    monoloco = MonoLoco(model)
    outputs, varss = monoloco.forward(keypoints, kk)
    dic_out = monoloco.post_process(outputs, varss, boxes, keypoints, kk)


