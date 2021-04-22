
"""
Adapted from https://github.com/openpifpaf/openpifpaf/blob/main/tests/test_train.py,
which is: 'Copyright 2019-2021 by Sven Kreiss and contributors. All rights reserved.'
and licensed under GNU AGPLv3
"""

import os
import subprocess

import gdown

OPENPIFPAF_MODEL = 'https://drive.google.com/uc?id=1b408ockhh29OLAED8Tysd2yGZOo0N_SQ'

TRAIN_COMMAND = [
    'python3', '-m', 'monoloco.run',
    'train',
    '--mode=stereo',
    '--joints', 'tests/sample_joints-kitti-stereo.json',
    '--lr=0.001',
    '-e=20',
    ]


PREDICT_COMMAND = [
    'python3', '-m', 'monoloco.run',
    'predict',
    '--mode=stereo',
    '--glob', 'docs/000840*.png',
    '--output_types', 'multi', 'json',
    '--decoder-workers=0',  # for windows'
]


def test_train_stereo(tmp_path):
    # train a model
    train_cmd = TRAIN_COMMAND + ['--out={}'.format(os.path.join(tmp_path, 'train_test.pkl'))]
    print(' '.join(train_cmd))
    subprocess.run(train_cmd, check=True, capture_output=True)
    print(os.listdir(tmp_path))

    # find the trained model checkpoint
    final_model = next(iter(f for f in os.listdir(tmp_path) if f.endswith('.pkl')))
    pifpaf_model = os.path.join(tmp_path, 'pifpaf_model.pkl')
    print('Downloading OpenPifPaf model in temporary folder')
    gdown.download(OPENPIFPAF_MODEL, pifpaf_model)

    # run predictions with that model
    model = os.path.join(tmp_path, final_model)

    predict_cmd = PREDICT_COMMAND + [
        '--model={}'.format(model),
        '--checkpoint={}'.format(pifpaf_model),
        '-o={}'.format(tmp_path),
    ]
    print(' '.join(predict_cmd))
    subprocess.run(predict_cmd, check=True, capture_output=True)
    print(os.listdir(tmp_path))
    assert 'out_000840.png.multi.png' in os.listdir(tmp_path)
