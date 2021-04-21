"""
Adapted from https://github.com/openpifpaf/openpifpaf/blob/main/tests/test_train.py,
which is: 'Copyright 2019-2021 by Sven Kreiss and contributors. All rights reserved.'
and licensed under GNU AGPLv3
"""

import os
import subprocess


TRAIN_COMMAND = [
    'python3', '-m', 'monoloco.run',
    'train',
    '--joints', 'tests/sample_joints-kitti-mono.json',
    '--lr=0.001',
    '-e=10',
    ]


PREDICT_COMMAND = [
    'python3', '-m', 'monoloco.run',
    'predict',
    'docs/002282.png',
    '--output_types', 'multi', 'json',
]

PREDICT_COMMAND_SOCIAL_DISTANCE = [
    'python3', '-m', 'monoloco.run',
    'predict',
    'docs/frame0032.jpg',
    '--social_distance',
    '--output_types', 'front', 'bird',
]


def test_train(tmp_path):
    # train a model
    train_cmd = TRAIN_COMMAND + ['--out={}'.format(os.path.join(tmp_path, 'train_test.pkl'))]
    print(' '.join(train_cmd))
    subprocess.run(train_cmd, check=True, capture_output=True)
    print(os.listdir(tmp_path))

    # find the trained model checkpoint
    final_model = next(iter(f for f in os.listdir(tmp_path) if f.endswith('.pkl')))

    # run predictions with that model
    model = os.path.join(tmp_path, final_model)
    print(model)
    predict_cmd = PREDICT_COMMAND + [
        '--model={}'.format(model),
        '-o={}'.format(tmp_path),
    ]
    print(' '.join(predict_cmd))
    subprocess.run(predict_cmd, check=True, capture_output=True)
    print(os.listdir(tmp_path))
    assert 'out_002282.png.multi.png' in os.listdir(tmp_path)
    assert 'out_002282.png.monoloco.json' in os.listdir(tmp_path)

    predict_cmd_sd = PREDICT_COMMAND_SOCIAL_DISTANCE + [
        '--model={}'.format(model),
        '-o={}'.format(tmp_path),
    ]
    print(' '.join(predict_cmd_sd))
    subprocess.run(predict_cmd_sd, check=True, capture_output=True)
    print(os.listdir(tmp_path))
    assert 'out_frame0032.jpg.front.png' in os.listdir(tmp_path)
    assert 'out_frame0032.jpg.bird.png' in os.listdir(tmp_path)
