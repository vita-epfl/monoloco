
from setuptools import setup

# This is needed for versioneer to be importable when building with PEP 517.
# See <https://github.com/warner/python-versioneer/issues/193> and links
# therein for more information.

import os, sys
sys.path.append(os.path.dirname(__file__))
import versioneer

setup(
    name='monoloco',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=[
        'monoloco',
        'monoloco.network',
        'monoloco.eval',
        'monoloco.train',
        'monoloco.prep',
        'monoloco.visuals',
        'monoloco.utils'
    ],
    license='GNU AGPLv3',
    description=' A 3D vision library from 2D keypoints',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Lorenzo Bertoni',
    author_email='lorenzo.bertoni@epfl.ch',
    url='https://github.com/vita-epfl/monoloco',
    zip_safe=False,

    install_requires=[
        'openpifpaf>=v0.12.1',
        'matplotlib',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
            'gdown',
            'scipy',  # for social distancing gaussian blur
        ],
        'eval': [
            'tabulate',
            'sklearn',
            'pandas',
        ],
        'prep': [
            'nuscenes-devkit==1.0.2',
        ],
    },
)
