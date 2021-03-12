from setuptools import setup

# extract version from __init__.py
with open('monoloco/__init__.py', 'r') as f:
    VERSION_LINE = [l for l in f if l.startswith('__version__')][0]
    VERSION = VERSION_LINE.split('=')[1].strip()[1:-1]

setup(
    name='monoloco',
    version=VERSION,
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
    description=' Perceiving Humans: from Monocular 3D Localization to Social Distancing '
                '/ MonStereo: When Monocular and Stereo Meet at the Tail of 3D Human Localization',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Lorenzo Bertoni',
    author_email='lorenzo.bertoni@epfl.ch',
    url='https://github.com/vita-epfl/monoloco',
    zip_safe=False,

    install_requires=[
        'openpifpaf>=v0.12.1',
        'matplotlib'
    ],
    extras_require={
        'eval': [
            'tabulate==0.8.3',
            'sklearn',
            'pandas',
            'pylint',
            'pytest',
        ],
        'prep': [
            'nuscenes-devkit==1.0.2',
        ],
    },
)
