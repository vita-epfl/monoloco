
from setuptools import setup
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
        'versioneer',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
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
