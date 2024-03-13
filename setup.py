from setuptools import setup, find_packages

setup(
    name='labelmergeandsplit',
    version='1.0.0',
    url='https://github.com/aaronkujawa/label_merge_and_split.git',
    author='Aaron Kujawa',
    author_email='aaron.kujawa@kcl.ac.uk',
    description='Merge and split labels based on greedy graph coloring',
    packages=['labelmergeandsplit'],
    install_requires=['torch', 'monai', 'networkx', 'numpy', 'nibabel', 'pandas', 'scipy', 'tqdm', 'natsort',
                      'matplotlib']
)