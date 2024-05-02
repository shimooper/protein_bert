from setuptools import setup

setup(
    name = 'protein-bert',
    version = '1.0.1',
    description = 'A BERT-like deep language model for protein sequences.',
    url = 'https://github.com/nadavbra/protein_bert',
    author = 'Nadav Brandes',
    author_email  ='nadav.brandes@mail.huji.ac.il',
    packages = ['proteinbert', 'proteinbert.shared_utils'],
    license = 'MIT',
    scripts = [
        'bin/create_uniref_db',
        'bin/create_uniref_h5_dataset',
        'bin/pretrain_proteinbert',
        'bin/set_h5_testset',
    ],
    install_requires = [
        'tensorflow==2.4.0',
        'tensorflow_addons==0.12.1',
        'numpy==1.19.5',
        'pandas==1.2.3',
        'h5py==2.10.0',
        'lxml==5.2.1',
        'pyfaidx==0.5.8',
    ],
)