from setuptools import setup

setup(
    name='liquid_helium_svp',
    version='24.9',
    description='Python package describing liquid helium-4 properties at saturated vapour pressure',
    url='https://github.com/tsepelin/liquid-helium-svp/tree/main/liquid_helium_svp',
    authors=['Viktor Tsepelin',
            'Theo Noble',
            'Erik Tsepelin',
            ],
    author_email='v.tsepelin@lancaster.ac.uk',
    license='MIT',
    packages=['liquid_helium_svp' ],
    install_requires=['numpy',
                      'scipy',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: MIT License',
        'Operating System :: Any',
        'Programming Language :: Python :: 3.9',
    ],
)