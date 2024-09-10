from setuptools import setup

setup(
    name='LHE',
    version='0.1.0',
    description='A example Python package',
    url='github link',
    author='Viktor Tsepelin',
    author_email='v.tsepelin@lancaster.ac.uk',
    license='MIT',
    packages=['Liquid_helium' ],
    install_requires=['mpi4py>=2.0',
                      'numpy',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: MIT License',
        'Operating System :: Any',
        'Programming Language :: Python :: 3.9',
    ],
)