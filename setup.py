from setuptools import setup, find_packages

version = __import__('eemeter').get_version()


setup(
    name='eemeter',
    version=version,
    description='Open Energy Efficiency Meter',
    long_description=(
        "Standard methods for calculating energy efficiency savings."
    ),
    url='https://github.com/impactlab/eemeter/',
    author='Matt Gee, Phil Ngo, Brandon Willard',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    keywords='open energy efficiency meter',
    packages=find_packages(),
    setup_requires=['numpy', 'pytest-runner'],
    install_requires=[
        'holidays',
        'lxml',
        'numpy',
        'pandas >= 0.18',
        'patsy',
        'pytz',
        'requests',
        'scipy',
        'scikit-learn',
        'amimodels',
    ],
    dependency_links=['http://github.com/openeemeter/amimodels/tarball/master#egg=amimodels',
                      ],
    package_data={'': ['*.json', '*.gz']},
    tests_require=['pytest'],
)
