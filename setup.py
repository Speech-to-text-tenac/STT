#!/usr/bin/env python
"""Setup for the application."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['pytest==7.1.1']

test_requirements = ['pandas', 'matplotlib', 'sklearn',
                     'tensorflow', 'sql', 'pytest>=3', ]

setup(
    author="",
    email="",
    python_requires='>=3.6',
    description="Speech to text",
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords='STT, Data_Preprocessing, NLP, pytest',
    name='AB_hypothesis_testing',
    packages=find_packages(include=['src', 'src.*']),
    test_suite='Tests',
    tests_require=test_requirements,
    url='https://github.com/Speech-to-text-tenac/STT',
    version='0.1.0',
    zip_safe=False,
)
