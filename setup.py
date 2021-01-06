import os
from distutils.core import setup
from setuptools import find_packages

pkg_dir = os.path.dirname(__name__)

with open(os.path.join(pkg_dir, 'requirements.txt')) as fd:
    requirements = [req.strip() for req in fd.read().splitlines()
                    if not req.strip().startswith('#')]

# Extra dependencies that cannot be included only in install_requires
# such as private repositories
requirements += []

setup(
    name='effnett5',
    version='0.1.0',
    packages=find_packages('.'),
    long_description=open('README.md').read(),
    install_requires=requirements,
)
