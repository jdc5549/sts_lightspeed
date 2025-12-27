"""
Minimal setup.py for sts_lightspeed to work as an editable pip package.

The actual build is handled by CMake. This just makes the built .so file
importable when installed with: pip install -e .
"""
from setuptools import setup, find_packages
from setuptools.dist import Distribution
import os


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(foo):
        return True


# Check if the module has been built
build_dir = os.path.join(os.path.dirname(__file__), 'build')
if not os.path.exists(build_dir):
    print("\n" + "="*70)
    print("WARNING: build/ directory not found!")
    print("Please build sts_lightspeed before installing:")
    print("  cmake -S . -B build -DPYTHON_EXECUTABLE=$(which python)")
    print("  cmake --build build --target slaythespire -j4")
    print("="*70 + "\n")


setup(
    name='sts_lightspeed',
    version='0.1.0',
    description='High Performance, RNG accurate implementation of Slay the Spire',
    author='gamerpuppy',
    url='https://github.com/gamerpuppy/sts_lightspeed',

    # Tell pip to include the build directory in the package
    packages=[''],
    package_dir={'': 'build'},
    package_data={'': ['*.so']},

    distclass=BinaryDistribution,
    python_requires='>=3.8',
    zip_safe=False,
)
