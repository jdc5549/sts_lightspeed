"""
Minimal setup.py for sts_lightspeed to work as an editable pip package.

The actual build is handled by CMake. For editable installs, this creates
a .pth file that adds the build directory to sys.path.
"""
from setuptools import setup
from setuptools.command.develop import develop
import os
import sys


class DevelopCommand(develop):
    """Custom develop command that creates a .pth file"""
    def run(self):
        # Check if the module has been built
        build_dir = os.path.join(os.path.dirname(__file__), 'build')
        if not os.path.exists(build_dir):
            print("\n" + "="*70)
            print("WARNING: build/ directory not found!")
            print("Please build sts_lightspeed before installing:")
            print("  cmake -S . -B build -DPYTHON_EXECUTABLE=$(which python)")
            print("  cmake --build build --target slaythespire -j4")
            print("="*70 + "\n")

        # Run the standard develop command
        develop.run(self)

        # Create a .pth file to add build/ to the path
        install_dir = self.install_dir
        pth_file = os.path.join(install_dir, 'sts_lightspeed.pth')
        with open(pth_file, 'w') as f:
            f.write(os.path.abspath(build_dir) + '\n')
        print(f"Created {pth_file} pointing to {build_dir}")


setup(
    name='sts_lightspeed',
    version='0.1.0',
    description='High Performance, RNG accurate implementation of Slay the Spire',
    author='gamerpuppy',
    url='https://github.com/gamerpuppy/sts_lightspeed',

    # No packages - this is just a C extension
    packages=[],

    cmdclass={'develop': DevelopCommand},
    python_requires='>=3.8',
    zip_safe=False,
)
