from setuptools import setup, find_packages

install_requires=['scipy>=1.3.3', 'numpy>=1.16', 'tqdm>=4.48', 'matplotlib>=3.0.3', 'astropy>=4.0', 'pynverse>=0.1.4', 'pytest>=5.3', 'pytest-cov>=2.8']

__packagename__ = "rascal"

setup(
    name=__packagename__,
    version='0.1.0',
    packages=find_packages(),
    author='Josh Veitch-Michaelis',
    author_email='j.veitchmichaelis@gmail.com',
    license='BSD',
    long_description=open('README.md').read(),
    zip_safe=False,
    include_package_data=True,
    install_requires = install_requires,
    python_requires='>=3.6'
)
