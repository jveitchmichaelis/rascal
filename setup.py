from setuptools import setup, find_packages

install_requires=['scipy', 'numpy>=1.16', 'tqdm', 'matplotlib', 'astropy', 'pynverse', 'pytest']

__packagename__ = "rascal"

setup(
    name=__packagename__,
    version='0.1.0',
    packages=find_packages(),
    author='Josh Veitch-Michaelis',
    author_email='j.veitchmichaelis@gmail.com',
    license='GPL',
    long_description=open('README.md').read(),
    zip_safe=False,
    include_package_data=True,
    install_requires = install_requires
)
