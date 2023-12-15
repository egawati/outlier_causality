import codecs
from setuptools import setup, find_packages

entry_points = {
    'console_scripts': [
    ]
}


def _read(fname):
  with codecs.open(fname, encoding='utf-8') as f:
    return f.read()


setup(
    name='EasyRCA',
    version=_read('version.txt').strip(),
    author='C. K. Assaad',
    description="Root causes analysis",
    license='Apache',
    keywords='Base',
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    url="https://github.com/ckassaad/EasyRCA",
    zip_safe=True,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    namespace_packages=['EasyRCA'],
    install_requires=[
        'setuptools',
        'numpy',
        'pandas',

    ],
    entry_points=entry_points
)
