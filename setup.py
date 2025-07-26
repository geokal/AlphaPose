import os
import platform
import subprocess
import time

import numpy as np
import torch
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from torch.utils.cpp_extension import BuildExtension

MAJOR = 0
MINOR = 5
PATCH = 0
SUFFIX = ''
SHORT_VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH, SUFFIX)

version_file = 'alphapose/version.py'

def readme():
    with open('README.md') as f:
        content = f.read()
    return content

def get_git_hash():
    def _minimal_ext_cmd(cmd):
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha

def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    elif os.path.exists(version_file):
        try:
            from alphapose.version import __version__
            sha = __version__.split('+')[-1]
        except ImportError:
            raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'

    return sha

def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}

__version__ = '{}'
short_version = '{}'
"""
    sha = get_hash()
    VERSION = SHORT_VERSION + '+' + sha

    with open(version_file, 'w') as f:
        f.write(content.format(time.asctime(), VERSION, SHORT_VERSION))

def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != 'Windows':
        extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings', '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION']
        }

    # Hardcode the correct PyTorch include path
    #torch_include = '/Users/georgek/.local/share/virtualenvs/AlphaPose-kzn0QKfJ/lib/python3.9/site-packages/torch/include'
    torch_include = os.path.join(os.path.dirname(torch.__file__), 'include')
    
    extension = Extension(
        '{}.{}'.format(module, name),
        [os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=[np.get_include(), torch_include],
        language='c++',
        extra_compile_args=extra_compile_args)
    extension, = cythonize(extension)
    return extension

def get_ext_modules():
    ext_modules = [
        make_cython_ext(
            name='soft_nms_cpu',
            module='detector.nms',
            sources=['src/soft_nms_cpu.pyx']),
        # The nms_cpu.cpp extension has compatibility issues with PyTorch 2.0.1
        # make_cython_ext(
        #     name='nms_cpu',
        #     module='detector.nms',
        #     sources=['src/nms_cpu.cpp']),
    ]
    return ext_modules

def get_install_requires():
    install_requires = [
        'six', 'terminaltables', 'scipy>=1.9.3',
        'opencv-python>=4.8.0', 'matplotlib>=3.7.2', 'visdom',
        'tqdm>=4.66.1', 'tensorboardx', 'easydict',
        'pyyaml>=6.0', 'halpecocotools',
        'torch>=2.0.0', 'torchvision>=0.15.0', 'torchaudio>=2.0.0',
        'munkres>=1.1.4', 'timm==0.1.20', 'natsort'
    ]
    if platform.system() != 'Windows':
        install_requires.append('pycocotools>=2.0.7')
    return install_requires

def is_installed(package_name):
    import pkg_resources
    for p in pkg_resources.working_set:
        if package_name in p.egg_name():
            return True
    return False

if __name__ == '__main__':
    write_version_py()
    setup(
        name='alphapose',
        version=get_version(),
        description='Code for AlphaPose',
        long_description=readme(),
        keywords='computer vision, human pose estimation',
        url='https://github.com/MVIG-SJTU/AlphaPose',
        packages=find_packages(exclude=('data', 'exp',)),
        package_data={'': ['*.json', '*.txt']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        license='GPLv3',
        python_requires=">=3.6",
        setup_requires=['pytest-runner', 'numpy>=1.23.5', 'cython>=0.29.32'],
        tests_require=['pytest'],
        install_requires=get_install_requires(),
        ext_modules=get_ext_modules(),
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
    if platform.system() == 'Windows' and not is_installed('pycocotools'):
        print("\nInstall third-party pycocotools for Windows...")
        cmd = 'python -m pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI'
        os.system(cmd)
    if not is_installed('cython_bbox'):
        print("\nInstall `cython_bbox`...")
        cmd = 'python -m pip install git+https://github.com/yanfengliu/cython_bbox.git'
        os.system(cmd)