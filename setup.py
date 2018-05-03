from setuptools import setup

setup(name = "sta_663_project",
      version = "1.0",
      author='Chen White',
      author_email='zach.m.white@duke.edu',
      url='https://github.com/zachmwhite/sta_663_project',
      py_modules = ['main_optimized'],
      packages=setuptools.find_packages(),
      scripts = ['main_optimized.py'],
      python_requires='>=3',
      )