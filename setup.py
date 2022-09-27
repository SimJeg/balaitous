from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
  long_description = f.read()

setup(name='balaitous',
      author='Simon Jégou',
      version='1.0',
      description='Codebase to run the Balaitous model',
      long_description=long_description,
      packages=find_packages(),
      license ='MIT',
      install_requires=[
          'numpy',
          'SimpleITK',
          'scikit-image',
          'torch',
          'torchvision',
          'lungmask @ git+https://git@github.com/JoHof/lungmask@master',
      ],
       entry_points ={'console_scripts': ['balaitous = balaitous.cli:cli']})
