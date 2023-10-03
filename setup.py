from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(name='balaitous',
      author='Simon Jégou',
      version='1.1',
      description='Codebase to run the Balaitous model',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(),
      license ='MIT',
      include_package_data=True,
      install_requires=[
          'numpy',
          'SimpleITK',
          'scikit-image',
          'torch',
          'torchvision',
          'lungmask',
      ],
       entry_points ={'console_scripts': ['balaitous = balaitous.cli:cli']})
