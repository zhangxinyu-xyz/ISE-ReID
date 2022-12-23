from setuptools import setup, find_packages


setup(name='ISE',
      version='Pytorch-1.0',
      description='Implicit Sample Extension for Unsupervised Person Re-Identification',
      author='Xinyu Zhang',
      author_email='zhangxinyu14@baidu.com',
      # url='',
      install_requires=[
          'numpy', 'torch==1.2.0', 'torchvision==0.4.0',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu==1.6.4'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Person Re-identification',
      ])
