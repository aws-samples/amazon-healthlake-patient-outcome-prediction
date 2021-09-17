from setuptools import setup, find_packages

setup(name='PatientOutcomePrediction',
      version='1.0',
      description='Patient Outcome Prediction',
      author='shuaicao',
      author_email='shuaicao@amazon.com',
      packages=find_packages(exclude=('tests', 'docs')))