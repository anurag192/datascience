from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

requirements=[]
def get_requirements(file_path:str)->List[str]:
    with open(file_path,'r') as file:
        requirements=file.readlines()
        requirements=[req.replace('\n','') for req in requirements ]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='datascience',
    version='0.0.1',
    author='Anurag',
    author_email='anurag756singh@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)


