 import setuptools

 setuptools.setup(
    name='m1-ispy1-processing',
    version='0.0.1',
    install_requires=[],
    packages=setuptools.find_packages(),
    package_data={
        "pipeline": ["*.csv", "*.tcia"]
    },
    author="Jack Eadie",
    author_email="@jeadie",
    description="Processes and parses import data from the ISPY datset.", 
    keywords="hello world example examples",
    url="https://github.com/Jeadie/ENGG4801/tree/master/m1_ISPY_processing",  
)
