import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ocean-gases-ashao", # Replace with your own username
    version="1.0.0",
    author="Andrew E. Shao",
    author_email="aeshao@gmail.com",
    description="Do some basic operations with CFC measurements used in oceanography",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="github.com/ashao/ocean_gases.git",
    packages=setuptools.find_packages(),
    package_data={'ocean_gases':['CFC_atmospheric_histories_revised_2015_Table1.csv']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)


