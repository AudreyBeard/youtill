import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="youtill",
    version="0.0.1",
    author="Audrey Beard",
    author_email="audrey.s.beard@gmail.com",
    description="A collection of useful utilities to help you till cleaner, simpler code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joshuabeard/youtill",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU GPL v3",
        "Operating System :: OS Independent",
    ],
)
