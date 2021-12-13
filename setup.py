import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="linear_transforms",
    version="1.0.0.post2",
    author="Antonio Lopez Rivera",
    author_email="antonlopezr99@gmail.com",
    description="Python coordinate frame transform library",
    long_description_content_type="text/markdown",
    url="https://github.com/alopezrivera/transforms",
    py_modules=["transforms"],
    install_requires=[
        "sympy",
        "numpy",
        "Python-Alexandria>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
