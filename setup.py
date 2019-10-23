import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="emllib",
    version="0.0.3",
    maintainer="Federico Baldo",
    maintainer_email="federico.baldo2@unibo.it",
    # author="University of Bologna - DISI",
    description="Library fro Empirical Model Learning ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/emlopt/emllib",
    packages=setuptools.find_packages(),
    # classifiers=[
    #     "Empirical Model Learning",
    #     "Machine Learning", 
    #     "Artificial Intelligence", 
    #     "Optimization Problem",
    #     "Combinatorial Problem",
    #     "Constraint Programming"
    # ],
    python_requires='>=3.6',
)