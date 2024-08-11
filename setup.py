import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = '0.0.0'

REPO_NAME = "END-TO-END-MLFLOW-PROJECT"
AUTHOR_NAME = "Ntchinda-Giscard"
SRC_REPO = "recomProject"
AUTHOR_EMAIL = "ntchinda1998@gmail.com"

setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for ml project",
    long_description=long_description,
    long_description_content="text/mackdown",
    url=f"https://github.com{AUTHOR_NAME}/{REPO_NAME}",
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src")
    )