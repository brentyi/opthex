from setuptools import setup, find_packages

setup(
    name="opthex",
    version="0.0",
    author="brent, philipp",
    author_email="brentyi@berkeley.edu",
    license="MIT",
    description="opthex",
    url="https://github.com/brentyi/opthex",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.7",
    install_requires=["numpy", "mujoco_py>=2.0,<2.0.2.9",],
)
