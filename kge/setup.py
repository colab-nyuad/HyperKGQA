from setuptools import setup

setup(
    name="libkge",
    version="0.1",
    description="A knowledge graph embedding library",
    packages=["kge"],
    install_requires=[
        "torch",
        "pyyaml",
        "networkx",
        "pandas",
        "argparse",
        "path",
        "transformers",
        "sentencepiece",
        # please check correct behaviour when updating ax platform version!!
        "ax-platform==0.1.19",
        "sqlalchemy",
        "numba==0.51.*",
    ],
    python_requires=">=3.7,<3.9",
    zip_safe=False,
    entry_points={"console_scripts": ["kge = kge.cli:main",],},
)
