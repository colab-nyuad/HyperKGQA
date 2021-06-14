from setuptools import setup

setup(
    name="libkge",
    version="0.1",
    description="A knowledge graph embedding library",
    author="Universität Mannheim",
    author_email="rgemulla@uni-mannheim.de",
    packages=["kge"],
    install_requires=[
        "pyyaml",
        "pandas",
        "argparse",
        "path",
        "transformers",
        "sentencepiece",
        "networkx",
        # please check correct behaviour when updating ax platform version!!
        "ax-platform==0.1.19",
        "sqlalchemy",
        "numba==0.53.0"
    ],
    python_requires=">=3.7,<3.9",
    zip_safe=False,
    entry_points={"console_scripts": ["kge = kge.cli:main",],},
)
