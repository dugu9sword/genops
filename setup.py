from setuptools import setup, find_packages

setup(
    name="genops",
    version="0.1",
    keywords=["genops", ],
    description="eds sdk",
    long_description="My tool for research",
    license="MIT Licence",

    url="https://github.com/dugu9sword/genops",
    author="dugu9sword",
    author_email="dugu9sword@163.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=open("requirements.txt").readlines(),
    zip_safe=False,

    scripts=[],
    entry_points={}
)