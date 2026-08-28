#!/usr/bin/env python3
from glob import glob
from pathlib import Path
from setuptools import find_packages, setup

package_name = "nova_console"

web_files = [(f"share/{package_name}/web", glob("nova_console/web/*"))]
config_files = [(f"share/{package_name}/config", glob("config/*.yaml"))]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        *web_files,
        *config_files,
    ],
    install_requires=["setuptools", "PyYAML", "ptyprocess", "fastapi", "uvicorn"],
    zip_safe=True,
    maintainer="ginger",
    maintainer_email="ginger@example.com",
    description="NovaAgent web console: session manager + agent chat.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "nova_console_server = nova_console.server:main",
        ],
    },
)
