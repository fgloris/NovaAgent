from glob import glob
from setuptools import find_packages, setup

package_name = "nova_agentos"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="ginger",
    maintainer_email="ginger@example.com",
    description="LLM-driven AgentOS for the NovoAgent robot stack.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "agentos_node = nova_agentos.agentos_node:main",
        ],
    },
)
