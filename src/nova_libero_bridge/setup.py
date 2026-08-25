from glob import glob
from setuptools import find_packages, setup

package_name = "nova_libero_bridge"

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
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ginger",
    maintainer_email="ginger@example.com",
    description="ROS 2 bridge for controlling LIBERO simulation environments.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "libero_bridge_node = nova_libero_bridge.libero_bridge_node:main",
            "random_action_client = nova_libero_bridge.random_action_client:main",
        ],
    },
)
