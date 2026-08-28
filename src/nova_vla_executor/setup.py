from glob import glob
from setuptools import find_packages, setup

package_name = "nova_vla_executor"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "requests", "websocket-client"],
    zip_safe=True,
    maintainer="ginger",
    maintainer_email="ginger@example.com",
    description="NovaAgent VLA executor bridge: 本地 ROS2 连接远程 pi0 推理服务。",
    license="MIT",
    entry_points={
        "console_scripts": [
            "nova_vla_executor_node = nova_vla_executor.vla_executor_node:main",
            "nova_pi0_server = nova_vla_executor.pi0_server:main",
        ],
    },
)
