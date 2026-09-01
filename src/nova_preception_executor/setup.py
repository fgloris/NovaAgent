from glob import glob
from setuptools import find_packages, setup

package_name = "nova_preception_executor"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ginger",
    maintainer_email="ginger@example.com",
    description="NovaAgent perception executor (multi-view VLM 3D localization).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "nova_preception_executor_node = nova_preception_executor.perception_executor_node:main",
        ],
    },
)
