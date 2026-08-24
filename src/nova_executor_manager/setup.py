from setuptools import find_packages, setup

package_name = "nova_executor_manager"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ginger",
    maintainer_email="ginger@example.com",
    description="NovaAgent executor manager (heartbeat registry + MCPExecute router).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "nova_executor_manager_node = nova_executor_manager.executor_manager_node:main",
        ],
    },
)
