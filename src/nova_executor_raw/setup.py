from setuptools import find_packages, setup

package_name = "nova_executor_raw"
setup(name=package_name, version="0.1.0", packages=find_packages(),
      data_files=[("share/ament_index/resource_index/packages", [f"resource/{package_name}"]), (f"share/{package_name}", ["package.xml"])],
      install_requires=["setuptools"], zip_safe=True,
      entry_points={"console_scripts": ["nova_executor_raw_node = nova_executor_raw.executor_raw_node:main", "eef_test_client = nova_executor_raw.eef_test_client:main"]})
