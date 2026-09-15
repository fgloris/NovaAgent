from glob import glob
from setuptools import find_packages, setup
package_name = "nova_robot_description"
setup(name=package_name, version="0.1.0", packages=find_packages(),
      data_files=[("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
                  (f"share/{package_name}", ["package.xml"]),
                  (f"share/{package_name}/descriptions/panda_omron", glob("descriptions/panda_omron/*")),
                  (f"share/{package_name}/config", glob("config/*.yaml"))],
      install_requires=["setuptools", "PyYAML"], zip_safe=True,
      entry_points={"console_scripts": ["robot_context = nova_robot_description.context:main"]})
