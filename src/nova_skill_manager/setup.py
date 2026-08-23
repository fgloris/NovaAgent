from glob import glob
from pathlib import Path
from setuptools import find_packages, setup

package_name = "nova_skill_manager"

skills_files = [
    (
        str(Path("share") / package_name / "skills" / str(path.parent.relative_to("skills"))),
        [str(path)],
    )
    for path in sorted(Path("skills").rglob("*"))
    if path.is_file()
]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        *skills_files,
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="ginger",
    maintainer_email="ginger@example.com",
    description="Skill registry and runtime management for the NovoAgent robot stack.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "skill_manager_node = nova_skill_manager.skill_manager_node:main",
        ],
    },
)
