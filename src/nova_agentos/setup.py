from glob import glob
from pathlib import Path
from setuptools import find_packages, setup

package_name = "nova_agentos"

# skills/<name>/{SKILL.yaml,SKILL.md} 必须按子目录安装,SkillStore 依赖该结构
_skill_dirs = sorted(d for d in Path("skills").iterdir() if d.is_dir())
skill_files = [(f"share/{package_name}/skills/{d.name}", glob(f"skills/{d.name}/*")) for d in _skill_dirs]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        *skill_files,
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="ginger",
    maintainer_email="ginger@example.com",
    description="NovaAgent core: skill injection, LLM planning to DAG, tool execution.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "nova_agentos_node = nova_agentos.agentos_node:main",
        ],
    },
)
