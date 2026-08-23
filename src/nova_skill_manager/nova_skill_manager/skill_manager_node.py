#!/usr/bin/env python3
"""skill_manager_node: 技能注册表管理 + 内置技能执行。"""

from __future__ import annotations

import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from nova_interfaces.msg import SkillCommand, SkillInfo, SkillResult
from nova_interfaces.srv import (
    CreateSkill,
    DeleteSkill,
    GetSkill,
    ListSkills,
    LoadSkills,
    ValidateSkills,
)
from nova_skill_manager.executors import BUILTIN_EXECUTORS
from nova_skill_manager.executors.base import ExecContext
from nova_skill_manager.skills_loader import SkillsLoader
from nova_skill_manager.skills_validator import validate_directory

DEFAULT_WORKSPACE_SKILLS = Path.home() / "novaagent" / "skills"

SKILL_SCAFFOLD = """---
name: {name}
description: |
  描述该技能的作用。

  **Use this skill when:**
  - 用户明确请求该能力
  - 任务匹配该技能的接口
metadata: '{{"nova": {{"available": true, "kind": "builtin", "executor": "{module}:{class_name}"}}}}'
---

# {title}

## Workflow

1. 检查前置条件。
2. 说明行动计划。
3. 安全执行。
4. 校验结果。

## Safety Rules

- 不得虚构不存在的机器人能力。
- 危险动作执行前必须确认。
"""


class SkillManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("nova_skill_manager")

        self.declare_parameter("workspace_skills_dir", str(DEFAULT_WORKSPACE_SKILLS))
        workspace_dir = Path(str(self.get_parameter("workspace_skills_dir").value)).expanduser()

        self.loader = SkillsLoader(builtin_dir=self._builtin_skills_dir(), workspace_dir=workspace_dir)
        self.executor_pool = ThreadPoolExecutor(max_workers=2)
        self.executor_instances: dict[str, object] = {}
        self.busy = False

        self.result_pub = self.create_publisher(SkillResult, "/nova/skills/result", 10)
        self.create_subscription(SkillCommand, "/nova/skills/cmd", self.cmd_callback, 10)

        self.create_service(ListSkills, "/nova/skills/list", self.list_callback)
        self.create_service(GetSkill, "/nova/skills/get", self.get_callback)
        self.create_service(LoadSkills, "/nova/skills/load", self.load_callback)
        self.create_service(ValidateSkills, "/nova/skills/validate", self.validate_callback)
        self.create_service(CreateSkill, "/nova/skills/create", self.create_callback)
        self.create_service(DeleteSkill, "/nova/skills/delete", self.delete_callback)

        self.get_logger().info(
            f"skill registry: builtin={self.loader.builtin_dir} workspace={workspace_dir}"
        )

    def _builtin_skills_dir(self) -> Path:
        # 安装布局: share/nova_skill_manager/skills
        try:
            from ament_index_python.packages import get_package_share_directory

            share = Path(get_package_share_directory("nova_skill_manager")) / "skills"
            if share.is_dir():
                return share
        except Exception:
            pass
        # 源码布局: 包根目录/skills
        return Path(__file__).resolve().parents[1] / "skills"

    def _to_info(self, data: dict) -> SkillInfo:
        info = SkillInfo()
        info.name = data["name"]
        info.description = data["description"]
        info.available = data["available"]
        info.requires = data["requires"]
        info.source = data["source"]
        info.location = data["location"]
        return info

    def _publish_result(self, msg: SkillCommand, status: str, success: bool, info: str) -> None:
        result = SkillResult()
        result.task_id = msg.task_id
        result.skill_id = msg.skill_id
        result.status = status
        result.success = success
        result.info = info
        self.result_pub.publish(result)
        self.get_logger().info(f"skill={msg.skill_id} task={msg.task_id} -> {status}: {info}")

    # ---- 执行路径 ----
    def cmd_callback(self, msg: SkillCommand) -> None:
        if self.busy:
            self._publish_result(msg, "failed", False, "another skill is running")
            return

        record = self.loader.get(msg.skill_id)
        if record is None:
            self._publish_result(msg, "failed", False, f"unknown skill: {msg.skill_id}")
            return
        if not self.loader.is_available(msg.skill_id):
            self._publish_result(
                msg, "failed", False,
                f"skill unavailable, missing: {self.loader.missing_requirements(msg.skill_id)}",
            )
            return

        self.busy = True
        self.executor_pool.submit(self._run_skill, msg, record.name)
        self._publish_result(msg, "running", True, "started")

    def _run_skill(self, msg: SkillCommand, skill_name: str) -> None:
        try:
            params = json.loads(msg.params_json) if msg.params_json else {}
            executor = self._executor_for(skill_name)
            if executor is None:
                self._publish_result(msg, "failed", False, f"no executor for skill: {skill_name}")
                return
            result = executor.run(ExecContext(self, msg.task_id, msg.goal, params))
            status = "succeeded" if result.success else "failed"
            self._publish_result(msg, status, result.success, result.info)
        except Exception as exc:
            self.get_logger().error(f"skill {skill_name} crashed: {exc}")
            self._publish_result(msg, "failed", False, str(exc))
        finally:
            self.busy = False

    def _executor_for(self, skill_name: str):
        if skill_name in self.executor_instances:
            return self.executor_instances[skill_name]

        # 内置执行器优先,SKILL.md 的 metadata.executor 可覆盖为任意 module:Class
        executor_cls = BUILTIN_EXECUTORS.get(skill_name)
        if executor_cls is None:
            frontmatter = self.loader.frontmatter(skill_name)
            from nova_skill_manager.skills_loader import nova_meta

            executor_spec = nova_meta(frontmatter).get("executor")
            if executor_spec:
                module, _, class_name = executor_spec.partition(":")
                try:
                    module_obj = importlib.import_module(module)
                    executor_cls = getattr(module_obj, class_name, None)
                except ImportError as exc:
                    self.get_logger().error(f"cannot import executor {executor_spec}: {exc}")
        if executor_cls is None:
            return None
        instance = executor_cls(self)
        self.executor_instances[skill_name] = instance
        return instance

    # ---- 注册表服务 ----
    def list_callback(self, request, response: ListSkills.Response):
        del request
        response.skills = [self._to_info(self.loader.info(r)) for r in self.loader.list()]
        return response

    def get_callback(self, request, response: GetSkill.Response):
        record = self.loader.get(request.name)
        if record is None:
            response.found = False
            response.content = ""
            return response
        response.found = True
        response.content = record.content
        response.info = self._to_info(self.loader.info(record))
        return response

    def load_callback(self, request, response: LoadSkills.Response):
        parts = []
        for name in request.names:
            content = self.loader.load(name)
            if content:
                parts.append(f"### Skill: {name}\n\n{content}")
        response.content = "\n\n---\n\n".join(parts)
        response.success = True
        response.message = f"loaded {len(parts)}/{len(request.names)} skills"
        return response

    def validate_callback(self, request, response: ValidateSkills.Response):
        del request
        report = validate_directory(self.loader.builtin_dir, self.loader.list())
        response.valid = report.startswith("Skill validation passed.")
        response.report = report
        return response

    def create_callback(self, request, response: CreateSkill.Response):
        name = request.name.strip()
        if not name:
            response.success = False
            response.message = "skill name is empty"
            return response
        skill_dir = self.loader.workspace_dir / name
        if skill_dir.exists():
            response.success = False
            response.message = f"skill already exists: {skill_dir}"
            return response
        skill_dir.mkdir(parents=True, exist_ok=False)
        class_name = "".join(part.capitalize() for part in name.split("-")) + "Executor"
        module = f"nova_skill_manager.executors.{name.replace('-', '_')}"
        (skill_dir / "SKILL.md").write_text(
            SKILL_SCAFFOLD.format(
                name=name,
                title=name.replace("-", " ").title(),
                module=module,
                class_name=class_name,
            ),
            encoding="utf-8",
        )
        response.success = True
        response.message = f"created skill scaffold: {skill_dir}"
        self.get_logger().info(response.message)
        return response

    def delete_callback(self, request, response: DeleteSkill.Response):
        skill_dir = self.loader.workspace_dir / request.name
        if not skill_dir.exists():
            response.success = False
            response.message = f"skill not found: {skill_dir}"
            return response
        if skill_dir == self.loader.builtin_dir / request.name:
            response.success = False
            response.message = "cannot delete builtin skill"
            return response
        import shutil

        shutil.rmtree(skill_dir)
        response.success = True
        response.message = f"deleted skill: {skill_dir}"
        self.get_logger().info(response.message)
        return response


def main(args=None) -> int:
    rclpy.init(args=args)
    node = SkillManagerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.executor_pool.shutdown(wait=False)
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
