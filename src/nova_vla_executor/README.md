## pi0
- 权重：在https://huggingface.co/robocasa/robocasa365_checkpoints/ 下载
```
hf download robocasa/robocasa365_checkpoints --include "pi0/*" --local-dir ./robocasa365_checkpoints
```
- 环境：依赖 Python >= 3.11 + openpi 仓库:
```
uv venv --python 3.11
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi && uv sync && uv pip install -e .
uv pip install fastapi uvicorn
# 若缺 chex/pytest(uv sync 不装 dev 组,fork 可能漏列 chex):
uv pip install pytest
uv pip install --no-deps chex==0.1.90   # 必须 --no-deps,否则会把 jax 0.5.3 升到不兼容版本
```
- 运行 server：
```
python pi_server.py --checkpoint /path/to/checkpoint_dir --host 0.0.0.0 --port 8767
# --model 默认 pi0_robocasa_pretrain_human300,obs-key-map 默认 agentview=observation/image robot0_eye_in_hand=observation/wrist_image
```