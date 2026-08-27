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
```
- 运行 server：
```
python pi0_server.py --checkpoint /path/to/checkpoint_dir --model pi05_libero --host 0.0.0.0 --port 8767
```