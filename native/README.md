# Native ASR Worker (C++ / sherpa-onnx C API)

路由器量产阶段的 **ASR 子进程** 原生实现，替代 Python `asr_worker.py`，避免在识别瞬间再拉起一套 Python + sherpa 绑定。

部署集成与内存收益摘要见 **[docs/03-部署手册.md](../docs/03-部署手册.md)**。

## 角色

```
voice-routerd (Python 主守护，KWS + NLU)
       │
       │ fork/exec
       ▼
voice-router-asr-worker (本二进制，识别完即退出)
```

## 构建（开发机验证）

### 1. 编译 sherpa-onnx（含 C API）

```bash
git clone https://github.com/k2-fsa/sherpa-onnx.git
cd sherpa-onnx
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DCMAKE_INSTALL_PREFIX=$PWD/install
cmake --build build -j
cmake --install build
export SHERPA_ONNX_ROOT=$PWD/install
```

### 2. 编译 ASR worker

```bash
cd voice-router-agent-offline
bash native/build.sh
# 产物: native/bin/voice-router-asr-worker
```

### 3. 测试

```bash
# 用项目自带测试 wav（需先从 ASR 模型包解压）
ffmpeg -i input.wav -ar 16000 -ac 1 -f s16le - | \
  native/bin/voice-router-asr-worker --models-dir models | cat
```

## 交叉编译（OpenWrt MIPS）

1. 在 OpenWrt SDK 环境中交叉编译 sherpa-onnx（MIPS musl，关闭 GPU）
2. 设置环境变量后执行：

```bash
export OPENWRT_SDK=/path/to/sdk
export STAGING_DIR=$OPENWRT_SDK/staging_dir/target-mipsel_24kc_musl
export SHERPA_ONNX_ROOT=$STAGING_DIR/usr  # 按实际 sherpa 安装路径调整
bash native/build-mips-openwrt.sh
```

## 与 Python 守护进程集成

部署到路由器后设置：

```bash
export VOICE_ROUTER_ASR_WORKER=/opt/voice_router/bin/voice-router-asr-worker
```

`voice_router_lite.router.asr_subprocess` 会优先调用该二进制，否则回退到 Python worker。

## 内存收益（估算）

| 组件 | Python 子进程 | C++ worker |
|------|--------------|------------|
| 解释器 + 库 | ~30–40 MB | 0 |
| ASR 模型运行时 | ~25 MB | ~25 MB |
| **合计** | **~55–65 MB** | **~25 MB** |

峰值从 ~89MB 可降至 **~65MB**（仍含 Python 主进程 + KWS + NLU）。
