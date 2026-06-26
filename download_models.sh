#!/bin/bash
# 下载语音识别和唤醒词预训练模型
# ASR: sherpa-onnx streaming zipformer zh 14M (~25MB，适合 128MB 路由器)
# KWS: sherpa-onnx KWS zipformer 3.3M (~5MB)
# 执行: bash download_models.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  下载 ASR + KWS 预训练模型"
echo "  ASR: GitHub k2-fsa/sherpa-onnx (zh 14M mobile)"
echo "  KWS: ModelScope pkufool/sherpa-onnx-kws"
echo "============================================"

mkdir -p models/kws models/asr .cache/models

python3 << 'PYEOF'
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = os.getcwd()
CACHE = os.path.join(ROOT, ".cache", "models")


def download_file(url, dst_path, desc=""):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        print(f"  ⏭️  跳过 (已存在): {os.path.basename(dst_path)}")
        return dst_path
    print(f"  下载: {desc or url} ... ", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dst_path)
        size_mb = os.path.getsize(dst_path) / 1024 / 1024
        print(f"✅ {size_mb:.1f} MB")
    except Exception as exc:
        print(f"❌ 失败: {exc}")
        sys.exit(1)
    return dst_path


def pick_first(existing_dir, prefixes, prefer_int8=True):
    """按前缀匹配 ONNX 文件，优先 int8 量化版本。"""
    if not os.path.isdir(existing_dir):
        return None
    candidates = [
        os.path.join(existing_dir, name)
        for name in sorted(os.listdir(existing_dir))
        if name.endswith(".onnx")
    ]
    for prefix in prefixes:
        matched = [p for p in candidates if os.path.basename(p).startswith(prefix)]
        if not matched:
            continue
        if prefer_int8:
            int8 = [p for p in matched if ".int8." in os.path.basename(p) or os.path.basename(p).endswith(".int8.onnx")]
            if int8:
                return int8[0]
        return matched[0]
    return None


def copy_model_file(src, dst):
    if src is None or not os.path.isfile(src):
        raise FileNotFoundError(f"缺少模型文件: {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copy2(src, dst)
    size_mb = os.path.getsize(dst) / 1024 / 1024
    print(f"    → {os.path.basename(dst):<12s} {size_mb:>7.2f} MB")


def download_from_modelscope(model_id, target_dir, file_map):
    base = f"https://modelscope.cn/models/{model_id}/resolve/master"
    for remote_name, local_name in file_map:
        url = f"{base}/{remote_name}"
        download_file(url, os.path.join(target_dir, local_name), remote_name)


# ==========================================
# 1. KWS 唤醒词模型 (~5MB)
# ==========================================
print("\n[1/2] KWS 唤醒词模型 (3.3M 参数, 中文)...")
download_from_modelscope(
    "pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
    "models/kws",
    [
        ("tokens.txt", "tokens.txt"),
        ("encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx", "encoder.onnx"),
        ("decoder-epoch-12-avg-2-chunk-16-left-64.onnx", "decoder.onnx"),
        ("joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx", "joiner.onnx"),
        ("keywords.txt", "keywords.txt"),
    ],
)

# ==========================================
# 2. ASR 语音识别模型 (~25MB, zh 14M mobile)
# ==========================================
print("\n[2/2] ASR 语音识别模型 (streaming zipformer zh 14M mobile)...")
asr_tar = os.path.join(
    CACHE,
    "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23-mobile.tar.bz2",
)
asr_url = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23-mobile.tar.bz2"
)
download_file(asr_url, asr_tar, "ASR tarball")

extract_dir = os.path.join(CACHE, "asr-zh-14M-mobile")
if os.path.isdir(extract_dir):
    shutil.rmtree(extract_dir)
os.makedirs(extract_dir, exist_ok=True)
print("  解压 ASR 模型包...")
with tarfile.open(asr_tar, "r:bz2") as tar:
    tar.extractall(extract_dir)

# 找到解压后的根目录
model_root = extract_dir
for name in os.listdir(extract_dir):
    candidate = os.path.join(extract_dir, name)
    if os.path.isdir(candidate) and os.path.isfile(
        os.path.join(candidate, "tokens.txt")
    ):
        model_root = candidate
        break

encoder = pick_first(model_root, ["encoder-"])
decoder = pick_first(model_root, ["decoder-"], prefer_int8=False)
joiner = pick_first(model_root, ["joiner-"])
tokens = os.path.join(model_root, "tokens.txt")

print("  复制到 models/asr/ ...")
copy_model_file(encoder, "models/asr/encoder.onnx")
copy_model_file(decoder, "models/asr/decoder.onnx")
copy_model_file(joiner, "models/asr/joiner.onnx")
copy_model_file(tokens, "models/asr/tokens.txt")

# ==========================================
# 汇总
# ==========================================
print("\n" + "=" * 50)
print("  模型文件汇总")
print("=" * 50)
for d in ["models/kws", "models/asr", "models/nlu"]:
    if os.path.isdir(d):
        files = [
            f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))
        ]
        total = sum(os.path.getsize(os.path.join(d, f)) for f in files)
        print(f"\n{d}: {total/1024/1024:.1f} MB")
        for f in sorted(files):
            p = os.path.join(d, f)
            sz = os.path.getsize(p)
            if sz > 1024 * 1024:
                print(f"  {f:<30s} {sz/1024/1024:>8.1f} MB")
            else:
                print(f"  {f:<30s} {sz/1024:>8.1f} KB")

print("\n✅ 完成！")
print("提示: NLU 模型请运行  python -m voice_router_lite.nlu.train_nlu")
PYEOF
