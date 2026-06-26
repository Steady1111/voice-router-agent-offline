"""诊断 KWS 模型：验证模型加载、token 匹配、解码流程。"""
import sys
import os

# 确保可以通过 sherpa-onnx API 调试
try:
    import sherpa_onnx
except ImportError:
    print("ERROR: sherpa-onnx 未安装")
    sys.exit(1)

import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "kws")
SAMPLE_RATE = 16000

# 1. 检查模型文件
print("=" * 60)
print("1. 模型文件检查")
files = ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt", "keywords.txt"]
for f in files:
    path = os.path.join(MODEL_DIR, f)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    status = "✅" if os.path.exists(path) else "❌"
    print(f"  {status} {f}: {size/1024:.1f} KB")

# 2. 读取 token 表
print()
print("=" * 60)
print("2. Token 表")
tokens_path = os.path.join(MODEL_DIR, "tokens.txt")
with open(tokens_path) as f:
    tokens = [l.strip() for l in f]
print(f"  Token 总数: {len(tokens)}")
# 检查关键词里的 token 是否在表中
kw_path = os.path.join(MODEL_DIR, "keywords.txt")
with open(kw_path) as f:
    kw_lines = [l.strip() for l in f if l.strip()]

token_to_id = {}
for line in tokens:
    parts = line.rsplit(" ", 1)
    if len(parts) == 2:
        token_to_id[parts[0]] = int(parts[1])

print()
print("3. 关键词 Token 匹配检查")
for kw_line in kw_lines:
    kw_tokens = kw_line.split(" @")[0].split()
    found = all(t in token_to_id for t in kw_tokens if t)
    token_ids = [str(token_to_id[t]) for t in kw_tokens if t in token_to_id]
    status = "✅" if found else "❌ 缺失"
    keyword = kw_line.split("@")[-1] if "@" in kw_line else "???"
    print(f"  {status} {keyword}: {kw_tokens[:5]}... → ids=[{','.join(token_ids[:5])}...]")

# 3. 加载 KeywordSpotter
print()
print("=" * 60)
print("4. 加载 KeywordSpotter")
try:
    kws = sherpa_onnx.KeywordSpotter(
        tokens=os.path.join(MODEL_DIR, "tokens.txt"),
        encoder=os.path.join(MODEL_DIR, "encoder.onnx"),
        decoder=os.path.join(MODEL_DIR, "decoder.onnx"),
        joiner=os.path.join(MODEL_DIR, "joiner.onnx"),
        keywords_file=os.path.join(MODEL_DIR, "keywords.txt"),
        num_threads=1,
        sample_rate=SAMPLE_RATE,
        max_active_paths=8,
        keywords_threshold=0.0,
        num_trailing_blanks=4,
    )
    print("  ✅ KeywordSpotter 加载成功")
except Exception as e:
    print(f"  ❌ 加载失败: {e}")
    sys.exit(1)

# 4. 测试：喂入白噪声 + 尾随静音
print()
print("=" * 60)
print("5. 解码测试（白噪声）")
noise = np.random.randn(SAMPLE_RATE * 2).astype(np.float32) * 0.01  # 2秒低能量白噪声
tail = np.zeros(int(0.66 * SAMPLE_RATE), dtype=np.float32)

stream = kws.create_stream()
stream.accept_waveform(SAMPLE_RATE, noise)
stream.accept_waveform(SAMPLE_RATE, tail)
stream.input_finished()

decode_count = 0
result_text = ""
while kws.is_ready(stream):
    kws.decode_stream(stream)
    decode_count += 1
    r = kws.get_result(stream)
    print(f"  decode #{decode_count}: get_result()={repr(r)} type={type(r).__name__}")
    if r and isinstance(r, str) and r.strip():
        result_text = r.strip()

print(f"  总解码次数: {decode_count}")
print(f"  最终结果: '{result_text}'")

# 5. 测试高能量随机音频 + 尾随静音
print()
print("=" * 60)
print("6. 解码测试（高能量随机音频）")
noise2 = np.random.randn(SAMPLE_RATE * 2).astype(np.float32) * 0.5  # 高能量

stream2 = kws.create_stream()
stream2.accept_waveform(SAMPLE_RATE, noise2)
stream2.accept_waveform(SAMPLE_RATE, tail)
stream2.input_finished()

decode_count2 = 0
result_text2 = ""
while kws.is_ready(stream2):
    kws.decode_stream(stream2)
    decode_count2 += 1
    r = kws.get_result(stream2)
    if r and isinstance(r, str) and r.strip():
        result_text2 = r.strip()
        print(f"  decode #{decode_count2}: MATCH! result={repr(r)}")

print(f"  总解码次数: {decode_count2}")
print(f"  最终结果: '{result_text2}'")

print()
print("=" * 60)
print("诊断完成")
