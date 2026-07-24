# OpenWrt 路由器部署

将本目录内容安装到路由器 `/opt/voice_router/`。

完整步骤、双进程与环境变量见 **[docs/03-部署手册.md](../../docs/03-部署手册.md)**；方案背景见 **[docs/01-方案总览.md](../../docs/01-方案总览.md)**。

## 安装步骤

```bash
# 开发机
bash download_models.sh
python3 -m voice_router_lite.nlu.train_nlu   # 生成 fp32 + INT8 NLU

# 打包
tar czf voice_router.tar.gz voice_router_lite/ models/ audio_clips/ deploy/

# 路由器
scp voice_router.tar.gz root@192.168.1.1:/tmp/
ssh root@192.168.1.1
cd /opt && mkdir -p voice_router && tar xzf /tmp/voice_router.tar.gz -C voice_router
pip3 install -e /opt/voice_router  # 或预装 wheel

# procd 服务
cp /opt/voice_router/deploy/openwrt/voice-router.init /etc/init.d/voice-router
cp /opt/voice_router/deploy/openwrt/voice-router.json /opt/voice_router/deploy/openwrt/
/etc/init.d/voice-router enable
/etc/init.d/voice-router start
```

## 原生 ASR worker（推荐）

交叉编译 C++ 二进制可省去 Python ASR 子进程的 ~30MB 开销：

```bash
# 开发机：见 native/README.md
bash native/build-mips-openwrt.sh
scp native/bin/voice-router-asr-worker-mips root@router:/opt/voice_router/bin/voice-router-asr-worker
```

在 `voice-router.json` 同目录或 init 脚本中设置：

```bash
export VOICE_ROUTER_ASR_WORKER=/opt/voice_router/bin/voice-router-asr-worker
```

未设置时自动回退到 Python `asr_worker` 模块。

## 架构（128MB 优化）

```
USB-I2S → ALSA → voice-routerd (python -m voice_router_lite router)
                    ├── KWS 常驻（待机 ~8MB）
                    ├── 唤醒后卸载 KWS
                    ├── ASR 子进程（C++ 优先，识别完退出）
                    ├── NLU INT8 常驻（~4MB）
                    └── ubus → network / system / GPIO
```

## 内存目标

| 阶段 | Python ASR 子进程 | C++ ASR worker |
|------|------------------|----------------|
| 待机 | ~72MB | ~72MB |
| 峰值 | ~89MB | ~**65MB** |

ASR 在独立子进程中运行，退出后由内核完全回收，不与 KWS 同驻。

## 验证

```bash
logread -f | grep voice
ubus call voice-router status   # 若后续扩展 ubus 对象
python3 -m voice_router_lite router --text "打开风扇"
```
