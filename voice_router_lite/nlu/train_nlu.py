#!/usr/bin/env python3
"""字符级 CNN+LSTM NLU 模型训练脚本（路由器核心版）。
入口：python voice_router_lite/nlu/train_nlu.py
重心：路由器运维场景占 ~65%，智能家居设备控制占 ~35%
"""

import json, os, random
LEN = 50;  BS = 32;  EP = 40;  LR = 1e-3;  DROP = 0.3

# ═══════════════════════════════════════════════════════════════════════
# 1. 数据生成（BIO 标注）—— 路由器为核心重心
# ═══════════════════════════════════════════════════════════════════════
def _b(varies, slot=None):
    return {"varies": varies, "slot": slot}

def _build(builder):
    text_parts, bio_parts = [], []
    for seg in builder:
        val = random.choice(seg["varies"]); slot = seg.get("slot")
        bio_parts.append(
            [f"B-{slot}" if i == 0 else f"I-{slot}" for i in range(len(val))]
            if slot else ["O"] * len(val))
        text_parts.append(val)
    text = "".join(text_parts); bio = [b for bp in bio_parts for b in bp]
    return text, bio

def gen_data():
    S = []

    # ═══════════════════════════════════════════════════════════════════
    # 🔵 路由器核心场景（10 个意图，占 ~65%）
    # ═══════════════════════════════════════════════════════════════════

    # 1. router_reboot —— 重启路由器
    for _ in range(70):
        t,b=_build([_b(["重启路由器","重启路由","重启网络","重新启动路由器","重启一下"])]); S.append((t,"router_reboot",b))
    for _ in range(50):
        t,b=_build([_b(["网络卡了","网卡了","路由器卡了","路由器死机了","路由器没响应了"]),_b(["重启一下","帮我重启","重启吧"])]); S.append((t,"router_reboot",b))
    for _ in range(40):
        t,b=_build([_b(["路由器"]),_b(["重启","重新启动","重启一下"],"action")]); S.append((t,"router_reboot",b))
    for _ in range(30):
        t,b=_build([_b(["系统"]),_b(["重启","重新启动"],"action")]); S.append((t,"router_reboot",b))

    # 2. router_wifi_restart —— 重启/开关 WiFi
    for _ in range(60):
        t,b=_build([_b(["重启WiFi","重启WIFI","重启无线","WiFi重启一下","WIFI重启","WiFi重开","重新打开WiFi"])]); S.append((t,"router_wifi_restart",b))
    for _ in range(50):
        t,b=_build([_b(["WiFi断了","无线断了","WiFi连不上","无线连不上","WiFi信号没了"]),_b(["重启一下","帮我重启","重连一下"])]); S.append((t,"router_wifi_restart",b))
    for _ in range(40):
        t,b=_build([_b(["无线"]),_b(["重启","重开"],"action")]); S.append((t,"router_wifi_restart",b))
    for _ in range(40):
        t,b=_build([_b(["关闭","打开","开启","关掉"],"state"),_b(["WiFi","无线"])]); S.append((t,"router_wifi_restart",b))

    # 3. router_wifi_config —— WiFi 配置管理
    for _ in range(50):
        t,b=_build([_b(["改WiFi密码","修改WiFi密码","WiFi改个密码","WiFi密码改一下","换WiFi密码"],"config_type")]); S.append((t,"router_wifi_config",b))
    for _ in range(45):
        t,b=_build([_b(["WiFi改个名字","修改WiFi名称","WiFi改名","SSID改一下","WiFi换个名字"],"config_type")]); S.append((t,"router_wifi_config",b))
    for _ in range(45):
        t,b=_build([_b(["打开访客WiFi","开启访客网络","启用访客WiFi","关闭访客WiFi","关掉访客网络","禁用访客WiFi"],"config_type")]); S.append((t,"router_wifi_config",b))
    for _ in range(30):
        t,b=_build([_b(["切换到","改成","换成"],"action"),_b(["5G频段","2.4G频段","5G WiFi","2.4G WiFi"],"config_type")]); S.append((t,"router_wifi_config",b))
    for _ in range(30):
        t,b=_build([_b(["WiFi信道","信道"],"config_type"),_b(["改成","切换到","换成"],"action"),_b(["1","6","11","自动"],"config_value")]); S.append((t,"router_wifi_config",b))
    for _ in range(30):
        t,b=_build([_b(["WiFi模式","无线模式"],"config_type"),_b(["改成","换成"],"action"),_b(["混合","N模式","AC模式","AX模式"],"config_value")]); S.append((t,"router_wifi_config",b))

    # 4. router_network_query —— 网络信息查询
    for _ in range(35):
        t,b=_build([_b(["IP地址是多少","我的IP是什么","路由器IP是多少","查看IP","上网IP是什么"],"query_type")]); S.append((t,"router_network_query",b))
    for _ in range(35):
        t,b=_build([_b(["谁连了WiFi","连了几个设备","有哪些设备连了","查看连接的设备","联网设备有哪些"],"query_type")]); S.append((t,"router_network_query",b))
    for _ in range(35):
        t,b=_build([_b(["网速怎么样","测一下网速","网速快不快","当前网速多少","宽带速度多少"],"query_type")]); S.append((t,"router_network_query",b))
    for _ in range(30):
        t,b=_build([_b(["看看","查看"],"action"),_b(["IP地址","连接设备","网速","网络状态"],"query_type")]); S.append((t,"router_network_query",b))
    for _ in range(25):
        t,b=_build([_b(["路由器"]),_b(["运行多久了","温度多少","内存剩多少","CPU负载多少"],"query_type")]); S.append((t,"router_network_query",b))

    # 5. router_led_control —— 路由器指示灯控制
    for _ in range(45):
        t,b=_build([_b(["关掉","关了","关闭","熄灭"],"state"),_b(["路由器灯","路由器LED","指示灯","路由器的灯"])]); S.append((t,"router_led_control",b))
    for _ in range(45):
        t,b=_build([_b(["打开","开启","点亮"],"state"),_b(["路由器灯","路由器LED","指示灯","路由器的灯"])]); S.append((t,"router_led_control",b))
    for _ in range(30):
        t,b=_build([_b(["路由器指示灯","路由器的灯"]),_b(["关了","关掉","打开","开启","熄灭","点亮"],"state")]); S.append((t,"router_led_control",b))

    # 6. router_device_manage —— 设备管理（踢人/拉黑/白名单）★NEW★
    for _ in range(40):
        t,b=_build([_b(["踢掉","拉黑","屏蔽","禁止上网"],"manage_action"),_b(["这个设备","这部手机","这个电视","这个电脑"],"target")]); S.append((t,"router_device_manage",b))
    for _ in range(35):
        t,b=_build([_b(["把"]),_b(["客厅电视","卧室手机","书房电脑","访客手机"],"device_name"),_b(["踢掉","拉黑","屏蔽","禁止上网"],"manage_action")]); S.append((t,"router_device_manage",b))
    for _ in range(30):
        t,b=_build([_b(["取消拉黑","解除屏蔽","恢复上网","允许联网"],"manage_action"),_b(["这个设备","这部手机","这个电视"],"target")]); S.append((t,"router_device_manage",b))
    for _ in range(25):
        t,b=_build([_b(["显示","列出","查看"],"manage_action"),_b(["已拉黑的设备","屏蔽列表","黑名单"],"target")]); S.append((t,"router_device_manage",b))

    # 7. router_network_diag —— 网络诊断 ★NEW★
    for _ in range(55):
        t,b=_build([_b(["测速","测一下网速","网络测速","测速一下","测一测网速","测个速","网速测一下",
                         "网速怎么样","网速快不快","网速测试","帮我测速","测测网络速度",
                         "测速看看","网速测一测","跑个测速","测下网速","网速是多少"],"diag_type")]); S.append((t,"router_network_diag",b))
    for _ in range(45):
        t,b=_build([_b(["延迟多少","Ping一下","延迟怎么样","测下延迟","看看延迟","延迟高不高",
                         "延迟大吗","Ping值多少","测一下延迟","网络延迟","有延迟吗"],"diag_type")]); S.append((t,"router_network_diag",b))
    for _ in range(35):
        t,b=_build([_b(["丢包率多少","有没有丢包","检查丢包","丢包严重吗","看看丢包","丢包情况"],"diag_type")]); S.append((t,"router_network_diag",b))
    for _ in range(30):
        t,b=_build([_b(["DNS正常吗","DNS解析有问题","DNS服务器是什么","DNS有没有问题","DNS地址"],"diag_type")]); S.append((t,"router_network_diag",b))
    for _ in range(30):
        t,b=_build([_b(["网络诊断","全面诊断","网络检查","一键诊断","网络体检","检查网络",
                         "网络出什么问题了","帮我看看网络","排查网络问题"],"diag_type")]); S.append((t,"router_network_diag",b))
    for _ in range(25):
        t,b=_build([_b(["信号怎么样","信号好不好","WiFi信号","信号强度","信号覆盖"],"diag_type")]); S.append((t,"router_network_diag",b))

    # 8. router_system —— 系统运维 ★NEW★
    for _ in range(40):
        t,b=_build([_b(["检查固件更新","有新固件吗","固件升级","更新系统","检查更新"],"sys_action")]); S.append((t,"router_system",b))
    for _ in range(35):
        t,b=_build([_b(["备份配置","备份设置","导出配置","保存当前配置"],"sys_action")]); S.append((t,"router_system",b))
    for _ in range(35):
        t,b=_build([_b(["恢复出厂设置","恢复默认设置","重置路由器","恢复出厂"],"sys_action")]); S.append((t,"router_system",b))
    for _ in range(25):
        t,b=_build([_b(["查看系统日志","最近有什么异常","系统出什么故障了"],"sys_action")]); S.append((t,"router_system",b))
    for _ in range(20):
        t,b=_build([_b(["设置"]),_b(["定时重启","每天凌晨重启"],"sys_action")]); S.append((t,"router_system",b))

    # 9. router_qos —— QoS/带宽管理 ★NEW★
    for _ in range(35):
        t,b=_build([_b(["限制"]),_b(["这台电脑","这部手机","这个电视"],"device_name"),_b(["的网速","的带宽"],"qos_target")]); S.append((t,"router_qos",b))
    for _ in range(30):
        t,b=_build([_b(["给"]),_b(["游戏机","电视盒子","电脑"],"device_name"),_b(["优先","提速","更高的优先级"],"qos_action")]); S.append((t,"router_qos",b))
    for _ in range(25):
        t,b=_build([_b(["网速"]),_b(["限制","限速"],"qos_action"),_b(["50兆","100兆","10兆","20兆"],"bandwidth_value")]); S.append((t,"router_qos",b))
    for _ in range(25):
        t,b=_build([_b(["QoS","带宽管理","流量控制"]),_b(["打开","开启","关闭"],"state")]); S.append((t,"router_qos",b))
    for _ in range(20):
        t,b=_build([_b(["限速","给"]),_b(["这台电脑","这部手机","这个电视"],"device_name")]); S.append((t,"router_qos",b))

    # 10. router_security —— 安全策略 ★NEW★
    for _ in range(35):
        t,b=_build([_b(["打开","关闭","开启","禁用"],"state"),_b(["防火墙","安全防护"],"security_type")]); S.append((t,"router_security",b))
    for _ in range(30):
        t,b=_build([_b(["打开","关闭","开启","禁用"],"state"),_b(["VPN","VPN服务"],"security_type")]); S.append((t,"router_security",b))
    for _ in range(25):
        t,b=_build([_b(["家长控制","上网时间管理","儿童模式","青少年模式"],"security_type"),_b(["打开","开启","关闭","关掉"],"state")]); S.append((t,"router_security",b))
    for _ in range(20):
        t,b=_build([_b(["防蹭网","局域网安全"]),_b(["打开","开启","关闭","关掉"],"state")]); S.append((t,"router_security",b))
    for _ in range(20):
        t,b=_build([_b(["查看"]),_b(["安全日志","攻击记录","异常访问"],"security_type")]); S.append((t,"router_security",b))

    # ═══════════════════════════════════════════════════════════════════
    # 🏠 智能家居设备控制（6 个意图，占 ~35%，保留作为扩展能力）
    # ═══════════════════════════════════════════════════════════════════

    # 11. device_control —— 设备开关（合并原 set_device_state + control_ac）
    for _ in range(80):
        t,b=_build([_b(["打开","开启","启动","开一下"],"state"),_b(["风扇","灯","灯光","LED","插座","空调"],"device_type")]); S.append((t,"device_control",b))
    for _ in range(80):
        t,b=_build([_b(["关闭","关掉","停了","关一下"],"state"),_b(["风扇","灯","灯光","LED","插座","空调"],"device_type")]); S.append((t,"device_control",b))
    for _ in range(60):
        t,b=_build([_b(["把"]),_b(["风扇","灯","灯光","LED","空调"],"device_type"),_b(["打开","开启","关了","关掉"],"state")]); S.append((t,"device_control",b))

    # 12. device_adjust —— 设备调节（合并原 adjust_fan_speed + set_light_brightness）
    for _ in range(60):
        t,b=_build([_b(["风扇"]),_b(["调大一点","调大","调快一点","加大风速"],"direction")]); S.append((t,"device_adjust",b))
    for _ in range(60):
        t,b=_build([_b(["风扇"]),_b(["调小一点","调小","调慢一点","减小风速"],"direction")]); S.append((t,"device_adjust",b))
    for _ in range(50):
        t,b=_build([_b(["灯光","灯","亮度"]),_b(["调亮一点","调亮","亮一点","调暗","暗一点"],"direction")]); S.append((t,"device_adjust",b))
    for _ in range(40):
        t,b=_build([_b(["空调"]),_b(["温度调高","温度调低","调高温度","调低温度"],"direction")]); S.append((t,"device_adjust",b))

    # 13. device_query —— 设备状态查询
    for _ in range(60):
        t,b=_build([_b(["风扇","灯","LED","空调"],"device_type"),_b(["开着吗","关了吗","状态怎么样"])]); S.append((t,"device_query",b))
    for _ in range(40):
        t,b=_build([_b(["查看一下","看看"]),_b(["风扇","灯","LED"],"device_type"),_b(["的状态"])]); S.append((t,"device_query",b))

    # 14. timer_setting —— 定时设置
    for _ in range(50):
        t,b=_build([_b(["10分钟后","5分钟后","30分钟后","1小时后"],"timer_duration"),_b(["关闭","打开"],"state"),_b(["风扇","灯"],"device_type")]); S.append((t,"timer_setting",b))
    for _ in range(40):
        t,b=_build([_b(["定时"]),_b(["10分钟","1小时","30分钟","5分钟"],"timer_duration"),_b(["后关闭","后打开"])]); S.append((t,"timer_setting",b))
    for _ in range(20):
        t,b=_build([_b(["路由器"]),_b(["定时重启","每天凌晨重启","每周一重启"],"timer_duration")]); S.append((t,"timer_setting",b))

    # 15. scene_mode —— 场景模式
    for _ in range(50):
        t,b=_build([_b(["切换到","进入","开启","打开"]),_b(["睡眠模式","工作模式","离开模式","回家模式","节能模式"],"mode_name")]); S.append((t,"scene_mode",b))
    for _ in range(30):
        t,b=_build([_b(["设置成","设为"]),_b(["睡眠","工作","离开","回家","节能"],"mode_name"),_b(["模式"])]); S.append((t,"scene_mode",b))

    # 16. help —— 帮助
    for _ in range(80):
        t,b=_build([_b(["帮助","你能做什么","有什么功能","怎么用","指令有哪些","可以做什么","功能介绍","功能列表"])]); S.append((t,"help",b))

    # ═══════════════════════════════════════════════════════════════════
    # 数据增强：同义词替换
    # ═══════════════════════════════════════════════════════════════════
    syn = {"打开":["开启","启动"],"关闭":["关掉","停了"],"风扇":["风机"],"路由器":["路由"]}
    extra = []
    for text, intent, bio in S:
        for _ in range(random.randint(0,1)):
            nt = text
            for src, tgts in syn.items():
                if src in nt and random.random()<0.3:
                    nt = nt.replace(src, random.choice(tgts), 1)
            if nt != text:
                nb = bio[:len(nt)] if len(bio)>=len(nt) else bio+["O"]*(len(nt)-len(bio))
                extra.append((nt,intent,nb))
    S.extend(extra); random.shuffle(S)
    return S


# ═══════════════════════════════════════════════════════════════════════
# 2. 词表构建
# ═══════════════════════════════════════════════════════════════════════
def build_vocab(samples):
    chars = set()
    for t,_,_ in samples:
        for c in t: chars.add(c)
    v = {"<PAD>":0, "<UNK>":1}
    for i,c in enumerate(sorted(chars),2):
        v[c]=i
    return v


# ═══════════════════════════════════════════════════════════════════════
# 3. PyTorch 训练
# ═══════════════════════════════════════════════════════════════════════
def train(samples, vocab):
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    # 意图列表：路由器核心 10 个 + 智能家居 6 个
    INTENTS = [
        # 🔵 路由器核心
        'router_reboot', 'router_wifi_restart', 'router_wifi_config',
        'router_network_query', 'router_led_control', 'router_device_manage',
        'router_network_diag', 'router_system', 'router_qos', 'router_security',
        # 🏠 智能家居
        'device_control', 'device_adjust', 'device_query',
        'timer_setting', 'scene_mode', 'help',
    ]
    SLOTS = [
        'O', 'B-action','I-action','B-state','I-state','B-device_type','I-device_type',
        'B-direction','I-direction','B-timer_duration','I-timer_duration',
        'B-mode_name','I-mode_name',
        'B-query_type','I-query_type','B-config_type','I-config_type',
        'B-config_value','I-config_value','B-manage_action','I-manage_action',
        'B-target','I-target','B-device_name','I-device_name',
        'B-diag_type','I-diag_type','B-sys_action','I-sys_action',
        'B-qos_action','I-qos_action','B-qos_target','I-qos_target',
        'B-bandwidth_value','I-bandwidth_value','B-security_type','I-security_type',
    ]

    im = {n:i for i,n in enumerate(INTENTS)}; i2i = {i:n for i,n in enumerate(INTENTS)}
    sm = {l:i for i,l in enumerate(SLOTS)}; i2s = {i:l for i,l in enumerate(SLOTS)}

    HO = {'风':'封峰丰','扇':'山删珊','开':'凯揩恺','关':'观冠官','灯':'登蹬瞪',
          '大':'达打','小':'晓消萧','亮':'谅量','暗':'按岸','网':'往忘','卡':'咖喀',
          '密':'秘蜜','码':'马玛','频':'品拼','道':'到倒导','踢':'提替梯',
          '黑':'嘿嗨','屏':'平凭萍','断':'段短锻','诊':'真珍针','延':'言严沿',
          '重':'虫从崇','置':'志智制','级':'急吉辑','安':'按暗岸','火':'伙活或'}

    def err(t, b):
        m = random.randint(1,4)
        if m == 1 and len(t) > 2:
            for s, tgts in HO.items():
                if s in t: i = t.index(s); t = t[:i]+random.choice(list(tgts))+t[i+1:]; break
        elif m == 2 and len(t) > 2: i = random.randint(0,len(t)-1); t = t[:i]+t[i+1:]; b = b[:i]+b[i+1:]
        elif m == 3 and len(t) > 1: i = random.randint(0,len(t)-1); t = t[:i+1]+t[i]+t[i+1:]; b = b[:i+1]+[b[i]]+b[i+1:]
        return t,b

    class DS(Dataset):
        def __init__(s): s.d = samples
        def __len__(s): return len(s.d)
        def __getitem__(s, i):
            t, intent, b = s.d[i]
            if random.random() < 0.3: t, b = err(t, b)
            t, b = t[:LEN], b[:LEN]; n = len(t)
            ids = [vocab.get(c, 1) for c in t] + [0]*(LEN-n)
            a = [1]*n + [0]*(LEN-n)
            si = [sm.get(lb, 0) for lb in b] + [0]*(LEN-n)
            return {'input_ids':torch.tensor(ids, dtype=torch.long),
                    'attention_mask':torch.tensor(a, dtype=torch.long),
                    'intent':torch.tensor(im[intent], dtype=torch.long),
                    'slots':torch.tensor(si, dtype=torch.long)}

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Embedding(len(vocab), 128, padding_idx=0)
            s.c1 = nn.Conv1d(128, 128, 3, padding=1)
            s.c2 = nn.Conv1d(128, 256, 3, padding=1)
            s.lstm = nn.LSTM(256, 256, 2, batch_first=True, bidirectional=True, dropout=DROP)
            s.drop = nn.Dropout(DROP)
            s.ifc = nn.Linear(512, len(INTENTS))
            s.sfc = nn.Linear(512, len(SLOTS))
        def forward(s, i, a):
            x = s.emb(i).transpose(1,2); x = torch.relu(s.c1(x)); x = torch.relu(s.c2(x))
            x = x.transpose(1,2); o,_ = s.lstm(x); o = s.drop(o)
            l = a.sum(1).clamp(min=1)
            li = (l-1).unsqueeze(1).unsqueeze(2).expand(-1,1,o.size(-1))
            return s.ifc(o.gather(1,li).squeeze(1)), s.sfc(o)

    ds = DS(); dl = DataLoader(ds, BS, True); model = M()
    ci = nn.CrossEntropyLoss(); cs = nn.CrossEntropyLoss(ignore_index=0)
    opt = torch.optim.Adam(model.parameters(), LR)

    print(f'训练 {len(samples)}条 | {len(vocab)}字符 | ep={EP} bs={BS}\n')
    for ep in range(1, EP+1):
        model.train(); tl = ci_t = cs_t = ts = 0
        for b in dl:
            opt.zero_grad(); il, sl = model(b['input_ids'], b['attention_mask'])
            li = ci(il, b['intent']); ls = cs(sl.transpose(1,2), b['slots'])
            (li+ls).backward(); opt.step(); tl += li.item()+ls.item()
            ci_t += (il.argmax(1)==b['intent']).sum().item()
            m = (b['slots']!=0) & (b['attention_mask']==1)
            cs_t += ((sl.argmax(2)==b['slots'])&m).sum().item(); ts += m.sum().item()
        ia = ci_t/len(samples)*100; sa = cs_t/max(ts,1)*100
        bar = chr(9608)*(ep*20//EP) + chr(9617)*(20-ep*20//EP)
        print(f'  Ep{ep:3d} [{bar}] loss={tl/len(dl):.4f} intent={ia:.1f}% slot={sa:.1f}%')

    # ONNX 导出
    print('\n导出 ONNX...')
    os.makedirs('models/nlu', exist_ok=True)
    model.eval()
    dum = torch.zeros(1, LEN, dtype=torch.long); dma = torch.ones(1, LEN, dtype=torch.long)
    torch.onnx.export(model, (dum,dma), 'models/nlu/intent_model.onnx',
                      input_names=['input_ids','attention_mask'],
                      output_names=['intent_logits','slot_logits'],
                      dynamic_axes={'input_ids':{0:'batch'},'attention_mask':{0:'batch'},
                                    'intent_logits':{0:'batch'},'slot_logits':{0:'batch'}},
                      opset_version=13)
    sz = os.path.getsize('models/nlu/intent_model.onnx')/1024/1024
    print(f'  ONNX fp32: {sz:.1f} MB')

    # INT8 动态量化（路由器部署）
    int8_path = 'models/nlu/intent_model.int8.onnx'
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(
            'models/nlu/intent_model.onnx',
            int8_path,
            weight_type=QuantType.QUInt8,
        )
        sz8 = os.path.getsize(int8_path) / 1024 / 1024
        print(f'  ONNX int8: {sz8:.1f} MB → {int8_path}')
    except Exception as exc:
        print(f'  INT8 量化跳过: {exc}')

    with open('models/nlu/intent_labels.json','w') as f:
        json.dump(i2i, f, ensure_ascii=False, indent=2)
    with open('models/nlu/slot_labels.json','w') as f:
        json.dump(i2s, f, ensure_ascii=False, indent=2)

    # 自检 —— 路由器场景重点
    print('\n自检:')
    tests = [
        # 🔵 路由器核心场景
        '重启路由器','重启WiFi','IP地址是多少','改WiFi密码','关掉路由器灯',
        '网速怎么样','踢掉这个手机','检查固件更新','备份配置','恢复出厂设置',
        '限速这台电脑','打开防火墙','测一下网速','延迟多少','家长控制打开',
        '切换到5G频段','查看黑名单','谁连了WiFi',
        # 🏠 智能家居
        '打开风扇','关闭灯光','调大一点','亮度调暗','风扇开着吗',
        '10分钟后关闭','睡眠模式','帮助',
    ]
    for t in tests:
        n=min(len(t),LEN);ids=torch.tensor([[vocab.get(c,1) for c in t[:n]]+[0]*(LEN-n)])
        a=torch.tensor([[1]*n+[0]*(LEN-n)])
        with torch.no_grad():
            il,_=model(ids,a);pid=il.argmax(1).item()
        q = '"'
        print(f'  {q}{t}{q} → {i2i.get(pid,"?")}')
    return True


# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('='*60)
    print('  NLU 模型训练 — 字符级 CNN+LSTM（路由器核心版）')
    print('='*60)
    print('\n[1/3] 生成训练数据...')
    samples = gen_data()
    cnt = {}
    for _,i,_ in samples: cnt[i]=cnt.get(i,0)+1

    # 分类统计
    router_intents = ['router_reboot','router_wifi_restart','router_wifi_config',
                      'router_network_query','router_led_control','router_device_manage',
                      'router_network_diag','router_system','router_qos','router_security']
    router_cnt = sum(cnt.get(i,0) for i in router_intents)
    total = len(samples)
    print(f'  {total} 条 (路由器 {router_cnt} 条, {router_cnt/total*100:.0f}%)')
    print(f'  🔵 路由器核心场景:')
    for k in router_intents:
        c = cnt.get(k,0)
        bar = chr(9608)*(c//10) if c>0 else ''
        print(f'    {k:<25s} {c:>4d} {bar}')
    print(f'  🏠 智能家居扩展:')
    for k in sorted(cnt.keys()):
        if k not in router_intents:
            c = cnt.get(k,0)
            bar = chr(9617)*(c//10) if c>0 else ''
            print(f'    {k:<25s} {c:>4d} {bar}')

    print('\n[2/3] 词表...')
    vocab = build_vocab(samples)
    print(f'  {len(vocab)} 字符')
    with open('models/nlu/vocab.json','w') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print('\n[3/3] 训练 & ONNX 导出...')
    try:
        train(samples, vocab)
        print('\n✅ 完成！模型: models/nlu/intent_model.onnx')
    except ImportError as e:
        print(f'\n❌ {e}')
        print('pip install torch onnx')
