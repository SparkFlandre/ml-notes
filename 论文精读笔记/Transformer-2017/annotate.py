# -*- coding: utf-8 -*-
"""在 Transformer 架构图四周添加悬浮标注（callout + leader line）。"""
import os
from PIL import Image, ImageDraw, ImageFont

WS = r"D:\KimiData\kimi\tasks\2026-10-05\17-27-06-8c21acd1"
SRC = os.path.join(WS, "transformer_arch.png")
DST = os.path.join(WS, "transformer_arch_annotated.png")

# ---- 字体（微软雅黑，含中文） ----
FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
if not os.path.exists(FONT_REG):
    raise SystemExit("未找到微软雅黑字体")

def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)

F_TITLE = font(20, bold=True)
F_BODY = font(17)
F_NUM = font(18, bold=True)

# ---- 画布：原图居中，左右各留 460，上下留白 ----
img = Image.open(SRC).convert("RGB")
IW, IH = img.size                      # 1068 x 769
ML, MR, MT, MB = 470, 470, 70, 30
W, H = IW + ML + MR, IH + MT + MB
canvas = Image.new("RGB", (W, H), "#FFFFFF")
OX, OY = ML, MT                        # 原图在新画布上的偏移
canvas.paste(img, (OX, OY))
d = ImageDraw.Draw(canvas)

def P(x, y):
    """原图坐标 -> 新画布坐标"""
    return (x + OX, y + OY)

# ---- 样式 ----
C_BORDER = "#9AA0A6"
C_FILL = "#F6F7F9"
C_TEXT = "#1F2328"
C_SUB = "#4B5563"
C_LINE = "#6B7280"
C_DOT = "#111111"

def wrap_text(text, fnt, max_w):
    """按像素宽度折行（中英文混排，逐字符）。"""
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur); cur = ""; continue
        if d.textlength(cur + ch, font=fnt) <= max_w:
            cur += ch
        else:
            lines.append(cur); cur = ch
    if cur:
        lines.append(cur)
    return lines

def callout(cx, cy, num, title, body, box_w=400):
    """在 (cx, cy) 为左上角画标注框，返回框的矩形。"""
    pad = 14
    body_lines = wrap_text(body, F_BODY, box_w - 2 * pad - 6)
    title_h = 26
    body_h = len(body_lines) * 24
    box_h = pad * 2 + title_h + 6 + body_h
    x0, y0, x1, y1 = cx, cy, cx + box_w, cy + box_h
    d.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=C_FILL, outline=C_BORDER, width=1)
    # 编号圆点
    r = 13
    ccx, ccy = x0 + pad + r, y0 + pad + r
    d.ellipse([ccx - r, ccy - r, ccx + r, ccy + r], fill=C_DOT)
    tw = d.textlength(str(num), font=F_NUM)
    d.text((ccx - tw / 2, ccy - 11), str(num), font=F_NUM, fill="#FFFFFF")
    d.text((x0 + pad + 2 * r + 8, y0 + pad - 2), title, font=F_TITLE, fill=C_TEXT)
    ty = y0 + pad + title_h + 6
    for ln in body_lines:
        d.text((x0 + pad, ty), ln, font=F_BODY, fill=C_SUB)
        ty += 24
    return (x0, y0, x1, y1)

def leader(rect, side, target):
    """从标注框侧边引一条线到目标点，目标处画小圆点。"""
    x0, y0, x1, y1 = rect
    tx, ty = target
    if side == "right":
        sx, sy = x1, (y0 + y1) // 2
    else:
        sx, sy = x0, (y0 + y1) // 2
    mid_x = (sx + tx) // 2
    d.line([ (sx, sy), (mid_x, sy), (mid_x, ty), (tx, ty) ], fill=C_LINE, width=2)
    d.ellipse([tx - 5, ty - 5, tx + 5, ty + 5], fill=C_LINE)

# ================= 左侧标注（从上到下） =================
# 1. 位置编码（指向左侧波浪 + ⊕）
r = callout(20, 40, 1, "位置编码 Positional Encoding",
            "sin/cos 编码与词嵌入逐元素相加；\n自注意力本身无序，必须注入位置信息")
leader(r, "right", P(420, 552))

# 2. 多头自注意力（指向编码器橙色 MHA）
r = callout(20, 190, 2, "多头自注意力",
            "softmax(QK^T/√d_k)·V；8 头×64 维\n除以√d_k 防止 softmax 饱和、梯度消失")
leader(r, "right", P(412, 460))

# 3. Add & Norm（指向编码器中部 Add&Norm）
r = callout(20, 340, 3, "Add & Norm 残差+归一化",
            "LayerNorm(x + Sublayer(x))\n残差是梯度高速公路，6 层堆叠的前提")
leader(r, "right", P(400, 425))

# 4. N× 六层堆叠（指向编码器 N×）
r = callout(20, 490, 4, "×6 层堆叠",
            "浅层学局部搭配，深层学句法语义\n每层独立权重，输出直接接力给下一层")
leader(r, "right", P(360, 470))

# 5. 词嵌入（指向 Input Embedding）
r = callout(20, 640, 5, "词嵌入 Embedding",
            "token→512 维向量，×√d_model 缩放\n不预训练，随任务端到端学习")
leader(r, "right", P(410, 605))

# ================= 右侧标注（从上到下） =================
RX = OX + IW + 20   # 右侧标注起点 x

# 6. Linear + Softmax（指向 Softmax/Linear）
r = callout(RX, 40, 6, "输出层 Linear + Softmax",
            "投影到词表维度，与输入嵌入共享权重\nSoftmax 输出下一个词的概率分布")
leader(r, "left", P(670, 165))

# 7. FFN（指向解码器蓝色 FFN）
r = callout(RX, 185, 7, "前馈网络 FFN",
            "512→2048→512 先升维再降维\n约占每层 2/3 参数，逐位置独立作用")
leader(r, "left", P(668, 255))

# 8. 交叉注意力（指向解码器中部 MHA）
r = callout(RX, 330, 8, "交叉注意力 Cross-Attention",
            "Q 来自解码器，K、V 来自编码器输出\n每生成一词就查询一遍源句表示")
leader(r, "left", P(668, 340))

# 9. 掩码注意力（指向 Masked MHA）
r = callout(RX, 475, 9, "掩码多头注意力 Masked MHA",
            "softmax 前上三角加 −∞（非无穷小）\n屏蔽未来 token，保证自回归")
leader(r, "left", P(668, 462))

# 10. 输出嵌入 + 右移（指向 Output Embedding）
r = callout(RX, 625, 10, "输出嵌入与右移",
            "训练：teacher forcing 整句并行\n推理：同一解码器反复调用、逐词生成")
leader(r, "left", P(668, 605))

# ---- 标题 ----
d.text((OX, 18), "Transformer 架构详解（标注基于笔记要点）", font=font(24, bold=True), fill=C_TEXT)

canvas.save(DST, "PNG")
print("saved:", DST, canvas.size)
