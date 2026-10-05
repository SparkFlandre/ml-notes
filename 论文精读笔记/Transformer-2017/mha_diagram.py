# -*- coding: utf-8 -*-
"""多头注意力机制（Multi-Head Attention）完整结构详解图。"""
import math
import os
from PIL import Image, ImageDraw, ImageFont

WS = r"D:\KimiData\kimi\tasks\2026-10-05\17-27-06-8c21acd1"
DST = os.path.join(WS, "transformer_mha_detail.png")

FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)

W, H = 2200, 1180
img = Image.new("RGB", (W, H), "#FFFFFF")
d = ImageDraw.Draw(img)

# ---- 颜色 ----
C_TEXT = "#1F2328"
C_SUB = "#4B5563"
C_BORDER = "#9AA0A6"
C_LINE = "#6B7280"
Q_FILL, Q_BD = "#E8F0FE", "#7A9CC6"
K_FILL, K_BD = "#FCE8E6", "#C67A7A"
V_FILL, V_BD = "#E6F4EA", "#7AA87F"
N_FILL, N_BD = "#F1F3F4", "#9AA0A6"   # 中性
A_FILL, A_BD = "#FFF4E0", "#C9A35B"   # 注意力橙
PANEL_FILL = "#FAFBFC"

def ctext(cx, cy, s, fnt, fill=C_TEXT):
    tw = d.textlength(s, font=fnt)
    asc, desc = fnt.getmetrics()
    d.text((cx - tw / 2, cy - (asc + desc) / 2), s, font=fnt, fill=fill)

def box(x0, y0, x1, y1, lines, fill=N_FILL, outline=N_BD, title_size=19, sub_size=15, radius=10):
    d.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=outline, width=2)
    n = len(lines)
    lh = 24 if n > 1 else 26
    total = n * lh
    cy = (y0 + y1) / 2 - total / 2 + lh / 2
    for i, ln in enumerate(lines):
        fnt = font(title_size, bold=True) if i == 0 else font(sub_size)
        fill_c = C_TEXT if i == 0 else C_SUB
        ctext((x0 + x1) / 2, cy + i * lh, ln, fnt, fill_c)
    return (x0, y0, x1, y1)

def arrowhead(tip, direction, size=10, color=C_LINE):
    dx, dy = direction
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    tx, ty = tip
    b1 = (tx - ux * size + px * size * 0.5, ty - uy * size + py * size * 0.5)
    b2 = (tx - ux * size - px * size * 0.5, ty - uy * size - py * size * 0.5)
    d.polygon([tip, b1, b2], fill=color)

def draw_dashed(p1, p2, color=C_LINE, width=2, dash=8, gap=6):
    x1, y1 = p1; x2, y2 = p2
    L = math.hypot(x2 - x1, y2 - y1)
    if L == 0: return
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    t = 0
    while t < L:
        t2 = min(t + dash, L)
        d.line([(x1 + ux * t, y1 + uy * t), (x1 + ux * t2, y1 + uy * t2)], fill=color, width=width)
        t = t2 + gap

def elbow(points, dashed=False, color=C_LINE, width=2):
    for i in range(len(points) - 1):
        if dashed:
            draw_dashed(points[i], points[i + 1], color, width)
        else:
            d.line([points[i], points[i + 1]], fill=color, width=width)
    # 箭头方向 = 最后一段
    (ax, ay), (bx, by) = points[-2], points[-1]
    arrowhead((bx, by), (bx - ax, by - ay), color=color)

# ================= 标题 =================
d.text((60, 22), "多头注意力机制 Multi-Head Attention 结构详解", font=font(30, bold=True), fill=C_TEXT)
d.text((60, 64), "输入 n×512 → 三套线性投影得 Q/K/V → 拆成 8 个头并行做缩放点积注意力 → 拼接 → W^O 线性混合输出",
       font=font(17), fill=C_SUB)
d.text((1520, 30), "论文设定：d_model = 512，头数 h = 8，d_k = d_v = 64",
       font=font(17, bold=True), fill=C_SUB)

# ================= 上层：整体数据流 =================
# 输入 X
box(60, 250, 250, 370, ["输入 X", "n × 512"], N_FILL, N_BD, 22, 16)

# 三个线性投影
for (y0, name) in [(140, "W_Q"), (255, "W_K"), (370, "W_V")]:
    fill = {"W_Q": Q_FILL, "W_K": K_FILL, "W_V": V_FILL}[name]
    bd = {"W_Q": Q_BD, "W_K": K_BD, "W_V": V_BD}[name]
    box(330, y0, 520, y0 + 80, [f"Linear {name}", "512 × 512"], fill, bd)
    elbow([(250, 310), (290, 310), (290, y0 + 40), (330, y0 + 40)])

# Q/K/V
qkv = {"Q": (Q_FILL, Q_BD), "K": (K_FILL, K_BD), "V": (V_FILL, V_BD)}
for (y0, name) in [(140, "Q"), (255, "K"), (370, "V")]:
    fill, bd = qkv[name]
    box(590, y0, 780, y0 + 80, [f"{name} = X·{name == 'Q' and 'W_Q' or name == 'K' and 'W_K' or 'W_V'}", "n × 512"], fill, bd, 20, 15)
    d.line([(520, y0 + 40), (590, y0 + 40)], fill=C_LINE, width=2)
    arrowhead((590, y0 + 40), (1, 0))

# 拆分总线
d.line([(815, 144), (815, 509)], fill=C_LINE, width=2)
for y in (180, 295, 410):
    d.line([(780, y), (815, y)], fill=C_LINE, width=2)
d.text((788, 555), "Split 拆 8 头", font=font(14), fill=C_SUB)

# 头 1 / 头 2 / ⋮ / 头 8
heads_y = [(105, "头 1"), (260, "头 2"), (470, "头 8")]
for (y0, name) in heads_y:
    box(850, y0, 1040, y0 + 78, [name, "Q_i K_i V_i 各 n×64"], N_FILL, N_BD, 20, 14)
    elbow([(815, y0 + 39), (850, y0 + 39)])
def vdots(cx, cy):
    for i in (-1, 0, 1):
        d.ellipse([cx - 4, cy + i * 18 - 4, cx + 4, cy + i * 18 + 4], fill=C_SUB)

vdots(945, 415)

# 每头的缩放点积注意力
for (y0, _) in heads_y:
    box(1110, y0, 1330, y0 + 78, ["缩放点积注意力", "Scaled dot-product"], A_FILL, A_BD, 19, 14)
    d.line([(1040, y0 + 39), (1110, y0 + 39)], fill=C_LINE, width=2)
    arrowhead((1110, y0 + 39), (1, 0))
vdots(1220, 415)

# Concat
box(1400, 250, 1590, 380, ["Concat 拼接", "8 × (n×64)", "= n × 512"], N_FILL, N_BD, 21, 15)
for (y0, ty) in [(105, 280), (260, 315), (470, 350)]:
    elbow([(1330, y0 + 39), (1365, y0 + 39), (1365, ty), (1400, ty)])

# W^O 与输出
box(1660, 265, 1850, 365, ["Linear W^O", "512 × 512"], N_FILL, N_BD, 21, 15)
d.line([(1590, 315), (1660, 315)], fill=C_LINE, width=2); arrowhead((1660, 315), (1, 0))
box(1920, 265, 2130, 365, ["输出", "n × 512"], N_FILL, N_BD, 22, 16)
d.line([(1850, 315), (1920, 315)], fill=C_LINE, width=2); arrowhead((1920, 315), (1, 0))

# 放大虚线
elbow([(1220, 338), (1220, 560)], dashed=True)
d.text((1240, 450), "放大详解", font=font(16), fill=C_SUB)

# ================= 下层左：单头内部放大 =================
d.rounded_rectangle([60, 600, 1350, 1120], radius=14, fill=PANEL_FILL, outline=C_BORDER, width=2)
d.text((90, 622), "单个头内部：缩放点积注意力（第 i 个头，d_k = 64）", font=font(22, bold=True), fill=C_TEXT)

# 第一行：Q_i, K_i^T → MatMul → 得分 → ÷√d_k → Mask → Softmax
box(100, 690, 235, 765, ["Q_i", "n × 64"], Q_FILL, Q_BD, 20, 14)
box(100, 830, 235, 905, ["K_i 转置", "64 × n"], K_FILL, K_BD, 20, 14)

# MatMul1 圆
d.ellipse([290, 757, 375, 842], fill="#FFFFFF", outline=C_BORDER, width=2)
ctext(332, 800, "MatMul", font(15, bold=True))
elbow([(235, 727), (262, 727), (262, 780), (290, 780)])
elbow([(235, 867), (262, 867), (262, 820), (290, 820)])

box(430, 745, 570, 815, ["得分矩阵", "n × n"], N_FILL, N_BD, 19, 14)
d.line([(375, 800), (400, 800), (400, 780), (430, 780)], fill=C_LINE, width=2); arrowhead((430, 780), (1, 0))

box(630, 745, 760, 815, ["÷ √d_k", "= ÷ 8"], N_FILL, N_BD, 20, 14)
d.line([(570, 780), (630, 780)], fill=C_LINE, width=2); arrowhead((630, 780), (1, 0))

box(810, 745, 960, 815, ["Mask 掩码", "上三角 −∞", "仅解码器用"], N_FILL, N_BD, 18, 13)
d.line([(760, 780), (810, 780)], fill=C_LINE, width=2); arrowhead((810, 780), (1, 0))

box(1000, 745, 1140, 815, ["Softmax", "按行归一化"], A_FILL, A_BD, 19, 14)
d.line([(960, 780), (1000, 780)], fill=C_LINE, width=2); arrowhead((1000, 780), (1, 0))

# 第二行：V_i 与 权重 → MatMul → head_i
box(430, 880, 570, 950, ["V_i", "n × 64"], V_FILL, V_BD, 20, 14)
d.ellipse([640, 873, 725, 958], fill="#FFFFFF", outline=C_BORDER, width=2)
ctext(682, 915, "MatMul", font(15, bold=True))
d.line([(570, 915), (640, 915)], fill=C_LINE, width=2); arrowhead((640, 915), (1, 0))

box(800, 880, 960, 950, ["head_i", "n × 64"], A_FILL, A_BD, 20, 14)
d.line([(725, 915), (800, 915)], fill=C_LINE, width=2); arrowhead((800, 915), (1, 0))

# Softmax → 权重 → MatMul2 的长肘线
elbow([(1070, 815), (1070, 1000), (682, 1000), (682, 958)])
d.text((870, 965), "注意力权重 n × n", font=font(16, bold=True), fill=C_SUB)

d.text((100, 1060), "8 个 head_i（各 n×64）按特征维拼接 → 回到上层 Concat → W^O",
       font=font(16), fill=C_SUB)

# ================= 下层右：关键细节 =================
d.rounded_rectangle([1400, 600, 2140, 1120], radius=14, fill=PANEL_FILL, outline=C_BORDER, width=2)
d.text((1430, 622), "关键细节", font=font(22, bold=True), fill=C_TEXT)

NOTES = [
    "Q / K / V 直觉：Q = 我在找什么，K = 我的标签，V = 我实际携带的信息；三套独立可学习投影让“检索”与“内容”分工，且关系不对称（i→j ≠ j→i）",
    "为什么 ÷√d_k：q·k 的方差随 d_k 线性增长，除以 √d_k 使方差回到 1 量级，防止 softmax 饱和、梯度消失",
    "为什么多头：8 个头在 8 个 64 维子空间并行，各学不同关系（语法、指代、位置相邻等），类似 CNN 的多个卷积核；W^O 负责把 8 个头的结果重新混合",
    "参数与算力：W_Q / W_K / W_V / W^O 各为 512×512；8 头 × 64 维的总计算量与单头 512 维大致相当，但表达能力更强",
    "Mask：仅解码器自注意力使用（softmax 前上三角加 −∞，使未来位置概率为 0）；编码器自注意力与交叉注意力不加掩码",
    "交叉注意力复用同一结构：仅 Q 改由解码器提供，K / V 来自编码器最终输出",
]
ty = 672
for note in NOTES:
    d.ellipse([1432, ty + 8, 1442, ty + 18], fill=C_TEXT)
    lines = []
    cur = ""
    for ch in note:
        if d.textlength(cur + ch, font=font(16)) <= 640:
            cur += ch
        else:
            lines.append(cur); cur = ch
    if cur: lines.append(cur)
    for ln in lines:
        d.text((1456, ty), ln, font=font(16), fill=C_SUB)
        ty += 25
    ty += 10

img.save(DST, "PNG")
print("saved:", DST, img.size)
