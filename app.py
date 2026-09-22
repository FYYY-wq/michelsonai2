"""迈克尔逊干涉圆环吞吐自动计数 —— 网页版（Streamlit）

运行方式：
    streamlit run app.py
评委打开浏览器即可：上传视频 / 选择样例 → 自动分析 → 查看标注视频与统计。
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st


# ============ 核心算法（与 mechelson_count.py 一致，去掉弹窗交互） ============

def moving_average(x, n):
    n = max(1, int(n))
    if n == 1:
        return x.copy()
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, np.ones(n) / n, mode="valid")


def find_peaks(signal, distance, prominence):
    """无 SciPy 依赖的显著峰检测。"""
    span = max(2, distance // 2)
    candidates = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            left = np.min(signal[max(0, i - span):i + 1])
            right = np.min(signal[i:min(len(signal), i + span + 1)])
            if signal[i] - max(left, right) >= prominence:
                candidates.append(i)
    selected = []
    for i in sorted(candidates, key=lambda j: signal[j], reverse=True):
        if all(abs(i - j) >= distance for j in selected):
            selected.append(i)
    return sorted(selected)


def polar_maps(max_radius, angle_count=180):
    radii = np.arange(max_radius, dtype=np.float32)[None, :]
    angles = np.linspace(0, 2 * np.pi, angle_count, endpoint=False, dtype=np.float32)[:, None]
    cx = cy = float(max_radius)
    map_x = cx + np.cos(angles) * radii
    map_y = cy + np.sin(angles) * radii
    return map_x.astype(np.float32), map_y.astype(np.float32)


def fast_radial_profile(red_roi, maps):
    polar = cv2.remap(red_roi, maps[0], maps[1], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)
    return polar.mean(axis=0).astype(np.float32)


def laser_red_channel(frame):
    """提取 He-Ne 激光的超额红色，抑制白光、灰色背景和曝光波动。"""
    b, g, r = cv2.split(frame.astype(np.float32))
    excess_red = r - 0.5 * (g + b)
    excess_red = np.clip(excess_red, 0, 255)
    return cv2.GaussianBlur(excess_red, (5, 5), 0)


def auto_detect_center(frame):
    """自动估计干涉圆环圆心（无需鼠标点选）。"""
    red = cv2.GaussianBlur(frame[:, :, 2], (5, 5), 0)
    h, w = red.shape
    weights = np.maximum(red.astype(float) - np.percentile(red, 70), 0) ** 2
    yy, xx = np.indices(red.shape)
    x0 = int((weights * xx).sum() / max(weights.sum(), 1))
    y0 = int((weights * yy).sum() / max(weights.sum(), 1))

    def score(x, y):
        radius = min(220, x, y, w - x - 1, h - y - 1)
        if radius < 100:
            return -1.0
        polar = cv2.warpPolar(red, (radius, 240), (x, y), radius,
                              cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS).astype(float)
        polar = polar[:, 25:radius]
        return float(np.var(polar.mean(axis=0)) / (np.var(polar) + 1e-9))

    coarse = [(score(x, y), x, y)
              for y in range(max(100, y0 - 170), min(h - 100, y0 + 171), 10)
              for x in range(max(100, x0 - 170), min(w - 100, x0 + 171), 10)]
    if not coarse:
        return None
    _, bx, by = max(coarse)
    fine = [(score(x, y), x, y)
            for y in range(by - 10, by + 11, 2) for x in range(bx - 10, bx + 11, 2)]
    best_score, bx, by = max(fine)
    if best_score < 0.15:
        return None
    return (bx, by)


def run_analysis(video_path, params, progress_callback=None):
    """完整分析流程，返回结果字典。progress_callback(cur, total, msg) 用于进度条。"""
    cap = cv2.VideoCapture(str(video_path))
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("无法打开视频文件")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = first.shape[1], first.shape[0]

    if params.get("center_override"):
        center = tuple(params["center_override"])
    else:
        center = auto_detect_center(first)
        if center is None:
            cap.release()
            raise RuntimeError("自动圆心识别失败，请在左侧勾选「手动指定圆心」并输入坐标")

    max_allowed = int(min(center[0], center[1], width - center[0], height - center[1])) - 2
    max_radius = min(params["max_radius"], max_allowed)
    if max_radius < 50:
        cap.release()
        raise RuntimeError("视频画面中圆心附近可用区域太小，请检查圆心坐标")
    radius = params["radius"]
    center_radius = params["center_radius"]

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    signals, profiles = [], []
    maps = polar_maps(max_radius)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        x, y = center
        roi = frame[y - max_radius:y + max_radius + 1, x - max_radius:x + max_radius + 1]
        if roi.size == 0:
            break
        profile = fast_radial_profile(laser_red_channel(roi), maps)
        cr = min(center_radius, max_radius - 1)
        if cr > 3:
            weights = np.arange(3, cr, dtype=np.float32)
            signals.append(np.average(profile[3:cr], weights=weights))
        else:
            signals.append(float(profile[cr]))
        profiles.append(profile)
        if progress_callback:
            progress_callback(len(signals), total_frames, "逐帧分析中")

    signals = np.asarray(signals, np.float32)
    smooth = moving_average(signals, round(params["smooth"] * fps))
    lo, hi = np.percentile(smooth, [5, 95])
    normalized = (smooth - lo) / max(hi - lo, 1e-6)
    peaks = find_peaks(normalized, max(2, round(params["min_period"] * fps)), params["prominence"])

    # 径向位移 → 吞入/吐出方向
    velocity = np.zeros(len(profiles), np.float32)
    r0, r1 = max(10, radius - 55), min(max_radius, radius + 55)
    for i in range(1, len(profiles)):
        a, b = profiles[i - 1][r0:r1], profiles[i][r0:r1]
        a, b = a - a.mean(), b - b.mean()
        shifts = range(-4, 5)
        scores = [np.dot(a[max(0, s):len(a) + min(0, s)],
                         b[max(0, -s):len(b) + min(0, -s)]) for s in shifts]
        velocity[i] = list(shifts)[int(np.argmax(scores))]
    velocity = moving_average(velocity, round(0.35 * fps))

    nonzero_velocity = velocity[np.abs(velocity) > 0.02]
    fallback_velocity = float(np.median(nonzero_velocity)) if len(nonzero_velocity) else 0.0
    events = []
    for p in peaks:
        nearby = velocity[max(0, p - round(1.0 * fps)):min(len(velocity), p + round(1.0 * fps) + 1)]
        moving = nearby[np.abs(nearby) > 0.02]
        v = float(np.median(moving)) if len(moving) else fallback_velocity
        direction = "吐出" if v < 0 else "吞入"
        events.append({"frame": int(p), "time_s": round(p / fps, 3),
                       "direction": direction, "radial_shift": round(v, 3)})

    swallowed = sum(1 for e in events if e["direction"] == "吞入")
    emitted = len(events) - swallowed

    # 生成标注视频
    out_dir = Path(tempfile.gettempdir()) / "michelson_out"
    out_dir.mkdir(exist_ok=True)
    output_path = str(out_dir / "annotated.mp4")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    event_map = {e["frame"]: e for e in events}
    running_in = running_out = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in event_map:
            d = event_map[i]["direction"]
            running_in += (d == "吞入")
            running_out += (d == "吐出")
        cv2.circle(frame, center, radius, (0, 255, 255), 2)
        cv2.circle(frame, center, center_radius, (255, 255, 0), 2)
        cv2.drawMarker(frame, center, (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.rectangle(frame, (12, 12), (390, 105), (0, 0, 0), -1)
        cv2.putText(frame, f"Total: {running_in + running_out}", (25, 42), 0, .8, (255, 255, 255), 2)
        cv2.putText(frame, f"In: {running_in}   Out: {running_out}", (25, 78), 0, .8, (255, 255, 255), 2)
        near = [p for p in event_map if 0 <= i - p <= round(.3 * fps)]
        if near:
            p = near[-1]
            d = event_map[p]["direction"]
            color = (0, 80, 255) if d == "吞入" else (255, 120, 0)
            cv2.putText(frame, "COUNT!  " + ("IN" if d == "吞入" else "OUT"),
                        (25, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.15, color, 3)
        writer.write(frame)
        i += 1
        if progress_callback:
            progress_callback(i, total_frames, "生成标注视频")

    cap.release()
    writer.release()

    return {
        "center": center, "fps": round(fps, 1), "total_frames": total_frames,
        "swallowed": swallowed, "emitted": emitted, "total": len(events),
        "events": events, "output_video": output_path,
        "normalized": normalized.tolist(),
    }


# ============ Streamlit 界面 ============

st.set_page_config(page_title="迈克尔逊干涉圆环计数系统", page_icon="🔬", layout="wide")

st.title("🔬 迈克尔逊干涉圆环吞吐自动计数系统")
st.markdown("上传干涉条纹视频，系统自动识别圆心、逐帧分析圆环 **吞入 / 吐出** 数量，并输出标注视频与事件明细。")

# ---- 侧边栏参数 ----
st.sidebar.header("⚙️ 分析参数")
radius = st.sidebar.slider("计数半径 (px)", 20, 300, 120)
center_radius = st.sidebar.slider("圆心判定半径 (px)", 10, 150, 35)
prominence = st.sidebar.slider("峰值显著度（越大漏计越多）", 0.02, 0.30, 0.08, 0.01)
smooth_s = st.sidebar.slider("时间平滑 (秒)", 0.05, 1.0, 0.20, 0.05)
min_period = st.sidebar.slider("最短事件间隔 (秒)", 0.1, 2.0, 0.35, 0.05)

st.sidebar.markdown("---")
manual_center = st.sidebar.checkbox("手动指定圆心（自动失败时勾选）")
cx = cy = None
if manual_center:
    cx = st.sidebar.number_input("圆心 X", 0, 4000, 640)
    cy = st.sidebar.number_input("圆心 Y", 0, 4000, 360)

# ---- 视频输入 ----
st.subheader("📹 第一步：选择视频")
SAMPLE = "test.mp4"
src = st.radio("视频来源", ["使用内置样例视频", "上传自己的 mp4"], horizontal=True, label_visibility="collapsed")

video_file = None
tmp_upload = None
if src == "上传自己的 mp4":
    up = st.file_uploader("选择视频文件", type=["mp4", "avi", "mov", "mkv"])
    if up:
        tmp_upload = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_upload.write(up.read())
        tmp_upload.flush()
        video_file = tmp_upload.name
        st.video(video_file)
else:
    if os.path.exists(SAMPLE):
        video_file = SAMPLE
        if "result" not in st.session_state:
            st.video(SAMPLE)
            st.caption("↑ 内置样例视频（test.mp4），点击下方按钮即可自动分析")
    else:
        st.warning("未找到内置样例视频 test.mp4，请改为上传视频")

# ---- 分析按钮 ----
st.subheader("🚀 第二步：开始分析")

# 默认自动分析样例视频（缓存，只算一次）；上传新视频后点按钮重新分析
params = {
    "radius": radius, "center_radius": center_radius, "max_radius": 260,
    "smooth": smooth_s, "min_period": min_period, "prominence": prominence,
}
if manual_center and cx is not None:
    params["center_override"] = (int(cx), int(cy))

auto_key = f"auto_{video_file}_{radius}_{center_radius}_{prominence}_{smooth_s}_{min_period}"
if src == "使用内置样例视频" and os.path.exists(SAMPLE) and auto_key not in st.session_state:
    bar = st.progress(0.0, "正在自动分析样例视频…")
    def cb(cur, total, msg):
        bar.progress(min(cur / max(total, 1), 1.0), f"{msg}：{cur} / {total} 帧")
    try:
        result = run_analysis(SAMPLE, params, cb)
        st.session_state["result"] = result
        st.session_state[auto_key] = True
        bar.empty()
    except Exception as e:
        bar.empty()
        st.warning(f"样例自动分析跳过：{e}，可点击下方按钮重试")

clicked = st.button("开始分析" if src == "上传自己的 mp4" else "重新分析",
                    type="primary", use_container_width=True)
if video_file and clicked:
    bar = st.progress(0.0, "准备中…")
    def cb2(cur, total, msg):
        bar.progress(min(cur / max(total, 1), 1.0), f"{msg}：{cur} / {total} 帧")
    try:
        result = run_analysis(video_file, params, cb2)
        st.session_state["result"] = result
        bar.empty()
        st.success("分析完成！")
    except Exception as e:
        bar.empty()
        st.error(f"分析失败：{e}")

# ---- 结果展示 ----
if "result" in st.session_state:
    r = st.session_state["result"]
    st.divider()
    st.subheader("📊 第三步：分析结果")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总吞吐圈数", r["total"])
    c2.metric("吞入（向内）", r["swallowed"])
    c3.metric("吐出（向外）", r["emitted"])
    c4.metric("识别圆心", f"({r['center'][0]}, {r['center'][1]})")

    st.markdown("")
    col_v, col_t = st.columns([3, 2])
    with col_v:
        st.markdown("**🎬 标注视频**（画面中实时计数）")
        st.video(r["output_video"])
    with col_t:
        st.markdown("**📋 事件明细**")
        df = pd.DataFrame(r["events"])
        st.dataframe(df, use_container_width=True, height=320)

    st.markdown("")
    st.markdown("**📈 中心亮度随时间变化曲线**（峰值即一次圆环吞入/吐出事件）")
    st.line_chart(pd.DataFrame({"中心亮度": r["normalized"]}), height=260)

    # 提供 CSV 下载
    if not df.empty:
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 下载事件明细 CSV", csv_bytes, "ring_events.csv", "text/csv")
