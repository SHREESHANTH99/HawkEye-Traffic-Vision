"""
app.py — Gridlock: Traffic Violation Detection Dashboard
  Streamlit-based UI for real-time/batch violation detection.
  Supports: image upload, video upload, webcam feed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
import tempfile
import time

from src.detect import GridlockDetector
from src.violations import ViolationChecker, summarize_violations
from src.alpr import ALPRPipeline
from src.utils import annotate_frame, save_violation_to_csv, load_violations_log

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Gridlock — Traffic Violation Detection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-card: #12121a;
    --bg-sidebar: #0e0e16;
    --accent: #ff4757;
    --accent2: #ffa502;
    --accent3: #2ed573;
    --text-primary: #f1f2f6;
    --text-muted: #747d8c;
    --border: rgba(255,255,255,0.07);
    --glow-red: rgba(255, 71, 87, 0.25);
}

html, body, [data-testid="stApp"] {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}

/* Headers */
h1, h2, h3 { font-family: 'Inter', sans-serif !important; font-weight: 700 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #ff6b81) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px var(--glow-red) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px var(--glow-red) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
}

/* Status badges */
.badge-violation {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(255,71,87,0.15);
    color: #ff4757;
    border: 1px solid rgba(255,71,87,0.3);
}
.badge-ok {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(46,213,115,0.15);
    color: #2ed573;
    border: 1px solid rgba(46,213,115,0.3);
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1a0a0f 0%, #0a0a1f 50%, #0a1a0a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.hero h1 {
    margin: 0 !important;
    font-size: 2rem !important;
    background: linear-gradient(90deg, #ff4757, #ffa502, #2ed573);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: var(--text-muted);
    margin: 8px 0 0 0;
    font-size: 15px;
}

/* Live indicator */
.live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #ff4757;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
    margin-right: 6px;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255,71,87,0.7); }
    70% { box-shadow: 0 0 0 8px rgba(255,71,87,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,71,87,0); }
}

.section-header {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 20px 0 12px 0;
}

/* Sliders */
.stSlider { accent-color: #ff4757 !important; }

/* Selectbox */
.stSelectbox select {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #ff4757, #ffa502) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session state init ───────────────────────────────────────────────────────

if "violation_log" not in st.session_state:
    st.session_state.violation_log = []
if "total_processed" not in st.session_state:
    st.session_state.total_processed = 0
if "detector" not in st.session_state:
    st.session_state.detector = None
if "alpr" not in st.session_state:
    st.session_state.alpr = None

# ─── Load models (cached) ─────────────────────────────────────────────────────

@st.cache_resource
def load_detector(model_path: str, conf: float):
    return GridlockDetector(model_path=model_path, conf_threshold=conf)

@st.cache_resource
def load_alpr(plate_model_path: str = None):
    return ALPRPipeline(plate_model_path=plate_model_path)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-size:32px">🚦</div>
        <div style="font-size:18px; font-weight:800; letter-spacing:2px; color:#ff4757;">GRIDLOCK</div>
        <div style="font-size:11px; color:#747d8c; letter-spacing:1px;">VIOLATION DETECTION SYSTEM</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">⚙ Model Configuration</div>', unsafe_allow_html=True)

    # Model path selector
    model_options = {
        "YOLOv8 Nano (pretrained)": "yolov8n.pt",
        "YOLOv8 Small (pretrained)": "yolov8s.pt",
        "Custom Gridlock Model": "models/gridlock_v1/weights/best.pt",
    }
    model_choice = st.selectbox("Detection Model", list(model_options.keys()))
    model_path = model_options[model_choice]

    # Guard: warn if custom model weights don't exist yet
    if model_choice == "Custom Gridlock Model" and not Path(model_path).exists():
        st.warning(
            "Custom weights not found. Train first:\n"
            "`python src/train.py`\n\n"
            "Falling back to YOLOv8 Nano.",
            icon="⚠️",
        )
        model_path = "yolov8n.pt"

    conf_threshold = st.slider(
        "Confidence Threshold", min_value=0.10, max_value=0.90,
        value=0.45, step=0.05,
        help="Lower = more detections, higher = fewer false positives"
    )

    overlap_thresh = st.slider(
        "Overlap Threshold", min_value=0.10, max_value=0.80,
        value=0.30, step=0.05,
        help="How much a person bbox must overlap a vehicle bbox to be associated"
    )

    st.markdown('<div class="section-header">🚨 Violation Rules</div>', unsafe_allow_html=True)
    check_helmet  = st.checkbox("No Helmet Detection", value=True)
    check_triple  = st.checkbox("Triple Riding Detection", value=True)
    check_signal  = st.checkbox("Signal Jump Detection", value=False)
    triple_count  = st.number_input("Triple Riding Threshold (riders)", min_value=2, max_value=5, value=3)

    # Signal ROI inputs — shown only when Signal Jump is enabled
    signal_roi = None
    if check_signal:
        st.markdown(
            "<div style='font-size:11px;color:#747d8c;margin-top:4px;'>" 
            "Enter stop-line region in pixel coords (x1, y1, x2, y2):</div>",
            unsafe_allow_html=True,
        )
        roi_col1, roi_col2 = st.columns(2)
        with roi_col1:
            roi_x1 = st.number_input("x1", min_value=0, value=0,   step=10, label_visibility="visible")
            roi_y1 = st.number_input("y1", min_value=0, value=300,  step=10, label_visibility="visible")
        with roi_col2:
            roi_x2 = st.number_input("x2", min_value=0, value=640,  step=10, label_visibility="visible")
            roi_y2 = st.number_input("y2", min_value=0, value=400,  step=10, label_visibility="visible")
        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
            signal_roi = [int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2)]
        else:
            st.caption("x2 must be > x1 and y2 must be > y1")

    st.markdown('<div class="section-header">📸 Input Mode</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Select Input",
        ["Upload Image", "Upload Video"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    if st.button("🔄 Clear Violation Log", use_container_width=True):
        st.session_state.violation_log = []
        st.session_state.total_processed = 0
        st.rerun()

# ─── Main content ─────────────────────────────────────────────────────────────

# Hero banner
st.markdown("""
<div class="hero">
    <h1>🚦 Gridlock — Traffic Violation Detection</h1>
    <p>AI-powered real-time detection of traffic violations · Helmet compliance · Triple riding · Licence plate recognition</p>
</div>
""", unsafe_allow_html=True)

# KPI row
col1, col2, col3, col4 = st.columns(4)
violation_counts = summarize_violations(st.session_state.violation_log) if st.session_state.violation_log else {}
with col1:
    st.metric("🔍 Frames Processed", st.session_state.total_processed)
with col2:
    st.metric("⚠️ Total Violations", len(st.session_state.violation_log))
with col3:
    st.metric("🪖 No Helmet", violation_counts.get("NO_HELMET", 0))
with col4:
    st.metric("🏍️ Triple Riding", violation_counts.get("TRIPLE_RIDING", 0))

st.markdown("---")

# Main columns
left_col, right_col = st.columns([3, 2])

# ─── Left: Detection panel ────────────────────────────────────────────────────

with left_col:
    st.markdown("### 📷 Detection View")

    # Load models
    try:
        detector = load_detector(model_path, conf_threshold)
        alpr = load_alpr()
        checker = ViolationChecker(
            overlap_threshold=overlap_thresh,
            triple_riding_threshold=triple_count,
            signal_roi=signal_roi,   # None when disabled, [x1,y1,x2,y2] when set
        )
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # ── Image mode ────────────────────────────────────────────────────────────
    if input_mode == "Upload Image":
        uploaded = st.file_uploader(
            "Upload a traffic image",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("🔍 Running detection..."):
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                detections = detector.predict(frame)

                # Run violation checks
                violations = []
                if check_helmet or check_triple:
                    violations = checker.check(detections, frame_id=0, timestamp=ts)

                # ALPR on each violated vehicle
                for v in violations:
                    plate_bboxes, _ = alpr.detect_plates_in_vehicle(frame, v.bbox)
                    plate_text = "UNKNOWN"
                    if plate_bboxes:
                        result = alpr.read_plate_from_frame(frame, plate_bboxes[0])
                        if result:
                            plate_text = result.plate_number
                    v.plate_text = plate_text

                    # Save to session & CSV
                    st.session_state.violation_log.append(v)
                    save_violation_to_csv(v, plate_text)

                st.session_state.total_processed += 1

                # Annotate and display
                annotated = annotate_frame(frame, detections, violations)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_column_width=True, caption="Annotated Detection")

            # Detection summary
            if violations:
                st.error(f"🚨 **{len(violations)} violation(s) detected!**")
                for v in violations:
                    badge_color = {"NO_HELMET": "🔴", "TRIPLE_RIDING": "🟠", "SIGNAL_JUMP": "🟡"}.get(v.violation_type, "⚠️")
                    st.markdown(f"{badge_color} `{v.violation_type}` — Plate: **{v.plate_text}** — Conf: `{v.confidence:.2f}`")
            else:
                st.success("✅ No violations detected in this frame.")

            if detections:
                with st.expander(f"📋 All Detections ({len(detections)})", expanded=False):
                    det_data = [{"Class": d.cls, "Confidence": f"{d.conf:.2f}",
                                 "BBox": f"[{', '.join(str(round(x)) for x in d.bbox)}]"}
                                for d in detections]
                    st.dataframe(pd.DataFrame(det_data), use_container_width=True)

        else:
            st.markdown("""
            <div style="border: 2px dashed rgba(255,255,255,0.1); border-radius:12px;
                        padding:60px; text-align:center; color:#747d8c;">
                <div style="font-size:48px; margin-bottom:12px;">📸</div>
                <div style="font-size:16px; font-weight:500;">Upload a traffic image to start detection</div>
                <div style="font-size:13px; margin-top:6px;">Supports JPG, PNG, BMP</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Video mode ────────────────────────────────────────────────────────────
    elif input_mode == "Upload Video":
        uploaded = st.file_uploader(
            "Upload a traffic video",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )

        if uploaded:
            # Save to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            tfile.close()

            process_btn = st.button("▶ Process Video", use_container_width=True)

            if process_btn:
                frame_placeholder = st.empty()
                status_placeholder = st.empty()
                progress_bar = st.progress(0)

                cap = cv2.VideoCapture(tfile.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                frame_skip = st.session_state.get("frame_skip", 3)  # process every Nth frame

                for fid, frame, detections, ts in detector.stream_video(tfile.name):
                    if fid % frame_skip != 0:
                        continue

                    # Violations
                    violations = checker.check(detections, frame_id=fid, timestamp=ts)

                    # ALPR on violations
                    for v in violations:
                        pb, _ = alpr.detect_plates_in_vehicle(frame, v.bbox)
                        plate_text = "UNKNOWN"
                        if pb:
                            res = alpr.read_plate_from_frame(frame, pb[0])
                            if res:
                                plate_text = res.plate_number
                        v.plate_text = plate_text
                        st.session_state.violation_log.append(v)
                        save_violation_to_csv(v, plate_text)

                    # Annotate
                    annotated = annotate_frame(frame, detections, violations)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                    frame_placeholder.image(
                        annotated_rgb,
                        use_column_width=True,
                        caption=f"Frame {fid} | {len(violations)} violation(s)"
                    )

                    progress = min(fid / max(total_frames, 1), 1.0)
                    progress_bar.progress(progress)
                    status_placeholder.info(
                        f"⏳ Frame {fid}/{total_frames} | "
                        f"Violations this session: {len(st.session_state.violation_log)}"
                    )
                    st.session_state.total_processed += 1

                progress_bar.progress(1.0)
                status_placeholder.success("✅ Video processing complete!")
                os.unlink(tfile.name)

        else:
            st.markdown("""
            <div style="border: 2px dashed rgba(255,255,255,0.1); border-radius:12px;
                        padding:60px; text-align:center; color:#747d8c;">
                <div style="font-size:48px; margin-bottom:12px;">🎥</div>
                <div style="font-size:16px; font-weight:500;">Upload a traffic video clip</div>
                <div style="font-size:13px; margin-top:6px;">MP4, AVI, MOV, MKV — frames processed every 3rd frame</div>
            </div>
            """, unsafe_allow_html=True)

# ─── Right: Violation log ─────────────────────────────────────────────────────

with right_col:
    st.markdown("### 📋 Violation Log")

    if st.session_state.violation_log:
        # Build display dataframe
        log_data = []
        for v in reversed(st.session_state.violation_log[-50:]):  # show latest 50
            type_badge = {
                "NO_HELMET": "🔴 No Helmet",
                "TRIPLE_RIDING": "🟠 Triple Riding",
                "SIGNAL_JUMP": "🟡 Signal Jump",
            }.get(v.violation_type, v.violation_type)

            log_data.append({
                "Type": type_badge,
                "Plate": v.plate_text,
                "Confidence": f"{v.confidence:.2f}",
                "Frame": v.frame_id,
                "Time": v.timestamp,
            })

        df_log = pd.DataFrame(log_data)
        st.dataframe(df_log, use_container_width=True, height=350)

        # Export button
        csv_data = df_log.to_csv(index=False)
        st.download_button(
            label="⬇️ Export CSV",
            data=csv_data,
            file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Violation breakdown chart
        st.markdown("### 📊 Breakdown")
        if violation_counts:
            chart_data = pd.DataFrame.from_dict(
                violation_counts, orient="index", columns=["Count"]
            )
            chart_data = chart_data[chart_data["Count"] > 0]
            if not chart_data.empty:
                st.bar_chart(chart_data)

    else:
        st.markdown("""
        <div style="border: 1px solid rgba(255,255,255,0.07); border-radius:12px;
                    padding:40px; text-align:center; color:#747d8c; margin-top:16px;">
            <div style="font-size:36px; margin-bottom:12px;">📭</div>
            <div style="font-size:14px; font-weight:500;">No violations logged yet</div>
            <div style="font-size:12px; margin-top:6px;">Upload an image or video to begin</div>
        </div>
        """, unsafe_allow_html=True)

    # ── ALPR quick test ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔤 ALPR Plate Reader")
    plate_upload = st.file_uploader(
        "Upload a number plate crop",
        type=["jpg", "jpeg", "png"],
        key="plate_upload",
        label_visibility="collapsed",
    )
    if plate_upload:
        file_bytes = np.frombuffer(plate_upload.read(), np.uint8)
        plate_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        with st.spinner("Reading plate..."):
            result = alpr.read_plate_from_crop(plate_img)
        if result:
            st.image(
                cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB),
                caption="Plate Crop",
                width=250,
            )
            st.markdown(f"""
            <div style="background:#12121a; border:1px solid rgba(255,255,255,0.07);
                        border-radius:10px; padding:16px; margin-top:8px;">
                <div style="font-size:24px; font-weight:800; color:#ffa502; letter-spacing:3px;">
                    {result.plate_number}
                </div>
                <div style="font-size:12px; color:#747d8c; margin-top:4px;">
                    Raw OCR: <code>{result.raw_text}</code>
                </div>
                <div style="font-size:12px; color:#747d8c;">
                    Confidence: <code>{result.confidence:.2f}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#747d8c; font-size:12px; padding:8px 0 16px 0;">
    🚦 Gridlock Prototype &nbsp;·&nbsp; YOLOv8 + PaddleOCR + Streamlit &nbsp;·&nbsp;
    Built for Bengaluru Traffic Intelligence Research
</div>
""", unsafe_allow_html=True)
