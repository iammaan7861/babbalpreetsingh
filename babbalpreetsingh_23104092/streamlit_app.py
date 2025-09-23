# streamlit_app.py
import streamlit as st
from pathlib import Path
import tempfile
import shutil
import time
import os

# Import your tracker class and helpers
from object_tracking import ObjectTracking
from supervision import get_video_frames_generator, VideoInfo, VideoSink

st.set_page_config(page_title="Vehicle Tracker", layout="wide")

st.title("Vehicle Tracking (YOLOv8 + ByteTrack)")
st.markdown("Upload a video and get a tracked output video. Uses your project's `ObjectTracking` class.")

# Sidebar options
st.sidebar.header("Options")
model_choice = st.sidebar.selectbox("Model to use (local path or 'auto')",
                                    options=["yolo/yolov8x.pt", "yolov8n.pt", "auto"])
show_logs = st.sidebar.checkbox("Show processing logs", value=True)

# Upload widget
uploaded_file = st.file_uploader("Upload video (mp4 / avi)", type=["mp4", "avi", "mov"], accept_multiple_files=False)

# Output area
output_video_file = None

def run_tracking(input_path: str, output_path: str):
    """Run tracking and return path to output_path (synchronous)."""
    # Create tracker instance - it will load the model (may download or require local file)
    # If user selected 'auto', use 'yolov8n.pt' which Ultralytics will auto-download
    model_path = model_choice if model_choice != "auto" else "yolov8n.pt"
    # The ObjectTracking class currently loads YOLO inside its __init__ (using fixed path).
    # We will temporarily override by setting environment or editing ObjectTracking if needed.
    # Here we assume ObjectTracking initialises YOLO from "yolo/yolov8x.pt" by default.
    # If you want to let user select model, modify object_tracking.py to accept model path.
    # For now we proceed with ObjectTracking as-is.
    obj = ObjectTracking(input_path, output_path)

    # Use generator and VideoSink similar to ObjectTracking.process() to allow progress bar
    video_info = VideoInfo.from_video_path(input_path)
    total_frames = video_info.total_frames if hasattr(video_info, "total_frames") else None

    progress_bar = st.progress(0)
    status_text = st.empty()

    with VideoSink(target_path=output_path, video_info=video_info) as sink:
        for idx, frame in enumerate(get_video_frames_generator(source_path=input_path)):
            result_frame = obj.callback(frame, idx)
            sink.write_frame(frame=result_frame)

            # update progress
            if total_frames:
                progress_bar.progress(min((idx + 1) / total_frames, 1.0))
                status_text.text(f"Processing frame {idx+1}/{total_frames}")
            else:
                # fallback: increment progress slowly
                progress_bar.progress(min((idx % 100) / 100, 0.99))
                status_text.text(f"Processing frame {idx+1}")

    progress_bar.progress(1.0)
    status_text.text("Done processing.")
    return output_path

if uploaded_file:
    # Save upload to a temp file
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.info(f"Saved upload to {input_path}")

    # Prepare output path
    output_filename = f"tracked_{uploaded_file.name}"
    output_path = os.path.join(tmp_dir, output_filename)

    if st.button("Start tracking"):
        t0 = time.time()
        with st.spinner("Running tracker — this may take some time."):
            try:
                out = run_tracking(input_path, output_path)
            except Exception as e:
                st.error(f"Tracking failed: {e}")
                if show_logs:
                    st.exception(e)
                # cleanup
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                t1 = time.time()
                st.success(f"Tracking finished in {int(t1-t0)}s — output: {out}")

                # show result video player
                st.video(out)

                # Download button
                with open(out, "rb") as f:
                    video_bytes = f.read()
                st.download_button("Download tracked video", data=video_bytes, file_name=output_filename, mime="video/mp4")

                # cleanup optional: keep tmp_dir for debugging, or remove:
                # shutil.rmtree(tmp_dir, ignore_errors=True)
else:
    st.info("Please upload a video file to start.")

st.markdown("---")
st.markdown("Notes: The model may be large (YOLOv8x). Consider using `yolov8n.pt` for testing. GPU speeds up processing a lot.")
