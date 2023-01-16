import av
import torch
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


device = "cpu"
if not hasattr(st, "classifier"):
    st.model = torch.hub.load("ultralytics/yolov5", "yolov5s", _verbose=False)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        flipped = img[:, ::-1, :]  # vision processing
        im_pil = Image.fromarray(flipped)  # model processing
        results = st.model(im_pil, size=112)
        bbox_img = np.array(results.render()[0])
        return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)
