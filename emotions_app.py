# pip install streamlit numpy librosa transformers pyannote.audio paho-mqtt soundfile

import streamlit as st
import numpy as np
import librosa
import torch
import io
import os
import tempfile
import json
import time
import threading
import paho.mqtt.client as mqtt
from transformers import pipeline
from pyannote.audio import Pipeline

#connecting to the smart bulb via MQTT
MQTT_BROKER = "127.0.0.1" 
MQTT_PORT = 1883
ZIGBEE_BULB_TOPIC = "bulb's name" 

try:
    mqtt_client = mqtt.Client()
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
except Exception as e:
    mqtt_client = None
    print(f"MQTT Connection failed: {e}")


# Light Controller
# logic from the document 
# converting commands to the zigbee messages
class EmotionLightController:
    def __init__(self, client, topic):
        self.client = client
        self.topic = topic
        self.stop_event = threading.Event()
        self.current_thread = None

    def send_light(self, brightness, transition, kelvin=None, color_xy=None):
        if not self.client:
            return
            
        payload = {
            "state": "ON",
            "brightness": brightness,
            "transition": transition
        }
        
        # Convert Kelvin to Zigbee Mireds (micro reciprocal degrees) (1,000,000 / Kelvin)
        if kelvin:
            payload["color_temp"] = int(1000000 / kelvin) 
        elif color_xy:
            payload["color"] = {"x": color_xy[0], "y": color_xy[1]}
            
        self.client.publish(self.topic, json.dumps(payload))

    def stop_current_effect(self):
        self.stop_event.set()
        if self.current_thread:
            self.current_thread.join(timeout=1)
        self.stop_event.clear()

    # somple emotions: neutral, fear, disgust
    def neutral(self):
        self.stop_current_effect()
        self.send_light(kelvin=2900, brightness=110, transition=8)

    def fear(self):
        self.stop_current_effect()
        self.send_light(kelvin=3200, brightness=110, transition=10)

    def disgust(self):
        self.stop_current_effect()
        self.send_light(color_xy=(0.40, 0.36), brightness=90, transition=8)

    # dynamic emotions: surprise, happy, sad
    def surprise(self):
        self.stop_current_effect()
        def effect():
            self.send_light(color_xy=(0.44, 0.40), brightness=200, transition=1)
            time.sleep(1.5)
            if not self.stop_event.is_set():
                self.send_light(kelvin=3000, brightness=130, transition=5)
        self.current_thread = threading.Thread(target=effect)
        self.current_thread.start()

    def happy(self):
        self.stop_current_effect()
        def effect():
            while not self.stop_event.is_set():
                self.send_light(color_xy=(0.47, 0.43), brightness=145, transition=6)
                time.sleep(6)
                if self.stop_event.is_set(): break
                self.send_light(color_xy=(0.48, 0.44), brightness=160, transition=6)
                time.sleep(6)
        self.current_thread = threading.Thread(target=effect)
        self.current_thread.start()

    def sad(self):
        self.stop_current_effect()
        def effect():
            while not self.stop_event.is_set():
                self.send_light(kelvin=2000, brightness=70, transition=8)
                time.sleep(8)
                if self.stop_event.is_set(): break
                self.send_light(kelvin=2000, brightness=60, transition=8)
                time.sleep(8)
        self.current_thread = threading.Thread(target=effect)
        self.current_thread.start()


# Setting Streamlit UI
st.set_page_config(page_title="Emotional Ambient Light", layout="wide")

# Initialize state and controller
if "light_controller" not in st.session_state:
    st.session_state.light_controller = EmotionLightController(mqtt_client, ZIGBEE_BULB_TOPIC)
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neu"
    st.session_state.confidence = 0.0


UI_COLORS = {
    "hap": (255, 213, 79),   # Warm golden yellow
    "sad": (255, 140, 50),   # Warm orange
    "fea": (255, 230, 200),  # Soft warm white
    "dis": (180, 170, 160),  # Desaturated warm gray
    "sur": (255, 240, 180),  # Bright gold-white
    "neu": (255, 240, 220),  # Soft warm white
    "ang": (180, 170, 160)   # Fallback to disgust
}

st.sidebar.title("Controls")
enabled = st.sidebar.toggle("Enable physical lamp", value=True)
if not mqtt_client:
    st.sidebar.error("MQTT Broker disconnected. Lamp will not update.")

st.sidebar.markdown("---")
st.sidebar.subheader("Speaker Settings")
target_speaker = st.sidebar.radio(
    "Target Speaker",
    ["Speaker 0", "Speaker 1"],
    help="Select the patient/subject. Speaker 0 is usually the first person to talk."
)

st.sidebar.markdown("---")
uploaded_audio = st.sidebar.file_uploader("Upload audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])

# Loading Transformers 
@st.cache_resource
def load_emotion_model():
    return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=-1)

@st.cache_resource
def get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

@st.cache_resource
def load_diarization_model():
    hf_token = get_secret("HF_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "Missing HF_TOKEN. Add it in Streamlit Cloud secrets."
        )

    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token
    )
def render_ambient_light(rgb: tuple):
    r, g, b = rgb
    css = f"""
    <style>
        .stApp {{ background-color: #1a1a2e !important; }}
        .emotion-orb {{
            width: 300px; height: 300px; border-radius: 50%; margin: 2rem auto;
            background: radial-gradient(circle, rgba({r},{g},{b},0.9) 0%, rgba({r},{g},{b},0.4) 50%, transparent 70%);
            box-shadow: 0 0 80px rgba({r},{g},{b},0.5), 0 0 160px rgba({r},{g},{b},0.3);
            transition: all 2s ease-in-out;
        }}
    </style>
    <div class="emotion-orb"></div>
    """
    st.markdown(css, unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    current_rgb = UI_COLORS.get(st.session_state.current_emotion, UI_COLORS["neu"])
    render_ambient_light(current_rgb)

with col2:
    st.markdown("### Current State")
    emotion_names = {"hap": "Happy", "ang": "Angry", "sad": "Sad", "neu": "Neutral", "fea": "Fear", "dis": "Disgust", "sur": "Surprise"}
    st.metric("Emotion", emotion_names.get(st.session_state.current_emotion, st.session_state.current_emotion.capitalize()))
    st.metric("Confidence", f"{st.session_state.confidence:.0%}")


if uploaded_audio is not None:
    classifier = load_emotion_model()
    diarizer = load_diarization_model()
    
    with st.spinner("Separating voices and analyzing emotion..."):
        try:
            # Save temp file for Pyannote
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_audio.read())
                tmp_file_path = tmp_file.name

            # Load audio ourselves, so pyannote does not need AudioDecoder / torchcodec
            audio, sr = librosa.load(tmp_file_path, sr=16000, mono=True)

            waveform = torch.from_numpy(audio).float().unsqueeze(0)

            # Run diarization from waveform instead of file path
            diarization = diarizer({"waveform": waveform, "sample_rate": sr})

            os.remove(tmp_file_path) # Cleanup
            
            # Extract target speaker's audio
            target_speaker_id = "SPEAKER_00" if target_speaker == "Speaker 0" else "SPEAKER_01"
            target_audio_chunks = []
            
            # community-1 returns DiarizeOutput
            speaker_diarization = diarization.exclusive_speaker_diarization

            for turn, speaker in speaker_diarization:
                if speaker == target_speaker_id:
                    start_sample = int(turn.start * sr)
                    end_sample = int(turn.end * sr)
                    target_audio_chunks.extend(audio[start_sample:end_sample])
            
            if not target_audio_chunks:
                st.warning(f"Could not find {target_speaker} in this audio clip.")
            else:
                target_audio = np.array(target_audio_chunks)

                # Classify Emotion
                result = classifier({"raw": target_audio, "sampling_rate": 16000})
                top = result[0]
                
                emo_label = top["label"].lower()
                st.session_state.current_emotion = emo_label
                st.session_state.confidence = top["score"]
                
                # Trigger Physical Hardware only when the physical lamp is enabled.
                # Emotion analysis and the on-screen UI still work without the lamp.
                if enabled:
                    controller = st.session_state.light_controller

                    if emo_label == "hap":
                        controller.happy()
                    elif emo_label == "sad":
                        controller.sad()
                    elif emo_label in ["dis", "disgust", "ang"]:
                        controller.disgust()
                    elif emo_label == "fea":
                        controller.fear()
                    elif emo_label in ["sur", "surprise"]:
                        controller.surprise()
                    else:
                        controller.neutral()

            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing audio: {e}")
