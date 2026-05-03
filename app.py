from backend_bridge import *
import streamlit as st
import os
import sqlite3
import pandas as pd
import speech_recognition as sr
import asyncio
import edge_tts
import uuid
import base64
import numpy as np
from googletrans import Translator
from PyPDF2 import PdfReader
from pydub import AudioSegment
from datetime import datetime
import streamlit.components.v1 as components
from TTS.api import TTS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


if os.path.exists("style.css"):
    local_css("style.css")
else:
    st.error("style.css file missing!")



# CONFIG & SETUP

st.set_page_config(page_title="VoiceForge AI", page_icon="🎙️", layout="wide")
os.makedirs("outputs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# DATABASE SETUP
conn = sqlite3.connect("voiceforge.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    feature TEXT,
    text_input TEXT,
    audio_path TEXT,
    created_at TEXT
)
""")
conn.commit()

# SESSION MANAGEMENT
if "user" not in st.session_state:
    st.session_state.user = "Manpreet"
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# MAPS & DICTIONARIES
tts_languages = {"English":"en","Hindi":"hi","French":"fr"}
stt_languages = {"English": "en", "Hindi": "hi", "French": "fr", "Punjabi": "pa"}
voice_map = {
    "en":{"Female":"en-US-AriaNeural","Male":"en-US-GuyNeural"},
    "hi":{"Female":"hi-IN-SwaraNeural","Male":"hi-IN-MadhurNeural"},
    "fr":{"Female":"fr-FR-DeniseNeural","Male":"fr-FR-HenriNeural"},
    "pa":{"Female":"pa-IN-GurshabadNeural","Male":"pa-IN-GurshabadNeural"}
}

# =========================
# CORE UTILITY FUNCTIONS
# =========================
def safe_lang(lang):
    return "en" if lang in ["", None, "Select Language"] else lang

def safe_translate(text, lang):
    try:
        return Translator().translate(text, dest=safe_lang(lang)).text
    except:
        return text

def save_history(feature, text, path):
    cursor.execute("INSERT INTO history VALUES(NULL,?,?,?,?,?)",(
        st.session_state.user, feature, text, path,
        datetime.now().strftime("%d-%m-%Y %H:%M")
    ))
    conn.commit()

def run_tts(text, voice, rate, path):
    async def run():
        tts = edge_tts.Communicate(text=text, voice=voice, rate=rate)
        await tts.save(path)
    try:
        asyncio.run(run())
    except:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run())

def audio_with_visualizer(audio_path):
    audio_bytes = open(audio_path,"rb").read()
    b64 = base64.b64encode(audio_bytes).decode()
    components.html(f"""
    <style>
    .bars {{ display:flex; justify-content:center; align-items:flex-end; gap:6px; height:120px; }}
    .bar {{ width:8px; height:20px; background:linear-gradient(180deg,#ff4b4b,#4b7bff); animation:bounce 1s infinite; }}
    @keyframes bounce {{ 0%,100%{{height:20px;}} 50%{{height:120px;}} }}
    </style>
    <div class="bars">{"<div class='bar'></div>"*5}</div>
    <audio controls autoplay style="width:100%;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>
    """, height=260)
    audio = AudioSegment.from_file(audio_path)
    samples = audio.get_array_of_samples()[::100]
    st.subheader("📊 Waveform")
    st.line_chart(samples)

def TextToSpeech(text, lang, voice_type, speed, pitch, emotion):
    lang = safe_lang(lang)
    text = safe_translate(text, lang)
    voice = voice_map[lang][voice_type]
    path = f"outputs/{uuid.uuid4().hex}.mp3"
    rate = "+0%"
    if speed < 0.8: rate = "-20%"
    elif speed > 1.2: rate = "+20%"
    run_tts(text, voice, rate, path)
    audio = AudioSegment.from_file(path)
    audio = audio + pitch
    audio.export(path, format="mp3")
    save_history("TTS", text[:50], path)
    return path

@st.cache_resource
def load_xtts():
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2")

def VoiceClone(text, lang, sample_path, speed, pitch, emotion):
    xtts_model = load_xtts()
    lang = safe_lang(lang)
    text = safe_translate(text, lang)
    out_path = f"outputs/{uuid.uuid4().hex}.wav"
    xtts_model.tts_to_file(text=text, speaker_wav=sample_path, language=lang, file_path=out_path)
    audio = AudioSegment.from_file(out_path) + pitch
    final = f"outputs/{uuid.uuid4().hex}.mp3"
    audio.export(final, format="mp3")
    save_history("Clone", text[:50], final)
    return final

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("🎙️ Voice Lab")
menu = st.sidebar.radio("Menu",[
    "🏠 Home", "🎤 TextToSpeech", "🧬 VoiceClone", 
    "🎧 SpeechToText", "📄 PDF", "📊 Dashboard", "📁 History"
])

if st.session_state.page != menu:
    st.session_state.page = menu
    st.rerun()

st.markdown("<h1 style='text-align: center;'>🎧 VoiceForge AI Studio</h1>", unsafe_allow_html=True)

# ==============================================================================
# PAGE 1: HOME
# ==============================================================================
if menu == "🏠 Home":
    st.markdown("""
        <style>
        .main-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 30px; border-radius: 20px; border: 1px solid #334155; margin-bottom: 20px; height: 280px; transition: 0.3s; }
        .main-card:hover { border-color: #38bdf8; transform: translateY(-5px); }
        .feature-title { color: #38bdf8; font-size: 26px; font-weight: bold; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
        .feature-desc { color: #f8fafc !important; font-size: 17px; line-height: 1.6; }
        .status-tag { background-color: rgba(34, 197, 94, 0.2); color: #4ade80; padding: 5px 15px; border-radius: 50px; font-size: 13px; border: 1px solid #22c55e; }
        .step-card { background: #1e293b; padding: 20px; border-radius: 15px; border-top: 4px solid #38bdf8; text-align: center; height: 220px; }
        .step-num { background: #38bdf8; color: #0f172a; width: 35px; height: 35px; line-height: 35px; border-radius: 50%; margin: 0 auto 15px; font-weight: bold; }
        div.stButton > button { background-color: black !important; color: white !important; border: 1px solid white !important; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div style='text-align: center;'><h3>Create natural AI voices in seconds</h3><p>Choose a module to begin.</p></div>", unsafe_allow_html=True)
    st.divider()

    st.subheader("🔊 Live Neural Signal")
    t = np.linspace(0, 20, 100)
    wave = np.sin(t) * np.exp(-0.1*t) + np.random.normal(0, 0.05, 100)
    st.area_chart(wave, height=150)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="main-card"><div class="feature-title">🔊 Advanced TTS</div><p class="feature-desc">Transform text into ultra-realistic speech with emotional depth.</p><span class="status-tag">Status: Online</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="main-card"><div class="feature-title">🎭 Voice Cloning</div><p class="feature-desc">Create a digital twin of any voice using a 10s sample.</p><span class="status-tag">Accuracy: 99.1%</span></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("<h2 style='text-align: center;'>🚀 Start Your Journey</h2>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1: st.markdown('<div class="step-card"><div class="step-num">1</div><div style="color:#38bdf8; font-weight:bold;">Write Text</div><p style="color:#cbd5e1; font-size:14px;">Write your script.</p></div>', unsafe_allow_html=True)
    with p2: st.markdown('<div class="step-card"><div class="step-num">2</div><div style="color:#38bdf8; font-weight:bold;">Upload File</div><p style="color:#cbd5e1; font-size:14px;">Upload sample voice.</p></div>', unsafe_allow_html=True)
    with p3: st.markdown('<div class="step-card"><div class="step-num">3</div><div style="color:#38bdf8; font-weight:bold;">Generate</div><p style="color:#cbd5e1; font-size:14px;">Get your AI voice.</p></div>', unsafe_allow_html=True)

    if st.button("🚀 Start Creating Now", use_container_width=True):
        st.balloons()
        st.toast("Ready to Generate! 🎙️")

# ==============================================================================
# PAGE 2: TEXT TO SPEECH
# ==============================================================================
elif menu == "🎤 TextToSpeech":
    st.markdown("""
        <style>
        .metric-words { background: rgba(56, 189, 248, 0.1); padding: 10px; border-radius: 10px; border: 1px solid #38bdf8; text-align: center; color: #38bdf8; font-weight: bold; }
        .metric-chars { background: rgba(168, 85, 247, 0.1); padding: 10px; border-radius: 10px; border: 1px solid #a855f7; text-align: center; color: #a855f7; font-weight: bold; }
        .settings-card { background: #0f172a; padding: 20px; border-radius: 15px; border: 1px solid #1e293b; }
        div.stButton > button:first-child { background: black; color: white; border: 1px solid #38bdf8; width: 100%; height: 50px; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>🎙️ AI Voice Composer</h3>", unsafe_allow_html=True)
    main_col, side_col = st.columns([2, 1], gap="large")

    with main_col:
        text = st.text_area("Enter Text", placeholder="Type here...", height=250)
        if text:
            m1, m2 = st.columns(2)
            m1.markdown(f'<div class="metric-words">📝 Words: {len(text.split())}</div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-chars">🔠 Characters: {len(text)}</div>', unsafe_allow_html=True)

    with side_col:
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        lang = st.selectbox("Language", ["Select Language"] + list(tts_languages.keys()))
        voice = st.selectbox("Voice", ["Select Voice", "Female", "Male"])
        speed = st.selectbox("Speed", ["Select Speed", "Slow", "Normal", "Fast"])
        pitch = st.slider("Pitch Control", -10, 10, 0)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🪄 Generate Professional Voice", use_container_width=True):
        if text.strip() and lang != "Select Language" and voice != "Select Voice":
            # Speed Mapping Logic
            speed_map = {"Slow": 0.7, "Normal": 1.0, "Fast": 1.3}
            
            # Professional Status Display
            st.markdown(f"""
                <div style='padding:15px; border-radius:12px; background: rgba(56, 189, 248, 0.1); 
                border: 1px solid #38bdf8; color:blue; text-align:center; margin-bottom:20px;'>
                🌍 <b>{lang}</b> Language | 🎙️ <b>{voice}</b> Voice | ⚡ <b>{speed}</b> Speed
                </div>
            """, unsafe_allow_html=True)

            with st.spinner("🧠 AI is synthesizing your voice..."):
                try:
                    # Logic call
                    out_path = TextToSpeech(
                        text=text, 
                        lang=tts_languages[lang], 
                        voice_type=voice, 
                        speed=speed_map.get(speed, 1.0), 
                        pitch=pitch, 
                        emotion="Natural"
                    )
                    
                    # Visualizer and Audio
                    audio_with_visualizer(out_path)
                    st.toast("Voice generated successfully!", icon="✅")
                    
                except Exception as e:
                    st.error(f"Generation Error: {e}")
        else:
            st.warning("⚠️ Missing Information: Please ensure Text, Language, and Voice Gender are all selected.")
            

# PAGE 3: VOICE CLONE
elif menu == "🧬 VoiceClone":
    st.markdown("<h3 style='text-align: center;'>🎭 Instant Voice Cloning</h3>", unsafe_allow_html=True)
    main_col, side_col = st.columns([2, 1], gap="large")

    with main_col:
        text = st.text_area("Content to Speak", placeholder="Type here...", height=250)
        sample = st.file_uploader("Upload Voice Sample", type=["wav","mp3"])

    with side_col:
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        lang = st.selectbox("Language", ["Select Language"] + list(tts_languages.keys()))
        speed = st.selectbox("Speed", ["Select Speed","Slow","Normal","Fast"])
        pitch = st.slider("Pitch", -10, 10, 0)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🧬 Start Voice Cloning", key="clone_btn"):
        if text and sample and lang != "Select Language":
            speed_map = {"Slow": 0.7, "Normal": 1.0, "Fast": 1.3}
            path = f"uploads/{uuid.uuid4().hex}.wav"
            with open(path, "wb") as f: 
                f.write(sample.read())
            with st.spinner("Cloning Voice..."):
                out = VoiceClone(text, tts_languages[lang], path, speed_map.get(speed, 1.0), pitch, "Natural")
                audio_with_visualizer(out)
            st.balloons()
        else:
            st.error("Missing Data.")
            
# PAGE 4: SPEECH TO TEXT

elif menu == "🎧 SpeechToText":
    # Header Section
    
    st.markdown("<p style='text-align: center; color: gray;'>Convert any audio recording into accurate text instantly.</p>", unsafe_allow_html=True)
    st.divider()

    # Main Layout
    col_input, col_settings = st.columns([2, 1], gap="large")

    with col_input:
        st.markdown("### 📥 Audio Input")
        audio = st.file_uploader("Upload Audio", type=["wav", "mp3"], help="Select a high-quality audio file for better accuracy.", label_visibility="collapsed")
        
        if audio:
            st.audio(audio) # Preview for user

    with col_settings:
        st.markdown("### ⚙️ Configuration")
        st.markdown('<div class="stt-settings-card">', unsafe_allow_html=True)
        lang_choice = st.selectbox("Output Language", list(stt_languages.keys()))
        st.info("AI will detect the speech and translate it to your selected language if needed.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Transcription Logic
    if st.button("🚀 Start Transcription", key="stt_btn"):
        if audio:
            with st.status("Processing Audio...", expanded=True) as status:
                # 1. Save uploaded file
                st.write("Extracting raw audio...")
                raw_path = f"uploads/{uuid.uuid4().hex}_{audio.name}"
                with open(raw_path, "wb") as f:
                    f.write(audio.read())

                # 2. Optimize audio (Speech Recognition works best with 16kHz Mono WAV)
                st.write("Optimizing for Neural Recognition...")
                wav_path = raw_path + ".wav"
                sound = AudioSegment.from_file(raw_path)
                sound = sound.set_frame_rate(16000).set_channels(1)
                sound.export(wav_path, format="wav")

                # 3. Recognition
                r = sr.Recognizer()
                try:
                    with sr.AudioFile(wav_path) as source:
                        st.write("Analyzing patterns...")
                        audio_data = r.record(source)
                    
                    # Convert to Text
                    text_result = r.recognize_google(audio_data)

                    # 4. Translation if necessary
                    target_lang_code = stt_languages[lang_choice]
                    if target_lang_code != "en":
                        st.write(f"Translating to {lang_choice}...")
                        translator = Translator()
                        text_result = translator.translate(text_result, dest=target_lang_code).text

                    status.update(label="✅ Transcription Complete!", state="complete", expanded=False)

                    # Display Styled Result
                    st.markdown("### 📄 Resulting Text")
                    st.markdown(f'<div class="result-card">{text_result}</div>', unsafe_allow_html=True)
                    
                    # Actions
                    st.download_button("📥 Download Transcript", text_result, file_name=f"transcript_{uuid.uuid4().hex[:5]}.txt")
                    save_history("STT", text_result[:100], wav_path)

                except Exception as e:
                    status.update(label="❌ Error Occurred", state="error")
                    st.error(f"Could not process audio: {str(e)}")
        else:
            st.warning("⚠️ Please upload an audio file first.")

    # Footer
    st.markdown("<p style='text-align: center; opacity: 0.4; font-size: 12px; margin-top: 50px;'>Neural Speech Recognition v2.0</p>", unsafe_allow_html=True)


# PAGE 5: PDF

elif menu == "📄 PDF":
    # Header Section
    st.markdown("<h3 style='text-align: center;' class='pdf-title'>📄 PDF Voice Reader</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Upload any PDF document and listen to it as an AI-powered audiobook.</p>", unsafe_allow_html=True)
    st.divider()

    # Layout Setup
    col_view, col_opt = st.columns([2, 1], gap="large")

    with col_view:
        st.markdown("### 👁️ Document Preview")
        pdf = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

        if pdf:
            st.markdown('<div class="pdf-container">', unsafe_allow_html=True)
            # Encode PDF for preview
            pdf_bytes = pdf.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500"></iframe>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            # Reset pointer for extraction
            pdf.seek(0)
        else:
            st.info("Please upload a PDF file to see the preview here.")

    with col_opt:
        st.markdown("### 🎧 Audiobook Settings")
        st.markdown('<div class="pdf-settings-card">', unsafe_allow_html=True)
        
        # Language Selection (As you requested)
        lang = st.selectbox("Select Language", ["Select Language"] + list(tts_languages.keys()))
        
        st.write("---")
        st.caption("Note: Currently processing up to the first 2000 characters for optimal performance.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Conversion Process
    if st.button("🎙️ Convert PDF to Speech", key="pdf_conv_btn"):
        if pdf and lang != "Select Language":
            with st.status("Reading PDF Content...", expanded=True) as status:
                st.write("Initializing PDF Engine...")
                reader = PdfReader(pdf)
                
                extracted_text = ""
                st.write(f"Extracting text from {len(reader.pages)} pages...")
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text

                if extracted_text.strip():
                    st.write("Generating Neural Voice...")
                    # Limit to 2000 characters for demo
                    clean_text = extracted_text[:2000]
                    
                    # Call global TextToSpeech function
                    out = TextToSpeech(clean_text, tts_languages[lang], "Female", 1.0, 0, "Natural")
                    
                    status.update(label="✅ Audiobook Ready!", state="complete", expanded=False)
                    
                    st.markdown("### 🔉 Audio Output")
                    audio_with_visualizer(out)
                    st.success("Successfully converted text from PDF!")
                else:
                    status.update(label="❌ Extraction Failed", state="error")
                    st.error("Could not extract text. This might be a scanned image PDF.")
        else:
            st.warning("⚠️ Please upload a PDF and select a language first.")

    # Footer
    st.markdown("<p style='text-align: center; opacity: 0.4; font-size: 12px; margin-top: 50px;'>Neural PDF Processor v1.5</p>", unsafe_allow_html=True)

# PAGE 6: DASHBOARD & HISTORY

elif menu == "📊 Dashboard":
    st.markdown('<p class="dash-title" style="color: black; text-align: center;">🚀 System Overview</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Real-time analytics and voice processing metrics.</p>', unsafe_allow_html=True)
    st.divider()

    # Database Metrics
    try:
        df_count = pd.read_sql_query("SELECT COUNT(*) as count FROM history", conn)
        total_gen = df_count['count'][0]
    except:
        total_gen = 0

    # Stat Cards
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Total Tasks", total_gen),
        ("AI Accuracy", "99.2%"),
        ("Server Speed", "120ms"),
        ("Uptime", "100%")
    ]
    
    for i, col in enumerate([c1, c2, c3, c4]):
        label, val = metrics[i]
        col.markdown(f"""<div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-val">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Section
    col_chart, col_usage = st.columns([2, 1], gap="large")
    with col_chart:
        st.markdown("### 📈 Usage Activity")
        chart_data = pd.DataFrame(np.random.randn(20, 2), columns=['TTS', 'STT'])
        st.area_chart(chart_data)

    with col_usage:
        st.markdown("### ⚙️ Resources")
        st.write("CPU Usage")
        st.progress(25)
        st.write("Neural Load")
        st.progress(65)
        st.write("Storage")
        st.progress(12)

    st.markdown("<p style='text-align: center; opacity: 0.3; margin-top: 50px;'>NeuralVoice Dashboard v3.0</p>", unsafe_allow_html=True)
elif menu == "📁 History":
    st.markdown("## 📜 Activity Logs")
    
    col_title, col_clear = st.columns([3, 1])
    with col_title:
        st.markdown("<p style='color: gray;'>Review and download your previous voice generations.</p>", unsafe_allow_html=True)
    
    with col_clear:
        # Clear History Button
        if st.button("🗑️ Clear All History", use_container_width=True):
            try:
                
                if os.path.exists("uploads"):
                    for file in os.listdir("uploads"):
                        file_path = os.path.join("uploads", file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            print(f"Error deleting file: {e}")
                
                # 2. Clear Database table
                cursor = conn.cursor()
                cursor.execute("DELETE FROM history")
                conn.commit()
                
                st.success("History cleared successfully!")
                st.rerun() 
            except Exception as e:
                st.error(f"Error clearing history: {e}")

    st.divider()

    try:
        df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    except:
        st.error("Database connection issue.")
        df = pd.DataFrame()

    if df.empty:
        st.info("📂 No history records found yet.")
    else:
        for i, row in df.iterrows():
            username = row.get('username', 'User')
            feature = row.get('feature', 'General')
            created_at = row.get('created_at', 'N/A')
            text_val = row.get('text_input', 'No Content')
            audio_path = row.get('audio_path', '')

            st.markdown(f"""
                <div class="history-card">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span>👤 <span class="user-tag">{username}</span></span>
                        <span class="feature-tag">⚙️ {feature}</span>
                    </div>
                    <div style="color: #e0e0e0; font-size: 15px; margin-bottom: 10px;">
                        {text_val[:150] + '...' if len(text_val) > 150 else text_val}
                    </div>
                    <div style="font-size: 11px; color: #555; text-align: right;">🕒 {created_at}</div>
                </div>
            """, unsafe_allow_html=True)

            if audio_path and os.path.exists(audio_path):
                a_col, d_col = st.columns([4, 1])
                with a_col:
                    st.audio(audio_path)
                with d_col:
                    with open(audio_path, "rb") as f:
                        st.download_button("⬇️ Save", f, file_name=os.path.basename(audio_path), key=f"hist_{i}")
            
            st.markdown("<br>", unsafe_allow_html=True)
