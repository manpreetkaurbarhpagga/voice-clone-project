from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from TTS.api import TTS
from pydub import AudioSegment
from googletrans import Translator
import edge_tts
import uuid
import os
import sqlite3
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)


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

xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")



# TTS API

@app.post("/tts")
async def tts(text: str = Form(...), lang: str = Form("en")):

    tts = edge_tts.Communicate(text=text, voice="en-US-AriaNeural")
    out = f"outputs/{uuid.uuid4().hex}.mp3"
    await tts.save(out)

    return FileResponse(out, media_type="audio/mp3")



# VOICE CLONE

@app.post("/clone")
async def clone(text: str = Form(...), file: UploadFile = File(...), lang: str = Form("en")):

    in_path = f"uploads/{uuid.uuid4().hex}.wav"
    with open(in_path, "wb") as f:
        f.write(await file.read())

    out_wav = f"outputs/{uuid.uuid4().hex}.wav"

    xtts.tts_to_file(
        text=text,
        speaker_wav=in_path,
        language=lang,
        file_path=out_wav
    )

    final = f"outputs/{uuid.uuid4().hex}.mp3"
    AudioSegment.from_file(out_wav).export(final, format="mp3")

    return FileResponse(final, media_type="audio/mp3")



# STT

@app.post("/stt")
async def stt(file: UploadFile = File(...)):

    path = f"uploads/{uuid.uuid4().hex}.wav"
    with open(path, "wb") as f:
        f.write(await file.read())

    return {"text": "Speech converted (add model here)"}



# PDF

@app.post("/pdf")
async def pdf(file: UploadFile = File(...)):

    from PyPDF2 import PdfReader
    reader = PdfReader(file.file)

    text = ""
    for p in reader.pages:
        text += p.extract_text() or ""

    return {"text": text[:3000]}


# HISTORY

@app.get("/history")
def history():
    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    return {"data": cursor.fetchall()}
@app.get("/")
def root():
    return {"message": "VoiceForge AI Backend Running"}
# uvicorn main:app --reload