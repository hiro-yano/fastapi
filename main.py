from yt_dlp import YoutubeDL
import whisper
import google.generativeai as genai
from pydub import AudioSegment
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

# 設定: Gemini API Key（事前に取得し環境変数にセット）
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


class VideoRequest(BaseModel):
    url: str

def download_audio(youtube_url, output_file_name="audio", browser="chrome"):
    """YouTube動画の音声をダウンロード（ブラウザのクッキーを使用）"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_file_name,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'cookies': os.getenv("COOKIES") 
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    return output_file_name + ".mp3"


def transcribe_audio(audio_path):
    """Whisperで文字起こし"""
    model = whisper.load_model("small")  # モデルサイズは用途に応じて変更
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_text(text):
    """Gemini APIを使って要約"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(f"次の文章を要約してください:\n\n{text}")

    return response.text

@app.post("/process_video")
def process_video(request: VideoRequest):
    try:
        # 1. YouTube音声をダウンロード
        audio_file = download_audio(request.url)

        # 2. 文字起こし
        transcript = transcribe_audio(audio_file)

        # 3. 要約
        summary = summarize_text(transcript)

        return {"transcript": transcript, "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def start():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
