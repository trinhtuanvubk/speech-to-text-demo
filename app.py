import os
import requests
import base64
import time
import librosa
import torch
from flask import Flask, redirect, render_template, request, session
from flask_socketio import SocketIO, send, emit, join_room, leave_room
from loguru import logger

# from infer import VietASR
from inference import Inferencer

app = Flask(__name__)
app.config["SECRET_KEY"] = "dangvansam"
socketio = SocketIO(app)


lm_path = "model_repository/language_model/4gram_small.arpa"


device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"

wav2vec2 = Inferencer(
        device = "cpu", 
        huggingface_folder = "./model_repository/huggingface-hub", 
        model_path = "./model_repository/w2v2_ckpt/best_model.tar",
        lm_path = lm_path,
        use_lm = True
)

STATIC_DIR = "static"
UPLOAD_DIR = "upload"
RECORD_DIR = "record"

os.makedirs(os.path.join(STATIC_DIR, UPLOAD_DIR), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, RECORD_DIR), exist_ok=True)

@app.route("/")
def index():
    return render_template(
        template_name_or_list="index.html",
        audio_path=None,
        async_mode=socketio.async_mode
    )


@socketio.on('connect')
def connected():
    logger.info("CONNECTED: " + request.sid)
    emit('to_client', {'text': request.sid})


@socketio.on('to_server')
def response_to_client(data):
    logger.info(data["text"])
    emit('to_client', {'text': len(data["text"].split())})


@socketio.on('audio_to_server')
def handle_audio_from_client(data):
    filename = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(STATIC_DIR, RECORD_DIR, filename + ".wav")
    audio_file = open(filepath, "wb")
    decode_string = base64.b64decode(data["audio_base64"].split(",")[1])
    audio_file.write(decode_string)
    logger.info("asr processing...")
    transcript = wav2vec2.run(filepath)
    transcript = transcript.lower()
    logger.success(f'transcript: {transcript}')
    emit('audio_to_client', {'filepath': filepath, 'transcript': transcript})


@app.route('/upload', methods=['POST', 'GET'])
def handle_upload():
    if request.method == "POST":
        _file = request.files['file']
        if _file.filename == '':
            return index()
        logger.info(f'file uploaded: {_file.filename}')
        filepath = os.path.join(STATIC_DIR, UPLOAD_DIR, _file.filename)
        _file.save(filepath)
        logger.info(f'saved file to: {filepath}')
        transcript = wav2vec2.run(filepath)
        transcript = transcript.lower()
        logger.info(f'transcript: {transcript}')
        return render_template(
            template_name_or_list='index.html',
            transcript=transcript,
            audiopath=filepath
        )
    else:
        return redirect("/")

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=1435,
                 ssl_context="adhoc", debug=False)
