import os
import requests
import base64
import time
import librosa
import torch
from flask import Flask, redirect, render_template, request, session
from flask_socketio import SocketIO, send, emit, join_room, leave_room

import argparse
from loguru import logger
from six.moves import queue
# from queue import  Queue
import numpy as np

from inference import Inferencer
from scipy.io import wavfile

parser = argparse.ArgumentParser()
parser.add_argument('--lm_path', type=str,
                    default=None)
parser.add_argument('--huggingface_folder', type=str,
                    default="./model_repository/huggingface-hub")
parser.add_argument('--model_path', type=str,
                    default=None)
parser.add_argument('--use_language_model', action="store_true")
parser.add_argument('--device', type=int, default="cpu")
parser.add_argument('--port', type=int, default=1435)
args = parser.parse_args()

device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
# device = "cpu"
wav2vec2 = Inferencer(
    device=device,
    huggingface_folder=args.huggingface_folder,
    model_path=args.model_path,
    lm_path=args.lm_path,
    use_lm=args.use_language_model
)

asr_buff = queue.Queue()
asr_trans = queue.Queue()
# asr_buff = Queue()
# asr_trans = Queue()
text_buff = ""
data_chunks = []

app = Flask(__name__)
socketio = SocketIO(app)


STATIC_DIR = "static"
UPLOAD_DIR = "upload"
RECORD_DIR = "record"

os.makedirs(os.path.join(STATIC_DIR, UPLOAD_DIR), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, RECORD_DIR), exist_ok=True)

streaming_active = False
@app.route("/")
def index():
    return render_template(
        template_name_or_list="index_dev.html",
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
    # filename = time.strftime("%Y%m%d_%H%M%S")
    # filepath = os.path.join(STATIC_DIR, RECORD_DIR, filename + ".wav")
    # audio_file = open(filepath, "wb")
    # decode_string = base64.b64decode(data["audio_base64"].split(",")[1])
    # audio_file.write(decode_string)
    # logger.info("asr processing...")
    global streaming_active
    global text_buff
    global data_chunks
    streaming_active = True
    while streaming_active:
        asr_buff.put(data['arr'])
        logger.debug("asr_buff_ori:{}".format(len(data['arr'])))
        float_buff = np.array(np.frombuffer(asr_buff.get(), dtype=np.int16)/ 32767)
        logger.debug("asr_buff:{}".format(float_buff.shape))
        chunk = asr_buff.get()
        if chunk is None:
                return
        data_chunks.append(chunk)
        while True:
                try:
                    chunk = asr_buff.get(block=False)
                    if chunk is None:
                        return
                    data_chunks.append(chunk)
                except queue.Empty:
                    break
        full_data_chunks = b''.join(data_chunks)
        float_chunks = np.array(np.frombuffer(full_data_chunks, dtype=np.int16))
        wavfile.write("test.wav", 16000 ,float_chunks)
        # logger.debug("chunks:{}".format(float_chunks))
        # transcript = wav2vec2.run_with_buffer(float_buff)
        # if transcript == "" or transcript ==" ":
            # transcript = "nothing shit"
        transcript = wav2vec2.run("./cut_test.wav")
        import random
        a=random.randint(0, 100)
        transcript+=str(a)
        # transcript = "hihi {} \n".format(random.randint(0,100))
        # transcript = 'I'
        # transcript = transcript.lower() + str(a)
        # for text in transcript.split(" "):
        #     asr_trans.put(text)
        logger.success(f'transcript: {transcript}')
        emit('audio_to_client', {'transcript': transcript})
        # emit('audio_to_client', {'filepath': filepath, 'transcript': transcript})
        # if not asr_trans.empty():
        #     text_buff += asr_trans.get()
        #     emit('audio_to_client', {'transcript': text_buff})
        streaming_active = True


@socketio.on('stop_streaming')
def handle_stop_streaming():
    global streaming_active
    global text_buff
    
    streaming_active = False
    text_buff = ""
    data = []
    

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
    socketio.run(app, host="0.0.0.0", port=args.port, debug=False)
