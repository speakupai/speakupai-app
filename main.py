#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 18:41:52 2021

@author: Wilson Ye
"""
import base64
import json
import os
import io
import sys
import time
import uuid
from datetime import datetime

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.cuda.amp import autocast

from utils import data
from utils import gcp_storage
from models.hifi_gan import Generator
from models.wavenet import WaveNet
from hparams import hparams as hp


#init fastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

#inference environment variables
INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE") if os.environ.get("INFERENCE_DEVICE") else 'cuda:0'
SAMPLE_RATE = os.environ.get("SAMPLE_RATE") if os.environ.get("SAMPLE_RATE") else  hp.dsp.sample_rate
BATCHED = os.environ.get("BATCHED") if os.environ.get("BATCHED") else hp.inference.batched
BATCH_SIZE = os.environ.get("BATCH_SIZE") if os.environ.get("BATCH_SIZE") else hp.inference.batch_size
SEQUENCE_LENGTH = os.environ.get("SEQUENCE_LENGTH") if os.environ.get("SEQUENCE_LENGTH") else hp.inference.sequence_length
MIXED_PRECISION = os.environ.get("MIXED_PRECISION") if os.environ.get("MIXED_PRECISION") else hp.training.mixed_precision


#output storage
BUCKET = os.environ.get("BUCKET") if os.environ.get("BUCKET") else 'speakupai_inference_output_bucket'
AUDIO_OUTPUT_DIR = os.environ.get("AUDIO_OUTPUT_DIR") if os.environ.get("AUDIO_OUTPUT_DIR") else './output'

# Load checkpoint
checkpoint = torch.load('latest_checkpoint.pt', map_location=INFERENCE_DEVICE)

# Setup model
model = Generator(wavenet=WaveNet())
model.to(INFERENCE_DEVICE)
model.load_state_dict(checkpoint['generator_state_dict'])
model.eval()



@app.get("/")
async def index(request:Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(request: Request, audio_file: bytes=File(...)):
    
    response = {"success": False}

    if request.method == "POST":
        
        try:
            audio, _ = librosa.load(io.BytesIO(audio_file), sr=int(SAMPLE_RATE), mono=True)
                        
            #execute inference
            with torch.no_grad():
                x = audio
                target_length = len(x)
                x = torch.tensor(x).to(INFERENCE_DEVICE)

                x = data.preprocess_inference_data(x,
                                                         bool(BATCHED),
                                                         int(BATCH_SIZE),
                                                         int(SEQUENCE_LENGTH),
                                                         int(SAMPLE_RATE))
        
                with autocast(enabled=bool(MIXED_PRECISION)):
                    y = [model.inference(x_batch) for x_batch in x]
                
                #we noticed some tenors in y are not 2D and it caused torch.cat issues at postprocess_inference_data
                for i in range(len(y)):
                    if len(y[i].shape) < 2:
                        y[i] = y[i].unsqueeze(0)
                    
                y = data.postprocess_inference_data(y, bool(BATCHED), int(SAMPLE_RATE))
                y = y[:target_length].detach().cpu().numpy()
                output_file_name = 'output_audio_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.wav'
                output_path = os.path.join(AUDIO_OUTPUT_DIR, output_file_name)
                
                sf.write(output_path, y.astype(np.float32),
                         samplerate=int(SAMPLE_RATE))

                download_url = gcp_storage.upload_blob_and_generate_url(BUCKET, output_path, output_file_name)
                  
                # store download link
                response['download_link'] = download_url
                # indicate that the request was a success
                response["success"] = True

        except Exception as e:
                response["success"] = False
                response["error"] = str(e)
            
    # Return the data dictionary as a JSON response
    return response

