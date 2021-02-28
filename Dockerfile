FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1 

# inference device
ENV INFERENCE_DEVICE='cuda:0'

#inference settings
ENV SAMPLE_RATE=16000
ENV BATCHED=TRUE
ENV BATCH_SIZE=3
ENV SEQUENCE_LENGTH=32000
ENV MIXED_PRECISION=TRUE

#output storage
ENV BUCKET=speakupai_inference_output_bucket
ENV AUDIO_OUTPUT_DIR='./output'

ENV PORT=80

EXPOSE ${PORT}

COPY . /app
