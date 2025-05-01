from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import torchaudio
import os
import traceback

app = FastAPI(title="STT Server with Whisper Tiny")

# Cetak backend yang tersedia
print("Available torchaudio backends:", torchaudio.list_audio_backends())

# Muat model dan processor dari Hugging Face
processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")
model.eval()

# Endpoint untuk transkripsi audio
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_path = None
    try:
        # Validasi file
        if not file.filename.endswith(('.wav', '.mp3', '.m4a')):
            raise HTTPException(status_code=400, detail="Unsupported file format. Use WAV, MP3, or M4A.")

        # Simpan file sementara
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # Debug: Periksa apakah file ada
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=500, detail="Temporary file not created")
        file_size = os.path.getsize(temp_path)
        print(f"File saved to {temp_path}, size: {file_size} bytes")

        # Muat audio
        print(f"Loading audio from {temp_path}")
        waveform, sample_rate = torchaudio.load(temp_path)
        print(f"Audio loaded: sample_rate={sample_rate}, shape={waveform.shape}")

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        waveform = waveform.squeeze(0)  # Ubah ke 1D

        # Proses audio dengan Whisper
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(inputs["input_features"])
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Hapus file sementara
        os.remove(temp_path)
        temp_path = None

        return JSONResponse(content={"transcription": transcription})
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Error details: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# Endpoint kesehatan
@app.get("/health")
async def health_check():
    return {"status": "healthy"}