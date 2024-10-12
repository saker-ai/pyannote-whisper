import whisper
import os
from pyannote.audio import Pipeline
from pyannote.audio import Audio

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token="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")

#model = whisper.load_model("tiny.en")
#model = whisper.load_model("large-v3-turbo")
model = whisper.load_model("tiny")

audio_file = "data/output.wav"

diarization_result = pipeline(audio_file)

audio = Audio(sample_rate=16000, mono=True)
for segment, _, speaker in diarization_result.itertracks(yield_label=True):
    waveform, sample_rate = audio.crop(audio_file, segment)
    text = model.transcribe(waveform.squeeze().numpy())["text"]
    print(f"{segment.start:.2f}s {segment.end:.2f}s {speaker}: {text}")

