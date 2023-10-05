import whisper
import pyaudio
import wave


class AudioRecorder:
    """AudioRecorder class for recording voice from mic
    """
    def __init__(self, sample_rate=44100, channels=1, duration=5):
        self.output_file = "tmp.wav"
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration = duration

    def record_audio(self):
        audio = pyaudio.PyAudio()

        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024
            )

            print("Recording...")

            frames = []
            for _ in range(0, int(self.sample_rate / 1024 * self.duration)):
                data = stream.read(1024)
                frames.append(data)

            print("Finished recording.")

            with wave.open(self.output_file, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))

        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()



class SpeechToTextConverter:
    """Speech to text converter from whisper model
    """
    def __init__(self, model_size, audio="tmp.wav", gpu=True):
        self.model = whisper.load_model(model_size)
        self.audio = audio
        self.gpu = gpu


    def hear(self):
        print("Transcribing...")
        result = self.model.transcribe(self.audio, fp16=self.gpu)
        res = {
            "text": result["text"], 
            "language": result["language"]
        }

        return res



def call_ear(model_size, duration, gpu):
    """request ear for speech to text task

    Args:
        model_size (str): whisper model
        duration (int): voice record duration
        gpu (Bool): use gpu


    Returns:
        dict: transcribed text and detected language
    """
    ear = SpeechToTextConverter(model_size=model_size, gpu=gpu)
    recorder = AudioRecorder(duration=duration)
    recorder.record_audio()
    res = ear.hear()

    return res



if __name__ == "__main__":
    ear = SpeechToTextConverter(model_size="large", gpu=False)
    recorder = AudioRecorder(duration=5)
    recorder.record_audio()
    res = ear.hear()
    print("text: ", res["text"])
    print("language: ", res["language"])


