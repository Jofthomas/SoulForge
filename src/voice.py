"""
Example of how to synthesize speech using the Coqui Studio API.
Streams the download/playback of the audio.

"""
import shutil
import subprocess
import requests
from typing import Iterator

class Voice:
    '''
    model: either "v1" or "xtts"
    voice_id: ID of the voice to use for synthesis (example: 98d4af7d-aca0-4a70-a26e-4ca59023a248)
    language: supported language codes are "en","es","de","it","pt","fr","pl"
    save_dest: optional file path to save audio wav file
    '''
    def __init__(self, model, voice_id, language, save_dest=None, api_token='gY7ekqVyBhgv78D2DKTBj1XbCDgFplKzllxtp6h8yiJOVvzBxwUk9e2nHGIlpw4H'):
        self.voice_id = voice_id
        self.language = language
        self.model = model
        self.save_dest = save_dest
        self.api_token = api_token

    def is_installed(self, lib_name = "mpv") -> bool:
        lib = shutil.which(lib_name)
        if lib is None:
            return False
        return True

    def save(self, audio, filename) -> None:
        with open(filename, "wb") as f:
            f.write(audio)


    def stream(self, audio_stream: Iterator[bytes]) -> bytes:
        if not self.is_installed("mpv"):
            message = (
                "mpv not found, necessary to stream audio. "
                "On mac you can install it with 'brew install mpv'. "
                "On linux and windows you can install it from https://mpv.io/"
            )
            raise ValueError(message)

        mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
        mpv_process = subprocess.Popen(
            mpv_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        audio = b""

        for chunk in audio_stream:
            if chunk is not None:
                mpv_process.stdin.write(chunk)  # type: ignore
                mpv_process.stdin.flush()  # type: ignore
                audio += chunk

        if mpv_process.stdin:
            mpv_process.stdin.close()
        mpv_process.wait()

        return audio


    def tts(self, text: str) -> Iterator[bytes]:
        if self.model == "xtts":
            url = "https://app.coqui.ai/api/v2/samples/xtts/render/?format=wav"
        else:
            url = "https://app.coqui.ai/api/v2/samples?format=wav"

        res = requests.post(
            url,
            json={"text": text, "voice_id": self.voice_id, "language": self.language},
            headers={"Authorization": f"Bearer {self.api_token}"},
        )
        for chunk in res.iter_content(chunk_size=2048):
            if chunk:
                yield chunk

    def speak(self, text):
        """
        text: String to transform to audio
        """
        audio = self.stream(self.tts(text))
        if self.save_dest:
            self.save(audio, self.save_dest)
