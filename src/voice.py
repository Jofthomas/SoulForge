"""
Example of how to synthesize speech using the Coqui Studio API.
Streams the download/playback of the audio.

"""
import shutil
import subprocess
import requests
from typing import Iterator


def is_installed(lib_name: str) -> bool:
    lib = shutil.which(lib_name)
    if lib is None:
        return False
    return True


def play(audio: bytes) -> None:
    if not is_installed("ffplay"):
        message = (
            "ffplay from ffmpeg not found, necessary to play audio. "
            "On mac you can install it with 'brew install ffmpeg'. "
            "On linux and windows you can install it from https://ffmpeg.org/"
        )
        raise ValueError(message)
    args = ["ffplay", "-autoexit", "-", "-nodisp"]
    proc = subprocess.Popen(
        args=args,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = proc.communicate(input=audio)
    proc.poll()


def save(audio: bytes, filename: str) -> None:
    with open(filename, "wb") as f:
        f.write(audio)


def stream(audio_stream: Iterator[bytes]) -> bytes:
    if not is_installed("mpv"):
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


def tts(text: str, voice_id: str, model, api_token, language) -> Iterator[bytes]:
    if model == "xtts":
        url = "https://app.coqui.ai/api/v2/samples/xtts/render/?format=wav"
    else:
        url = "https://app.coqui.ai/api/v2/samples?format=wav"

    res = requests.post(
        url,
        json={"text": text, "voice_id": voice_id, "language": language},
        headers={"Authorization": f"Bearer {api_token}"},
    )
    for chunk in res.iter_content(chunk_size=2048):
        if chunk:
            yield chunk

def call_voice(text, voice_id, language, model = 'xtts', save_dest: str=None, api_token ='gY7ekqVyBhgv78D2DKTBj1XbCDgFplKzllxtp6h8yiJOVvzBxwUk9e2nHGIlpw4H'):
    '''
    text: String to transform to audio
    voice: ID of the voice to use for synthesis (example: 98d4af7d-aca0-4a70-a26e-4ca59023a248)
    model: either "v1" or "xtts"
    language: supported language codes are en,es,de,it,pt,fr,pl
    save_dest: optional file path to save audio wav file
    '''

    audio = stream(tts(text, voice_id, model, api_token, language))
    if save_dest:
        save(audio, save_dest)
