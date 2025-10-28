import os
import wave

from pdebug.utils.env import FFMPEG_INSTALLED
from pdebug.utils.fileio import no_print

if FFMPEG_INSTALLED:
    import ffmpeg

__all__ = ["load_wave", "mp3_to_wave", "split_audio"]


def load_wave(file_name: str):
    """Load audio from file."""
    fp = wave.open(file_name, "rb")
    try:
        assert (
            fp.getnchannels() == 1
        ), "{0}: sound format is incorrect! Sound must be mono.".format(
            file_name
        )
        assert fp.getsampwidth() == 2, (
            "{0}: sound format is incorrect! "
            "Sample width of sound must be 2 bytes."
        ).format(file_name)
        assert fp.getframerate() in (8000, 16000, 32000), (
            "{0}: sound format is incorrect! "
            "Sampling frequency must be 8000 Hz, 16000 Hz or 32000 Hz."
        )
        sampling_frequency = fp.getframerate()
        sound_data = fp.readframes(fp.getnframes())
    finally:
        fp.close()
        del fp
    return sound_data, sampling_frequency


def mp3_to_wave(src: str, dst: str = None, quiet: bool = False):
    """Convert mp3 to wave using ffmpeg."""
    _dst = os.path.splitext(src)[0] + ".wav" if dst is None else dst
    assert _dst != src

    if not FFMPEG_INSTALLED:
        raise RuntimeError("Install ffmpeg by `pip3 install ffmpeg-python`")

    out, _ = (
        ffmpeg.input(src)
        .output(_dst, acodec="pcm_s16le", ar="16k")
        .overwrite_output()
        .run(capture_stdout=not quiet)
    )
    return dst


def split_audio(src: str, dst: str, start_time: float, end_time: float):
    """Convert mp3 to wave using ffmpeg."""
    if not FFMPEG_INSTALLED:
        raise RuntimeError("Install ffmpeg by `pip3 install ffmpeg-python`")

    #  f"ffmpeg -i {audio_file} -acodec copy -ss {segment[0]} -to {segment[1]} {savename}"

    with no_print():
        out, _ = (
            ffmpeg.input(src)
            .output(dst, acodec="copy", ss=start_time, to=end_time)
            .overwrite_output()
            .run(capture_stdout=True)
        )
    return dst
