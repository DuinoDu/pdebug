import copy
import os
import sys
from typing import List, Optional

from pdebug.otn import manager as otn_manager
from pdebug.utils.audio_utils import load_wave, mp3_to_wave, split_audio

import numpy as np
import tqdm
import typer


def merge_segment(seg1, seg2):
    seg = copy.deepcopy(seg1)
    seg["start"] = seg1["start"]
    seg["end"] = seg2["end"]
    seg["text"] = seg1["text"] + seg2["text"]
    seg["tokens"] = seg1["tokens"] + seg2["tokens"]
    return seg


@otn_manager.NODE.register(name="speech_to_text")
def main(
    audio_dir: str,
    ext: str = ".mp3",
    split_output: str = None,
    split_min_duration_sec: float = 1.0,
    output: str = None,
    cache: bool = False,
):
    """Convert speech to txt using whisper."""
    if cache and os.path.exists(output):
        print(f"{output} exists, skip")
        return output

    try:
        import whisper
    except ModuleNotFoundError as e:
        print("Install whisper by `pip install -U openai-whisper`")
        return

    if split_output:
        os.makedirs(split_output, exist_ok=True)
    os.makedirs(output, exist_ok=True)

    model = whisper.load_model("large")

    audio_files = [
        os.path.join(audio_dir, x)
        for x in sorted(os.listdir(audio_dir))
        if x.endswith(ext)
    ]
    t = tqdm.tqdm(total=len(audio_files))
    for audio_file in audio_files:
        t.update()
        result = model.transcribe(
            audio_file, language="zh", initial_prompt="以下是普通话的句子"
        )

        if split_output:
            # merge too short split
            new_segments = []
            for ind, segment in enumerate(result["segments"]):
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]

                if end_time - start_time < split_min_duration_sec:
                    if ind == 0:
                        if len(result["segments"]) == 1:
                            print("Drop too short audio: {audio_file}")
                            break
                        segment_2 = merge_segment(
                            segment, result["segments"][ind + 1]
                        )
                        result["segments"][ind + 1] = segment_2
                    elif ind == len(result["segments"]) - 1:
                        segment_2 = merge_segment(
                            result["segments"][ind - 1], segment
                        )
                        new_segments.pop(-1)
                        new_segments.append(segment_2)
                    else:
                        # merge to near segment
                        prev_segment = result["segments"][ind - 1]
                        next_segment = result["segments"][ind + 1]
                        if (
                            start_time - prev_segment["end"]
                            < next_segment["start"] - end_time
                        ):
                            segment_2 = merge_segment(prev_segment, segment)
                            new_segments.pop(-1)
                            new_segments.append(segment_2)
                        else:
                            segment_2 = merge_segment(segment, next_segment)
                            result["segments"][ind + 1] = segment_2
                else:
                    new_segments.append(segment)

            for ind, segment in enumerate(new_segments):
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]

                savename1 = os.path.join(
                    split_output,
                    os.path.basename(audio_file)[:-4] + f"_{ind:03d}.wav",
                )
                if not os.path.exists(savename1):
                    split_audio(audio_file, savename1, start_time, end_time)
                savename2 = os.path.join(
                    output,
                    os.path.basename(audio_file)[:-4] + f"_{ind:03d}.txt",
                )
                if not os.path.exists(savename2):
                    with open(savename2, "w") as fid:
                        fid.write(text)
        else:
            text = result["text"]
            savename = os.path.join(
                output, os.path.basename(audio_file)[:-4] + f".txt"
            )
            with open(savename, "w") as fid:
                fid.write(text)

    return output


def detect_voice_activity(
    wav_file: str, frame_ms: float = 0.01
) -> List[List[float]]:
    """Voice activity detection.

    Args:
        wav_file: input wave file.
        frame_ms: frame duration, 0.01 | 0.02 | 0.03
    """
    sound, fs = load_wave(wav_file)
    frame_len = int(round(fs * frame_ms))
    n = int(len(sound) / (2 * frame_len))

    import webrtcvad

    vad = webrtcvad.Vad(3)

    last_state = False
    find_voice = False
    talk_segments = []
    for frame_ind in range(n):
        slice_start = frame_ind * 2 * frame_len
        slice_end = (frame_ind + 1) * 2 * frame_len
        if vad.is_speech(sound[slice_start:slice_end], fs):
            find_voice = True
        else:
            find_voice = False
        start_time = slice_start / 2.0 / fs
        end_time = slice_end / 2.0 / fs

        if not last_state and find_voice:
            print(start_time, "start to speed")
            talk_segments.append([start_time])

        elif last_state and not find_voice:
            print(end_time, "stop")
            talk_segments[-1].append(end_time)

        last_state = find_voice

    if talk_segments and len(talk_segments[-1]) == 1:
        audio_end_time = len(sound) / 2.0 / fs
        talk_segments[-1].append(audio_end_time)

    return talk_segments


@otn_manager.NODE.register(name="split_speech")
def split_speech(
    audio_dir: str,
    ext: str = ".mp3",
    wave_dir: str = None,
    output: str = None,
):
    """Convert speech to txt using whisper.

    Or you can use speech_to_text directly.
    """
    if ext == ".mp3":
        if wave_dir is None:
            wave_dir = audio_dir + "_wave"
        os.makedirs(wave_dir, exist_ok=True)

    if not output:
        output = audio_dir + "_split"
    os.makedirs(output, exist_ok=True)

    audio_files = [
        os.path.join(audio_dir, x)
        for x in sorted(os.listdir(audio_dir))
        if x.endswith(ext)
    ]
    t = tqdm.tqdm(total=len(audio_files))
    for audio_file in audio_files:
        t.update()

        if audio_file.endswith(".mp3"):
            wave_file = os.path.join(
                wave_dir, os.path.basename(audio_file)[:-4] + ".wav"
            )
            if not os.path.exists(wave_file):
                mp3_to_wave(audio_file, wave_file, quiet=True)
            audio_file = wave_file

        talk_segments = detect_voice_activity(audio_file, 0.02)

        for ind, segment in enumerate(talk_segments):
            savename = os.path.join(
                output, os.path.basename(audio_file)[:-4] + f"_{ind:03d}.wav"
            )
            if os.path.exists(savename):
                os.remove(savename)
            assert len(segment) == 2
            cmd = f"ffmpeg -i {audio_file} -acodec copy -ss {segment[0]} -to {segment[1]} {savename}"
            os.system(cmd)

    return output


@otn_manager.NODE.register(name="mp3_to_wave")
def mp3_to_wave(
    audio_dir: str,
    output: str = None,
    cache: bool = False,
):
    """Convert mp3 file to wave file."""
    if not output:
        output = audio_dir + "_wave"
    os.makedirs(output, exist_ok=True)

    audio_files = [
        os.path.join(audio_dir, x)
        for x in sorted(os.listdir(audio_dir))
        if x.endswith(".mp3")
    ]
    print(f"Found {len(audio_files)} files.")

    if cache and len(audio_files) == len(os.listdir(output)):
        print(f"{output} exists, skip")
        return output

    t = tqdm.tqdm(total=len(audio_files))
    for audio_file in audio_files:
        t.update()
        wave_file = os.path.join(
            wave_dir, os.path.basename(audio_file)[:-4] + ".wav"
        )
        mp3_to_wave(audio_file, wave_file)

    return output


@otn_manager.NODE.register(name="text_to_pinyin")
def text_to_pinyin(
    path: str,
    vits_chinese_root: str = None,
    bert_output: str = None,
    output: str = None,
    cache: bool = False,
):
    """Get chinese pinyin from text."""
    if not output:
        output = audio_dir + "_pinyin"
    os.makedirs(output, exist_ok=True)

    if bert_output:
        os.makedirs(bert_output, exist_ok=True)

    text_files = [
        os.path.join(path, x)
        for x in sorted(os.listdir(path))
        if x.endswith(".txt")
    ]
    print(f"Found {len(text_files)} files.")

    if cache and len(text_files) == len(os.listdir(output)):
        print(f"{output} exists, skip")
        return output

    assert vits_chinese_root, "Please set `vits_chinese_root` path."
    sys.path.append(vits_chinese_root)
    import torch
    from vits_pinyin import VITS_PinYin

    # Download prosody_model.pt from https://github.com/PlayVoice/vits_chinese/releases/tag/v1.0
    # and put to `vits_chinese_root`/bert/prosody_model.pt
    bert_root = os.path.join(vits_chinese_root, "bert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tts_front = VITS_PinYin(bert_root, device)

    t = tqdm.tqdm(total=len(text_files))
    for text_file in text_files:
        t.update()
        text = open(text_file, "r").readlines()
        assert len(text) == 1
        text = text[0]
        phonemes, char_embeds = tts_front.chinese_to_phonemes(text)
        savename = os.path.join(output, os.path.basename(text_file))
        if not os.path.exists(savename):
            with open(savename, "w") as fid:
                fid.write(phonemes)

        if bert_output:
            savename2 = os.path.join(
                bert_output, os.path.basename(text_file)[:-4] + ".npy"
            )
            if not os.path.exists(savename2):
                np.save(savename2, char_embeds, allow_pickle=False)

    return output


@otn_manager.NODE.register(name="prepare_vits")
def prepare_vits(
    text_path: str,
    pinyin_path: str,
    wave_path: str = None,
    bert_path: str = None,
    vits_chinese_root: str = None,
    output: str = None,
    spec_output: str = None,
    cache: bool = False,
    topk_as_train: int = 100,
):
    """Get chinese pinyin from text."""
    # os.makedirs(output, exist_ok=True)

    text_files = [
        os.path.join(text_path, x)
        for x in sorted(os.listdir(text_path))
        if x.endswith(".txt")
    ]
    print(f"Found {len(text_files)} files.")

    if (
        cache
        and os.path.exists(output)
        and os.path.exists(output + "/train.txt")
    ):
        print(f"{output} exists, skip")
        return output

    if spec_output:
        os.makedirs(spec_output, exist_ok=True)

    if not output:
        output = "./filelists"
    os.makedirs(output, exist_ok=True)

    assert vits_chinese_root, "Please set `vits_chinese_root` path."
    sys.path.append(vits_chinese_root)
    import torch
    from bert import TTSProsody
    from bert.prosody_tool import is_chinese, pinyin_dict
    from mel_processing import spectrogram_torch
    from utils import get_hparams_from_file, load_wav_to_torch

    def log(info: str):
        with open(f"tmp_prepare_vits.log", "a", encoding="utf-8") as flog:
            print(info, file=flog)

    def get_spec(hps, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        assert (
            sampling_rate == hps.data.sampling_rate
        ), f"{sampling_rate} is not {hps.data.sampling_rate}"
        audio_norm = audio / hps.data.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        return spec

    config_file = os.path.join(vits_chinese_root, "configs/bert_vits.json")
    hps = get_hparams_from_file(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_root = os.path.join(vits_chinese_root, "bert")
    prosody = TTSProsody(bert_root, device)

    scrips = []

    t = tqdm.tqdm(total=len(text_files))
    for text_file in text_files:
        t.update()
        filename = os.path.basename(text_file)[:-4]
        wave_file = os.path.join(wave_path, f"{filename}.wav")
        pinyin_file = os.path.join(pinyin_path, f"{filename}.txt")
        bert_file = os.path.join(bert_path, f"{filename}.npy")

        spec_file = os.path.join(spec_output, f"{filename}.spec.pt")
        spec = get_spec(hps, wave_file)
        torch.save(spec, spec_file)

        phone_items_str = open(pinyin_file, "r").readlines()[0]
        scrips.append(f"{wave_file}|{spec_file}|{bert_file}|{phone_items_str}")

    fout = open(f"{output}/all.txt", "w", encoding="utf-8")
    for item in scrips:
        print(item, file=fout)
    fout.close()

    fout = open(f"{output}/train.txt", "w", encoding="utf-8")
    for item in scrips[:topk_as_train]:
        print(item, file=fout)
    fout.close()

    fout = open(f"{output}/valid.txt", "w", encoding="utf-8")
    for item in scrips[topk_as_train:]:
        print(item, file=fout)
    fout.close()

    return output


if __name__ == "__main__":
    typer.run(main)
