import collections
import contextlib
import subprocess
import tempfile
import time
import wave
import shutil
from pathlib import Path
from multiprocessing import Manager, Pool, Process, Queue, cpu_count
from typing import List, Tuple

import cv2
import librosa
import numpy as np
import webrtcvad
from tqdm import tqdm
from typer import Typer

from infer import ArcFacePro3

app = Typer()


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    parts = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                parts.append([ring_buffer[0][0].timestamp, 0])
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                parts[-1][1] = frame.timestamp + frame.duration
                triggered = False
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        parts[-1][1] = frame.timestamp + frame.duration
    return parts


@app.command()
def process_dir(d: Path,
                output_dir: Path,
                connect_factor: float = 5,
                min_duration: float = None,
                max_duration: float = None,
                rotate_k: int = 0,
                processes: int = cpu_count()):
    """
    处理一个文件下的所有视频（*.MP4,*.MOV），生成数据集到output_dir
    :param max_duration:
    :param min_duration:
    :param d:
    :param output_dir:
    :param connect_factor: float, optional a 越大，越容易连接，生成的视频越长
    :param rotate_k 顺时针旋转90度的次数，默认不旋转，只能是正数
    :param processes: 进程数
    :return:
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    video_files = list(d.glob("*.MP4")) + list(d.glob("*.mp4")) + list(d.glob("*.MOV")) + list(d.glob("*.mov"))
    print(f"共有{len(video_files)}个视频")
    pool = Pool(processes=processes)
    queue = Manager().Queue()
    queue.put([0, len(video_files)])  # （已经处理完的视频个数，所有的视频个数），用于显示进度
    for video_file in video_files:
        pool.apply_async(func=process_video,
                         kwds=dict(video_file=video_file,
                                   output_dir=output_dir,
                                   connect_factor=connect_factor,
                                   min_duration=min_duration,
                                   max_duration=max_duration,
                                   rotate_k=rotate_k,
                                   queue=queue),
                         error_callback=lambda error: print(error))
    pool.close()
    pool.join()


@app.command()
def process_video(video_file: Path,
                  output_dir: Path,
                  connect_factor: float = 5,
                  min_duration: float = None,
                  max_duration: float = None,
                  rotate_k: int = 0,
                  queue: str = None):
    """

    :param video_file:
    :param output_dir:
    :param connect_factor:
    :param min_duration:
    :param max_duration:
    :param rotate_k:
    :param queue:
    :return:
    """
    global arc_face_pro_3
    print(f"开始处理{video_file}")
    temp_audio_file = Path(tempfile.mkstemp(suffix=".wav")[1])
    try:
        subprocess.run(f"""ffmpeg -i "{video_file}" -vn -ar 16000 -v error -ac 1 -y "{temp_audio_file}" """, shell=True)
    except Exception as err:
        print(f"提取音频错误{err}，跳过")
        temp_audio_file.unlink()
        return
    audio, sample_rate = read_wave(str(temp_audio_file))
    temp_audio_file.unlink()
    vad = webrtcvad.Vad(0)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    parts = vad_collector(sample_rate, 30, 30 * connect_factor, vad, frames)
    print(f"{video_file}检测到{len(parts)}段语音, 时长是{[round(part[1] - part[0], 2) for part in parts]}")
    for part in parts:
        duration = part[1] - part[0]
        if min_duration and duration < min_duration:
            continue
        if max_duration and duration > max_duration:
            continue
        with tempfile.TemporaryDirectory() as temp_data_output_dir:
            temp_data_output_dir = Path(temp_data_output_dir)
            if rotate_k != 0:
                rotate_filter = f"""-vf "transpose={rotate_k}" """
            else:
                rotate_filter = ""
            try:
                subprocess.run(
                    f"""ffmpeg -ss {part[0]} -t {duration} -i "{video_file}" -r 25 {rotate_filter} -qscale:v 2 -v error {temp_data_output_dir}/temp%d.jpg""",
                    shell=True, check=True, timeout=30)
            except Exception as err:
                print(f"ffmpeg 错误{err}，跳过")
                break
            for image_file in temp_data_output_dir.glob("*.jpg"):
                image = cv2.imdecode(np.fromfile(str(image_file), dtype=np.uint8), -1)
                image = image[:image.shape[0] - image.shape[0] % 4, :image.shape[1] - image.shape[1] % 4]
                faces = arc_face_pro_3.detect_faces(image)
                if len(faces) != 1:
                    # print(f"异常图片，检测到{len(faces)}个人脸，删除数据")
                    break
                face = faces[0]
                # if face.faceInfo.faceOrient != 1:
                #     print(f"异常图片，人脸不正，删除数据")
                #     shutil.rmtree(str(temp_data_output_dir))
                #     break
                # if face.score < 0.4:
                #     print(f"人脸质量{face.score}过低，删除数据")
                #     shutil.rmtree(str(temp_data_output_dir))
                #     break
                ltx = max(0, face.bbox.ltx)
                lty = max(0, face.bbox.lty)
                rbx = min(image.shape[1], face.bbox.rbx)
                rby = min(image.shape[0], face.bbox.rby)
                face_image = image[lty:rby, ltx:rbx]
                if min(face_image.shape[:2]) <= 0:
                    # print(f"异常图片，人脸太小，{face} 删除数据")
                    break
                cv2.imencode('.jpg', face_image)[1].tofile(
                    str(temp_data_output_dir.joinpath(image_file.stem[4:] + ".jpg")))
                image_file.unlink()
            else:
                try:
                    subprocess.run(
                        f"""ffmpeg -ss {part[0]} -t {duration} -i "{video_file}" -vn -ac 1 -ar 16000 -v error {temp_data_output_dir}/audio.wav""",
                        shell=True, check=True)
                    # 相当于锁 防止多个进程生成的文件夹冲突
                    if queue:
                        status = queue.get()
                    output_video_count = 0
                    while True:
                        data_output_dir = output_dir.joinpath(f"{output_video_count:05d}")
                        if not data_output_dir.exists():
                            break
                        output_video_count += 1
                    shutil.copytree(temp_data_output_dir, data_output_dir)
                    if queue:
                        queue.put(status)
                    print(f"生成数据{data_output_dir}")
                except Exception as err:
                    print(f"生成音频错误{err}，删除数据")
                    shutil.rmtree(str(temp_data_output_dir))
    if queue:
        status = queue.get()
        status[0] += 1
        print(f"视频{video_file}处理完毕，处理进度{status[0]}/{status[1]}")
        queue.put(status)
    else:
        print(f"视频{video_file}处理完毕")


@app.command()
def batch_process_dir():
    ds = list(Path("/data/数字人/LRS2/mvlrs_v1/main").iterdir()) + list(
        Path("/data/数字人/LRS2/mvlrs_v1/pretrain").iterdir())
    for d in ds:
        process_dir(d=d,
                    output_dir=Path("/data/数字人/output/LRS2"),
                    connect_factor=8,
                    min_duration=0.8,
                    max_duration=15,
                    processes=10)


def get_output_dir_audio_duration(d: Path, start_index: int, end_index: int) -> Tuple[int, float]:
    """
    获取一个文件夹内所有音频文件的总时长
    :arg output_dir: 一个数据的文件夹
    """
    count, duration = 0, 0
    output_dirs = list(d.iterdir())
    for output_dir in output_dirs[start_index: end_index]:
        if not output_dir.is_dir():
            continue
        count += 1
        audio_file = output_dir.joinpath("audio.wav")
        if not audio_file.exists():
            continue
        duration += librosa.get_duration(filename=str(audio_file))
    return count, duration


@app.command()
def get_output_dir_total_audio_duration(output_dir: Path):
    """
    计算总时长
    :arg output_dir: 数据文件夹，里面包含多个数据文件夹
    """
    dirs = list(output_dir.iterdir())
    total_count, total_duration = 0, 0
    bar = tqdm(total=len(dirs))

    def apply_async_callback(data: Tuple[int, float]):
        nonlocal total_count, total_duration, bar
        total_duration += data[1]
        total_count += data[0]
        bar.update(data[0])
        bar.set_postfix_str(f"total_duration: {total_duration / 3600}")

    processes = 12
    pool = Pool(processes=processes)
    part_length = 100
    for index in range(len(dirs) // part_length + 1):
        pool.apply_async(func=get_output_dir_audio_duration,
                         args=(output_dir, index * part_length, (index + 1) * part_length),
                         callback=apply_async_callback,
                         error_callback=lambda x: print(x))
    pool.close()
    pool.join()
    print(f"total_count:{total_count},total_duration:{total_duration / 3600}小时")


if __name__ == '__main__':
    arc_face_pro_3 = ArcFacePro3()
    app()
