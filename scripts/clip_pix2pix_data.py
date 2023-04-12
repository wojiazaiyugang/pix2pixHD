# 处理数据集
import tempfile
import random
import subprocess
from copy import deepcopy
from pathlib import Path

import cv2
import librosa
import numpy as np
import webrtcvad

from audio import melspectrogram
from clip_wave2lip_data import vad_collector, frame_generator, read_wave

if __name__ == '__main__':
    arc_face_pro_3 = None
    work_dir = Path(__file__).parent.parent.resolve()
    output_dir = work_dir.joinpath("datasets", "liumin_wav2lip")
    output_dir.mkdir(exist_ok=True)
    train, val, test = 45745, 5604, 5744
    for d in ["train", "val", "test"]:
        for ab in ["train_A", "train_B"]:
            output_dir.joinpath(d, ab).mkdir(exist_ok=True, parents=True)
    videos = []
    for suffix in [".mp4", ".mov", ".MP4", ".MOV"]:
        videos.extend(list(Path("/workspace/pix2pixHD/KrLongData").rglob(f"*{suffix}")))
    for video_file in videos:
        try:
            video = cv2.VideoCapture(str(video_file))
            video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            subprocess.run(f"ffmpeg -i {video_file} -vn -v error -y /workspace/HiInfer/audio.wav", shell=True, check=True)
            audio_file = Path("/workspace/HiInfer/audio.wav")
            audio, sample_rate = librosa.core.load(audio_file, sr=16000)
            orig_mel = melspectrogram(audio).T

            temp_audio_file = Path(tempfile.mkstemp(suffix=".wav")[1])
            subprocess.run(f"""ffmpeg -i "{video_file}" -vn -ar 16000 -v error -ac 1 -y "{temp_audio_file}" """, shell=True)
            temp_audio, sample_rate = read_wave(str(temp_audio_file))
            temp_audio_file.unlink()
            vad = webrtcvad.Vad(0)
            frames = list(frame_generator(30, temp_audio, sample_rate))
            parts = vad_collector(sample_rate, 30, 30 * 8, vad, frames)
            frame_index = -1
            while True:
                if train % 500 == 0:
                    print(video_file,train, val, test)
                ret, frame = video.read()
                if not ret:
                    break
                frame_index += 1
                audio_index = int(80. * (frame_index / float(25)))
                audio_time_second = int(frame_index / 25)
                # 去除静音帧
                if not any([part[0] <= audio_time_second <= part[1] for part in parts]):
                    continue
                start, end = audio_index - 40, audio_index + 40
                if start < 0 or end > len(orig_mel):
                    continue
                mel = orig_mel[start: end].T
                mel = mel.reshape(1, 80, 80)
                if not arc_face_pro_3:
                    from infer import ArcFacePro3

                    arc_face_pro_3 = ArcFacePro3()
                faces = arc_face_pro_3.detect_faces(frame)
                if len(faces) != 1:
                    continue
                face = faces[0]
                face.bbox.ltx = face.bbox.ltx - 10
                face.bbox.rbx = face.bbox.rbx + 10
                face.bbox.rby = face.bbox.rby + 10
                face_frame_label = frame[face.bbox.lty:face.bbox.rby, face.bbox.ltx: face.bbox.rbx, :]
                face_frame_train = deepcopy(face_frame_label)
                face_frame_train[face_frame_train.shape[0] // 2:, :, :] = 0
                v = random.random()
                if v < 0.8:
                    train += 1
                    output_a_file = output_dir.joinpath("train", "train_A").joinpath(f"{train}.jpg")
                    output_b_file = output_dir.joinpath("train", "train_B").joinpath(f"{train}.jpg")
                elif 0.8 <= v < 0.9:
                    val += 1
                    output_a_file = output_dir.joinpath("val", "train_A").joinpath(f"{val}.jpg")
                    output_b_file = output_dir.joinpath("val", "train_B").joinpath(f"{val}.jpg")
                else:
                    test += 1
                    output_a_file = output_dir.joinpath("test", "train_A").joinpath(f"{test}.jpg")
                    output_b_file = output_dir.joinpath("test", "train_B").joinpath(f"{test}.jpg")
                cv2.imwrite(str(output_a_file), face_frame_train)
                cv2.imwrite(str(output_b_file), face_frame_label)
                np.save(str(output_a_file.with_suffix(".npy")), mel)
                # print(output_a_file)
            # video_file.unlink()
        except Exception as err:
            print(f"{video_file}处理异常{err}")
