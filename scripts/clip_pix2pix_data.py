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
    output_dir = work_dir.joinpath("datasets", "liumin+qijian")
    output_dir.mkdir(exist_ok=True)
    train, val, test = 0, 0, 0
    for d in ["train", "val", "test"]:
        for ab in ["train_A", "train_B"]:
            output_dir.joinpath(d, ab).mkdir(exist_ok=True, parents=True)
    video_files = list(work_dir.joinpath("datasets", "original_video", "刘敏第二次录制视频").iterdir()) + list(work_dir.joinpath("datasets", "original_video", "祁健视频").iterdir())
    for video_file in video_files:
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
        print(f"parts {parts}")
        frame_index = -1
        while True:
            print(video_file,train, val, test)
            ret, frame = video.read()
            if not ret:
                break
            frame_index += 1
            # 读取前后各9帧对应音频，一共19帧长度
            # 一帧是25ms，对应音频0.025*16000=400个采样点
            # 前后各9帧，对应音频18*400=7200个采样点

            # 前后各个
            # 当前帧frame_index
            audio_index = int(80. * (frame_index / float(25)))
            audio_time_second = int(frame_index / 25)
            # 去除静音帧
            if not any([part[0] <= audio_time_second <= part[1] for part in parts]):
                continue
            start, end = audio_index - 40, audio_index + 40
            if start < 0 or end > len(orig_mel):
                continue
            mel = orig_mel[start: end].T
            # mel = librosa.feature.melspectrogram(y=sample_audio, sr=sample_rate, S=None, n_mels=16)  # mel=512*64
            # mel = process(sample_audio)  # 16*32*32 = 512*32
            # print(mel.shape)

            # mel = mel.reshape(1, 32, 32)
            mel = mel.reshape(1, 80, 80)
            # exit()
            # mel = mel.reshape(16, 32, 32)
            # mfcc = librosa.feature.mfcc(y=sample_audio, sr=sample_rate, n_mels=n_mels)
            # mfcc归一化
            # mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min())
            # mfcc = (mfcc * 255).astype(np.uint8)
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
            # result = np.hstack((face_frame_label, face_frame_train))
            v = random.random()
            if v < 0.8:
                train += 1
                # if train > 20000:
                #     break
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
            print(output_a_file)
