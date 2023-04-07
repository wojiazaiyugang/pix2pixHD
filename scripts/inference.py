"""
推理脚本
"""
"""
python test.py --dataroot ./datasets/weijiaxing_half/ --name 2023033 --model pix2pix --direction BtoA
"""
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import librosa
import numpy as np

arc_face_pro_3 = None


def infer(video_file: Path, audio_file: Path, name: str):
    global arc_face_pro_3
    project_dir = Path(__file__).parent.parent.resolve()
    work_dir = project_dir.joinpath("inference")
    shutil.rmtree(work_dir, ignore_errors=True)
    face_dir = work_dir.joinpath("test_A")
    image_dir = work_dir.joinpath("image")
    result_dir = work_dir.joinpath("result")
    for d in [image_dir, face_dir, result_dir]:
        d.mkdir(parents=True, exist_ok=True)


    audio, sample_rate = librosa.load(str(audio_file), sr=16000)
    print(f"audio shape {audio.shape}")
    video = cv2.VideoCapture(str(video_file))
    video_height, video_width = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_index = -1
    output_dir = project_dir.joinpath("results", name, "test_latest", "images")
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_face_bbox = {}  # {image_index: bbox}
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_index += 1
        print(frame_index)
        if frame_index > 30 * 25:
            break
        audio_index = frame_index * 400
        start, end = audio_index - 15872 - 256, audio_index + 15872 + 256
        if start < 0:
            continue
        if end > len(audio):
            break
        sample_audio = audio[start: end]
        mel = librosa.feature.melspectrogram(y=sample_audio, sr=sample_rate, S=None, n_mels=16*32)  # mel=512*32
        mel = mel.reshape(32, 32, 32)
        if not arc_face_pro_3:
            from infer import ArcFacePro3
            arc_face_pro_3 = ArcFacePro3()
        faces = arc_face_pro_3.detect_faces(frame)
        if len(faces) == 0:
            raise Exception(f"还没处理")
        face = faces[0]
        face_frame = frame[face.bbox.lty:face.bbox.rby, face.bbox.ltx: face.bbox.rbx, :]
        face_frame[face_frame.shape[0] // 2:, :, :] = 0
        output_face = face_dir.joinpath(f"{frame_index:0>5}.jpg")
        cv2.imwrite(str(output_face), face_frame)
        np.save(str(output_face.with_suffix(".npy")), mel)
        cv2.imwrite(str(image_dir.joinpath(f"{frame_index:0>5}.jpg")), frame)
        image_face_bbox[frame_index] = face.bbox
    s = time.time()
    command = (f"""python test.py """
               f"""--name {name} """
               f"""--netG global """
               f"""--dataroot {face_dir.parent} """
               f"""--label_nc 0 """
               f"""--no_instance """
               f"""--loadSize 512 """
               f"""--resize_or_crop resize_and_crop """
               f"""--how_many {frame_index} """)
    subprocess.run(
        command,
        cwd=project_dir, shell=True, check=True)
    print(f"{frame_index}， 耗时 {time.time() - s:.2f} 秒")
    for file_index in range(frame_index):
        if file_index in image_face_bbox:
            bbox = image_face_bbox[file_index]
            image = cv2.imread(str(image_dir.joinpath(f"{file_index:0>5}.jpg")))
            output_face = cv2.imread(str(output_dir.joinpath(f"{file_index:0>5}_synthesized_image.jpg")))
            output_face = cv2.resize(output_face, (bbox.rbx - bbox.ltx, bbox.rby - bbox.lty))
            # print(bbox, (bbox.rbx - bbox.ltx, bbox.rby - bbox.lty), output_face.shape, image.shape)
            image[bbox.lty:bbox.rby, bbox.ltx: bbox.rbx, :] = output_face
            cv2.imwrite(str(result_dir.joinpath(f"{file_index:0>5}.jpg")), image)
        else:
            black = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            cv2.imwrite(str(result_dir.joinpath(f"{file_index:0>5}.jpg")), black)
    subprocess.run(
        f"""ffmpeg -r 25 -f image2 -i {result_dir}/%05d.jpg -i {audio_file} -shortest -y result.mp4""",
        cwd=project_dir, shell=True, check=True)


if __name__ == '__main__':
    infer(Path("/workspace/pix2pixHD/liumin.mp4"),
          Path("/workspace/pix2pixHD/liumin.wav"),
          "liumin_onevideo")
    # 2023033
