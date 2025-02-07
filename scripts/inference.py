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
from audio import melspectrogram
from typer import Typer
from tqdm import tqdm

from face_parsing import init_parser, swap_regions

app = Typer()
arc_face_pro_3 = None


@app.command()
def infer(video_file: Path, audio_file: Path, name: str, epoch: str = "latest", start_frame_index: int = 0):
    """
    :arg: start_frame_index: 推理使用的d
    """
    global arc_face_pro_3
    project_dir = Path(__file__).parent.parent.resolve()
    work_dir = project_dir.joinpath("inference")
    shutil.rmtree(work_dir, ignore_errors=True)
    face_dir = work_dir.joinpath("test_A")
    image_dir = work_dir.joinpath("image")
    result_dir = work_dir.joinpath("result")
    for d in [image_dir, face_dir, result_dir]:
        d.mkdir(parents=True, exist_ok=True)

    audio, sample_rate = librosa.core.load(audio_file, sr=16000)
    orig_mel = melspectrogram(audio).T
    print(f"audio shape {audio.shape}")
    video = cv2.VideoCapture(str(video_file))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    video_height, video_width = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_index = -1
    write_index = 0  # 写图片的index ffmpeg的索引要从0开始
    output_dir = project_dir.joinpath("results", name, f"test_{epoch}", "images")
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_face_bbox = {}  # {image_index: bbox}
    progress_bar = tqdm()
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    while True:
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame_index += 1
        progress_bar.update(1)
        # if frame_index > 30 * 25:
        #     break
        audio_index = int(80. * (frame_index / float(25)))
        start, end = audio_index - 40, audio_index + 40
        if start < 0:
            continue
        if end > len(orig_mel):
            break
        mel = orig_mel[start: end].T
        mel = mel.reshape(1, 80, 80)
        if not arc_face_pro_3:
            from infer import ArcFacePro3
            arc_face_pro_3 = ArcFacePro3()
        faces = arc_face_pro_3.detect_faces(frame)
        if len(faces) == 0:
            raise Exception(f"还没处理")
        face = faces[0]
        face_frame = frame[face.bbox.lty:face.bbox.rby, face.bbox.ltx: face.bbox.rbx, :].copy()
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
               f"""--which_epoch {epoch} """
               f"""--loadSize 512 """
               f"""--resize_or_crop resize_and_crop """
               f"""--how_many {frame_index} """)
    subprocess.run(
        command,
        cwd=project_dir, shell=True, check=True)
    print(f"{frame_index}， 耗时 {time.time() - s:.2f} 秒")
    seg_net = init_parser(str(project_dir.joinpath("checkpoints", "face_segmentation.pth")))
    for file_index in range(frame_index):
        if file_index in image_face_bbox:
            bbox = image_face_bbox[file_index]
            image = cv2.imread(str(image_dir.joinpath(f"{file_index:0>5}.jpg")))
            output_face = cv2.imread(str(output_dir.joinpath(f"{file_index:0>5}_synthesized_image.jpg")))
            output_face = cv2.resize(output_face, (bbox.rbx - bbox.ltx, bbox.rby - bbox.lty))
            # print(bbox, (bbox.rbx - bbox.ltx, bbox.rby - bbox.lty), output_face.shape, image.shape)
            output_face = swap_regions(image[bbox.lty:bbox.rby, bbox.ltx: bbox.rbx], output_face, seg_net)
            image[bbox.lty:bbox.rby, bbox.ltx: bbox.rbx] = output_face
            cv2.imwrite(str(result_dir.joinpath(f"{write_index:0>5}.jpg")), image)
            write_index += 1
        else:
            pass
            # black = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            # generate a green image
            # black[:, :, 0] = 112
            # black[:, :, 1] = 222
            # black[:, :, 2] = 119
            # cv2.imwrite(str(result_dir.joinpath(f"{file_index:0>5}.jpg")), black)
    # frame_index - write_index就是前面没有画面对应的音频长度，截掉音频从而对对齐
    subprocess.run(f"""ffmpeg -r 25 -f image2 -i {result_dir}/%05d.jpg -ss {(frame_index - write_index) / fps} -i {audio_file} -shortest -y result.mp4""",
        cwd=project_dir,
        shell=True,
        check=True)


if __name__ == '__main__':
    app()
    # infer(Path("/workspace/pix2pixHD/liumin.mp4"),
    #       Path("/workspace/pix2pixHD/2.MP3"),
    #       "pretrain_liumin",
    #       epoch="latest")
    # 2023033
