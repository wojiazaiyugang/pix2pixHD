# fairseq 使用
from pathlib import Path
import torch
import torch.nn.functional as F
import soundfile as sf
from fairseq import checkpoint_utils
import librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

work_dir = Path(__file__).parent.parent.resolve()

model_path = str(work_dir.joinpath("chinese-wav2vec2-base-fairseq-ckpt.pt"))
wav_path = "/root/HiInfer/audio_5s.wav"


def postprocess(feats, normalize=False):
    if feats.dim() == 2:
        feats = feats.mean(-1)

    assert feats.dim() == 1, feats.dim()

    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats


print("loading model(s) from {}".format(model_path))
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
print("loaded model(s) from {}".format(model_path))
print(f"normalize: {saved_cfg.task.normalize}")

model = models[0]
model = model.to(device)
model = model.half()
model.eval()

# wav, sr = sf.read(wav_path)
# wav, sr = librosa.load(wav_path, sr=16000)
# wav = wav[:len(wav) // 5]


def process(wav):
    feat = torch.from_numpy(wav).float()
    feat = postprocess(feat, normalize=saved_cfg.task.normalize)
    feats = feat.view(1, -1)
    padding_mask = (
        torch.BoolTensor(feats.shape).fill_(False)
    )
    inputs = {
        "source": feats.half().to(device),
        "padding_mask": padding_mask.to(device),
    }

    with torch.no_grad():
        logits = model.extract_features(**inputs)
    return logits["features"].cpu().numpy()

if __name__ == '__main__':
    process(wav)
