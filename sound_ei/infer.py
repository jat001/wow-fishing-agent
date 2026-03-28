import glob
import os
import shutil
import time
from typing import Optional

import fire
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from loguru import logger
from transformers import AutoModelForAudioClassification

from sound_ei import loopback
from sound_ei.dataset_bite import feature_extractor
from sound_ei.loopback import loopback_stream, AudioDevice

sound_checkpoint = None


def get_best_checkpoint(path: str = "models/bite_model"):
    global sound_checkpoint
    if sound_checkpoint:
        return sound_checkpoint
    n = 1000000
    model = None
    for ckpt in glob.glob(f"{path}/checkpoint-*"):
        no = int(ckpt.split("-")[-1])
        if no < n:
            n = no
            model = ckpt
    if not model:
        raise Exception("model checkpoint not found")
    logger.info("get best checkpoint: {}", model)
    sound_checkpoint = model
    return model


def infer(
        model,
        wave: np.ndarray | torch.Tensor,
        sample_rate: int = feature_extractor.sampling_rate,
) -> torch.Tensor:
    """数据需要是channel first"""
    if type(wave) == np.ndarray:
        wave = torch.from_numpy(wave)
    if len(wave.shape) == 2:
        wave = wave[0, :]
    if sample_rate != feature_extractor.sampling_rate:
        wave = F.resample(wave, sample_rate, feature_extractor.sampling_rate)
    inputs = feature_extractor(
        wave, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
    )

    logits = model(input_values=inputs['input_values'].to(model.device)).logits
    probs = torch.softmax(logits, dim=1)
    return probs


def load_model(ckpt: str, device: str = 'cuda:0'):
    start = time.time()
    model = AutoModelForAudioClassification.from_pretrained(ckpt).to(device)
    logger.info("load model in {:.2}s", time.time() - start)
    return model


def stream(
        model: torch.nn.Module,
        audio_device: AudioDevice = loopback.default_device,
        window: int = 3,
        step: int = 1,
        maxlen: int = 16,
        channels: int = 2,
        device: str = 'cuda:0'
):
    with loopback_stream(device=audio_device, chunk_seconds=1) as stream:

        chunk = int(step * audio_device.sample_rate)

        buffer = torch.zeros([channels, maxlen * chunk], dtype=torch.float32, device=device)

        i = 0
        with torch.no_grad():
            for i in range(maxlen):
                wave1 = np.frombuffer(stream.read(chunk), dtype=np.float32).reshape(-1, audio_device.channels).T
                if audio_device.channels != channels:
                    if audio_device.channels == 1 and channels == 2:
                        wave1 = np.repeat(wave1, 2, axis=0)
                    elif audio_device.channels > 2 and channels == 2:
                        left = wave1[0].copy()
                        right = wave1[1].copy()

                        if audio_device.channels >= 3:
                            left += 0.707 * wave1[2]
                            right += 0.707 * wave1[2]

                        if audio_device.channels >= 5:
                            left += 0.707 * wave1[4]
                            right += 0.707 * wave1[5]

                        if audio_device.channels >= 7:
                            left += 0.5 * wave1[6]
                            right += 0.5 * wave1[7]

                        wave1 = np.array([left, right])
                    else:
                        mono = wave1.mean(axis=0)
                        wave1 = np.tile(mono, (channels, 1))
                buffer[:, i * chunk: (i + 1) * chunk] = torch.from_numpy(wave1.copy())
                if i < window - 1:
                    continue

                start = time.time()
                probs = infer(model, buffer[:, (i - window + 1) * chunk: (i + 1) * chunk],
                              audio_device.sample_rate)
                prob = probs.cpu().numpy()[0, 1]
                pred = "bite" if prob > 0.5 else "other"
                logger.info("audio chunk infer [{:02d}{:4d}ms] => {} ({:.3f})", i, int(1000 * (time.time() - start)),
                            pred, prob)
                if pred == "bite":
                    return "bite", buffer[:, :(i + 1) * chunk], i + 1
        return "other", buffer, i + 1


def bite_as_negative(
        save_bite_limit: int = 5,
        save_path: str = f"datasets/record/negative_{int(time.time())}",
):
    logger.info("save bite as negative examples")
    device: AudioDevice = loopback.default_device
    i = 0
    for t, label, data in stream(device):
        if label == "bite":
            os.makedirs(save_path, exist_ok=True)
            torchaudio.save(f"{save_path}/{i}_0.wav", data, device.sample_rate)
            i += 1
            if i >= save_bite_limit:
                break


def miss_bite_predict(model_path: str = get_best_checkpoint(),
                      glob_pattern: str = "datasets/miss-bite/*.ogg",
                      window: int = 3):
    """
    对于采集到的miss音频进行重新预测

    Args:
        model_path (str, optional): Path to the model checkpoint. Defaults to the best checkpoint.
        glob_pattern (str, optional): Glob pattern to match audio files. Defaults to "datasets/miss-bite/*.ogg".
        window (int, optional): Window size for audio inference. Defaults to 3.
    """
    # Load the model from the specified checkpoint
    model = load_model(model_path)
    files = glob.glob(glob_pattern)
    for file in files:
        file = file.replace("\\", "/")
        audio, sr = torchaudio.load(file)

        t = 0
        for i in range(16 - window):
            prob = infer(model, audio[:, (i * sr): (i + window) * sr], sample_rate=sr)
            positive = float(prob.cpu().detach().numpy()[0, 1])
            if positive > 0.5:
                t = i + 2
                break

        if t > 0:
            target = file.replace("_16_miss", f"_{t + 1}")
            shutil.move(file, target)


class BiteListener:
    def __init__(self, device: str = 'cuda:0', audio_device: AudioDevice = loopback.default_device,
                 ckpt: str = get_best_checkpoint()):
        self.model = load_model(ckpt, device=device)
        self.audio_device = loopback.default_device

    def listen(self, max_len: int, window: int = 3) -> tuple[str, Optional[torch.Tensor]]:
        return stream(maxlen=max_len, window=window)


if __name__ == "__main__":
    fire.Fire()
