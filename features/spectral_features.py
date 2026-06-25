import os
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.stats import entropy, skew, kurtosis

_current_dir = Path(__file__).resolve().parent
LOCAL_MODEL_PATH = str(_current_dir.parent / "local_models" / "gpt2")

class SpectralFeatureExtractor:
    def __init__(self, model_name="gpt2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_token_probabilities(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)
        input_ids = inputs["input_ids"][0]

        # 向量化提取：probs[:, :-1, :] 对应 input_ids[:, 1:]
        probs_seq = probs[0, :-1, :]
        target_ids = input_ids[1:].unsqueeze(1)
        token_probs = probs_seq.gather(dim=1, index=target_ids).squeeze()

        return token_probs.cpu().numpy()

    def extract_features(self, text):
        probs = self.get_token_probabilities(text)

        if len(probs) < 5:
            return [0.0] * 14

        # FFT
        fft_vals = np.fft.fft(probs)
        fft_amps = np.abs(fft_vals) / len(probs)

        n_half = max(1, len(probs) // 2)
        half_amps = fft_amps[:n_half]

        # 频域特征
        avg_amp = np.mean(half_amps)

        # 按 30%/70% 划分低频/高频
        split_point = max(1, int(n_half * 0.3))
        low_freq_energy = np.mean(half_amps[:split_point]**2) if split_point > 0 else 0.0
        high_freq_energy = np.mean(half_amps[split_point:]**2) if split_point < n_half else 0.0

        # 频谱熵
        prob_dist = half_amps / (np.sum(half_amps) + 1e-10)
        spec_entropy = entropy(prob_dist + 1e-10)

        # 峰值频率
        peak_freq = float(np.argmax(half_amps) if len(half_amps) > 0 else 0)

        # 能量比
        energy_ratio = low_freq_energy / (low_freq_energy + high_freq_energy + 1e-10)

        # 时域统计特征
        p_mean = np.mean(probs)
        p_var = np.var(probs)
        p_skew = float(skew(probs)) if len(probs) >= 3 else 0.0
        p_kurt = float(kurtosis(probs, fisher=True)) if len(probs) >= 4 else 0.0

        # 谱平坦度
        valid_amps = half_amps[half_amps > 1e-8]
        if len(valid_amps) > 0:
            geo_mean = np.exp(np.mean(np.log(valid_amps + 1e-10)))
            arith_mean = np.mean(half_amps) + 1e-10
            spec_flatness = geo_mean / arith_mean
        else:
            spec_flatness = 0.0

        # 一阶自相关
        if len(probs) > 1:
            try:
                autocorr_1 = float(np.corrcoef(probs[:-1], probs[1:])[0, 1])
                if np.isnan(autocorr_1):
                    autocorr_1 = 0.0
            except:
                autocorr_1 = 0.0
        else:
            autocorr_1 = 0.0

        # 功率谱密度
        psd = half_amps ** 2
        psd_max = float(np.max(psd)) if len(psd) > 0 else 0.0
        psd_mean = float(np.mean(psd)) if len(psd) > 0 else 0.0

        return [
            low_freq_energy, high_freq_energy, spec_entropy, avg_amp, peak_freq,
            energy_ratio, p_mean, p_var, p_skew, p_kurt,
            spec_flatness, autocorr_1, psd_max, psd_mean
        ]


if __name__ == "__main__":
    extractor = SpectralFeatureExtractor(model_name=LOCAL_MODEL_PATH)
    test_text = "This is a simple test sentence to verify the spectral feature extraction."
    feats = extractor.extract_features(test_text)
    print("Spectral Features:", feats)