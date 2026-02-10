from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Literal

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioData:
    """Container for audio samples.

    :param samples: Float32 samples in range [-1, 1], shape (n,) or (n, ch).
    :param sample_rate: Sample rate in Hz.
    :return: AudioData instance.
    """
    samples: np.ndarray
    sample_rate: int


class AudioFileIO:
    """Handles loading and saving audio files (.wav or .mp3)."""

    def load(self, audio_path: Path | str, *, mono: bool = False) -> AudioData:
        """Load an audio file into float32 samples.

        :param audio_path: Path to input .wav or .mp3 file.
        :param mono: If True, downmix to mono.
        :return: AudioData object.
        """
        p = Path(audio_path)

        if p.suffix.lower() not in (".wav", ".mp3"):
            raise ValueError("Only .wav and .mp3 files are supported.")

        samples, sr = sf.read(str(p), always_2d=True, dtype="float32")

        if mono:
            samples = samples.mean(axis=1, keepdims=True)

        out = samples if samples.shape[1] > 1 else samples[:, 0]
        return AudioData(samples=out, sample_rate=int(sr))

    def save(self, audio: AudioData, out_path: Path | str) -> None:
        """Save audio to disk as WAV.

        :param audio: AudioData to save.
        :param out_path: Output path (.wav recommended).
        :return: None
        """
        p = Path(out_path)

        samples = np.asarray(audio.samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, None]

        sf.write(str(p), samples, audio.sample_rate)


class AudioPreprocessor:
    """Audio preprocessing: estimate noise floor and remove long silent gaps."""

    def __init__(self, io: Optional[AudioFileIO] = None) -> None:
        """Initialize preprocessor.

        :param io: Optional AudioFileIO instance.
        :return: None
        """
        self.io = io if io is not None else AudioFileIO()

    @staticmethod
    def _to_mono(samples: np.ndarray) -> np.ndarray:
        """Convert samples to mono.

        :param samples: Audio samples shape (n,) or (n, ch).
        :return: Mono samples shape (n,).
        """
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim == 1:
            return x
        return x.mean(axis=1)

    @staticmethod
    def estimate_noise_rms_global(
        audio: AudioData,
        *,
        window_seconds: float = 1.0,
        hop_seconds: float = 0.1,
    ) -> float:
        """Estimate noise RMS using the quietest window in the entire file.

        Slides a fixed-duration window across the whole recording, computes RMS for each window,
        and returns the minimum RMS observed (the quietest window).

        :param audio: Input audio.
        :param window_seconds: Duration (seconds) of the RMS window. Default 1.0.
        :param hop_seconds: Step size (seconds) between windows. Default 0.1.
        :return: Estimated global noise RMS.
        """
        x = AudioPreprocessor._to_mono(audio.samples)
        sr = audio.sample_rate

        win = int(max(1.0, window_seconds) * sr)
        hop = int(max(1, hop_seconds * sr))

        if x.shape[0] <= win:
            return float(np.sqrt(np.mean(x * x) + 1e-12))

        # Efficient sliding RMS via cumulative sum of squares
        x2 = x.astype(np.float64) ** 2
        csum = np.concatenate(([0.0], np.cumsum(x2)))

        min_rms = float("inf")
        # Evaluate RMS at window starts every 'hop' samples
        for start in range(0, x.shape[0] - win + 1, hop):
            end = start + win
            mean_sq = (csum[end] - csum[start]) / win
            rms = float(np.sqrt(mean_sq + 1e-12))
            if rms < min_rms:
                min_rms = rms

        return float(min_rms)

    @staticmethod
    def estimate_noise_rms_inital(
        audio: AudioData,
        *,
        first_seconds: float = 1.0,
        window_ms: float = 50.0,
        percentile: float = 50.0,
    ) -> float:
        """Estimate baseline noise RMS from the first second of audio.

        :param audio: Input audio.
        :param first_seconds: Seconds to analyze at start.
        :param window_ms: RMS window size in ms.
        :param percentile: Percentile of RMS values to use.
        :return: Noise RMS estimate.
        """
        x = AudioPreprocessor._to_mono(audio.samples)
        sr = audio.sample_rate

        n = min(int(first_seconds * sr), x.shape[0])
        if n <= 0:
            return 1e-12

        seg = x[:n]

        win = max(1, int(window_ms * 0.001 * sr))
        n_wins = max(1, seg.shape[0] // win)

        seg = seg[: n_wins * win]
        frames = seg.reshape(n_wins, win)

        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        return float(np.percentile(rms, percentile))

    @staticmethod
    def remove_silent_gaps(
        audio: AudioData,
        *,
        noise_rms: float,
        threshold_multiplier: float = 1.25,
        window_ms: float = 50.0,
        gap_length_sec: float = 5.0,
    ) -> Tuple[AudioData, dict]:
        """Remove silent regions longer than gap_length_sec.

        :param audio: Input audio.
        :param noise_rms: Baseline noise RMS.
        :param threshold_multiplier: Silence threshold factor.
        :param window_ms: RMS window size in ms.
        :param gap_length_sec: Minimum silent duration to remove.
        :return: (Cleaned AudioData, stats dict).
        """
        sr = audio.sample_rate
        x_mono = AudioPreprocessor._to_mono(audio.samples)
        x = np.asarray(audio.samples, dtype=np.float32)

        win = max(1, int(window_ms * 0.001 * sr))
        n_wins = int(np.ceil(x_mono.shape[0] / win))

        pad = n_wins * win - x_mono.shape[0]
        if pad > 0:
            x_mono = np.pad(x_mono, (0, pad))
            if x.ndim == 1:
                x = np.pad(x, (0, pad))
            else:
                x = np.pad(x, ((0, pad), (0, 0)))

        frames = x_mono.reshape(n_wins, win)
        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)

        thr = noise_rms * threshold_multiplier
        is_silent = rms <= thr

        min_silent_wins = int(np.ceil(gap_length_sec * sr / win))

        keep = np.ones(n_wins, dtype=bool)
        removed_regions = 0
        removed_wins = 0

        i = 0
        while i < n_wins:
            if not is_silent[i]:
                i += 1
                continue

            j = i
            while j < n_wins and is_silent[j]:
                j += 1

            run_len = j - i
            if run_len >= min_silent_wins:
                keep[i:j] = False
                removed_regions += 1
                removed_wins += run_len

            i = j

        keep_samples = np.repeat(keep, win)

        if x.ndim == 1:
            x_clean = x[keep_samples]
        else:
            x_clean = x[keep_samples, :]

        cleaned = AudioData(samples=x_clean, sample_rate=sr)

        stats = {
            "removed_regions": removed_regions,
            "removed_seconds": float(removed_wins * win / sr),
            "noise_rms": float(noise_rms),
            "threshold_rms": float(thr),
        }

        return cleaned, stats

    def clean_file(
        self,
        audio_path: Path | str,
        noise_mode: Literal['intial', 'global'] = 'global',
        gap_length_sec: float = 5.0,
        suffix: str = "_cleaned",
    ) -> Tuple[Path, dict]:
        """Load an audio file, remove long silent gaps, and save cleaned WAV next to it.

        :param audio_path: Input file path (.wav or .mp3).
        :param noise_mode: Either "global" or "initial" for how to estimate noise floor
        :param gap_length_sec: Minimum silent duration to remove.
        :param suffix: Output filename suffix.
        :return: (Output path, stats).
        """
        in_path = Path(audio_path)

        audio = self.io.load(in_path)
        if noise_mode.lower() == 'initial':
            noise_rms = self.estimate_noise_rms_inital(audio)
        else:
            noise_rms = self.estimate_noise_rms_global(audio)

        cleaned, stats = self.remove_silent_gaps(
            audio,
            noise_rms=noise_rms,
            gap_length_sec=gap_length_sec,
        )

        out_path = in_path.with_name(f"{in_path.stem}{suffix}.wav")
        self.io.save(cleaned, out_path)

        return out_path, stats

    @classmethod
    def clean_folder(
        cls,
        folder: Path | str,
        out_subfolder: str = "cleaned",
        gap_length_sec: float = 5.0,
        noise_mode: Literal["initial", "global"] = "initial",
        suffix: str = "_cleaned",
    ) -> List[Tuple[Path, Path, Dict]]:
        """Clean all audio files in a folder and save results to a subfolder.

        Processes all .wav and .mp3 files in the given folder, removes long silent gaps,
        and writes cleaned .wav outputs into:

            <folder>/<out_subfolder>/

        :param folder: Folder containing audio files.
        :param out_subfolder: Name of subfolder to store cleaned outputs.
        :param gap_length_sec: Minimum silent duration to remove (seconds).
        :param noise_mode: Either "global" or "initial" for estimating noise floor.
        :param suffix: Suffix appended to cleaned output filenames.
        :return: List of tuples (input_path, output_path, stats).
        """
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        out_dir = folder_path / out_subfolder
        out_dir.mkdir(parents=True, exist_ok=True)

        pre = cls()

        results: List[Tuple[Path, Path, Dict]] = []

        audio_files = sorted(
            list(folder_path.glob("*.wav")) +
            list(folder_path.glob("*.mp3"))
        )

        for in_path in audio_files:
            audio = pre.io.load(in_path)

            # Noise floor estimation mode
            if noise_mode == "global":
                noise_rms = pre.estimate_noise_rms_global(audio)
            elif noise_mode == "initial":
                noise_rms = pre.estimate_noise_rms(audio)
            else:
                raise ValueError("noise_mode must be 'initial' or 'global'")

            cleaned, stats = pre.remove_silent_gaps(
                audio,
                noise_rms=noise_rms,
                gap_length_sec=gap_length_sec,
            )

            out_path = out_dir / f"{in_path.stem}{suffix}.wav"
            pre.io.save(cleaned, out_path)

            stats["noise_mode"] = noise_mode
            stats["input_file"] = str(in_path.name)
            stats["output_file"] = str(out_path.name)

            results.append((in_path, out_path, stats))

        return results


if __name__ == "__main__":
    results = AudioPreprocessor.clean_folder(
        "recordings/",
        out_subfolder="cleaned_audio",
        gap_length_sec=5.0,
        noise_mode="global",
    )

    for inp, outp, stats in results:
        print(f"{inp.name} -> {outp.name}")
        print(f"  Removed {stats['removed_regions']} silent gaps")
        print(f"  Removed {stats['removed_seconds']:.2f} seconds total\n")
