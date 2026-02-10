from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from vosk import KaldiRecognizer, Model


@dataclass(frozen=True)
class TranscriptSegment:
    """A transcript segment with approximate timing.

    :param start_s: Start time in seconds.
    :param end_s: End time in seconds.
    :param text: Recognized text.
    :return: TranscriptSegment instance.
    """
    start_s: float
    end_s: float
    text: str


class VoskSpeechToText:
    """Local speech-to-text wrapper using Vosk (offline)."""

    def __init__(
        self,
        model_dir: Path | str,
        *,
        sample_rate_hz: int = 16000,
        enable_word_times: bool = True,
    ) -> None:
        """Initialize the Vosk model.

        :param model_dir: Path to a Vosk model directory.
        :param sample_rate_hz: Expected WAV sample rate in Hz (commonly 16000).
        :param enable_word_times: If True, request word-level timing output when available.
        :return: None
        """
        self.model_dir = Path(model_dir)
        self.sample_rate_hz = int(sample_rate_hz)
        self.enable_word_times = bool(enable_word_times)
        self.model = Model(str(self.model_dir))

    def _segment_from_result_json(self, res: dict) -> Optional[TranscriptSegment]:
        """Create a segment from a Vosk result chunk when word timings are present.

        :param res: Parsed JSON from Vosk recognizer.
        :return: TranscriptSegment or None if no timing info.
        """
        words = res.get("result")
        text = (res.get("text") or "").strip()
        if not text:
            return None

        if isinstance(words, list) and words:
            starts = [w.get("start") for w in words if isinstance(w, dict) and w.get("start") is not None]
            ends = [w.get("end") for w in words if isinstance(w, dict) and w.get("end") is not None]
            if starts and ends:
                return TranscriptSegment(start_s=float(min(starts)), end_s=float(max(ends)), text=text)

        return None

    def transcribe_wav_pcm16_mono(
        self,
        wav_path: Path | str,
        *,
        chunk_frames: int = 4000,
    ) -> Tuple[str, List[TranscriptSegment]]:
        """Transcribe a PCM16 mono WAV file.

        :param wav_path: Path to WAV file (must be PCM16, mono).
        :param chunk_frames: Frames per chunk fed to recognizer.
        :return: (full_text, segments)
        """
        p = Path(wav_path)
        if p.suffix.lower() != ".wav":
            raise ValueError("transcribe_wav_pcm16_mono expects a .wav file.")

        with wave.open(str(p), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()

            if n_channels != 1:
                raise ValueError(f"WAV must be mono. Found {n_channels} channels.")
            if sampwidth != 2:
                raise ValueError(f"WAV must be 16-bit PCM (sampwidth=2). Found sampwidth={sampwidth}.")
            if sr != self.sample_rate_hz:
                raise ValueError(
                    f"WAV sample rate must match recognizer sample_rate_hz={self.sample_rate_hz}. Found sr={sr}."
                )

            rec = KaldiRecognizer(self.model, self.sample_rate_hz)
            try:
                rec.SetWords(True if self.enable_word_times else False)
            except Exception:
                pass

            texts: List[str] = []
            segments: List[TranscriptSegment] = []

            total_frames = wf.getnframes()
            duration_s = float(total_frames) / float(sr) if sr > 0 else 0.0

            while True:
                data = wf.readframes(chunk_frames)
                if not data:
                    break

                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    t = (res.get("text") or "").strip()
                    if t:
                        texts.append(t)
                        seg = self._segment_from_result_json(res)
                        if seg is not None:
                            segments.append(seg)

            final_res = json.loads(rec.FinalResult())
            t_final = (final_res.get("text") or "").strip()
            if t_final:
                texts.append(t_final)
                seg = self._segment_from_result_json(final_res)
                if seg is not None:
                    segments.append(seg)

            full_text = " ".join(texts).strip()
            if not segments and full_text:
                segments = [TranscriptSegment(start_s=0.0, end_s=duration_s, text=full_text)]

            return full_text, segments

    @classmethod
    def transcribe_folder_to_adjacent_txt(
        cls,
        folder_path: Path | str,
        *,
        model_dir: Path | str,
        sample_rate_hz: int = 16000,
        enable_word_times: bool = True,
        overwrite: bool = False,
        glob_pattern: str = "*.wav",
        recursive: bool = False,
        encoding: str = "utf-8",
    ) -> List[Path]:
        """Transcribe all matching WAV files in a folder and write .txt files adjacent to each audio file.

        Assumes the folder contains WAVs compatible with Vosk: mono, PCM16, sample_rate_hz.

        Output for each file is: <audio_path>.with_suffix(".txt")

        :param folder_path: Folder containing cleaned audio WAVs.
        :param model_dir: Path to Vosk model directory.
        :param sample_rate_hz: Expected WAV sample rate in Hz (commonly 16000).
        :param enable_word_times: If True, request word timing output where supported.
        :param overwrite: If False, skip files that already have a .txt.
        :param glob_pattern: Pattern for files to process (default "*.wav").
        :param recursive: If True, search recursively.
        :param encoding: Encoding to use when writing .txt.
        :return: List of .txt paths written.
        """
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Not a folder: {folder}")

        stt = cls(
            model_dir=model_dir,
            sample_rate_hz=sample_rate_hz,
            enable_word_times=enable_word_times,
        )

        wav_paths = sorted(folder.rglob(glob_pattern) if recursive else folder.glob(glob_pattern))
        written: List[Path] = []

        for wav_path in wav_paths:
            txt_path = wav_path.with_suffix(".txt")
            if txt_path.exists() and not overwrite:
                continue

            try:
                text, _segs = stt.transcribe_wav_pcm16_mono(wav_path)
            except Exception as e:
                # Write a small sidecar error file so batch runs are debuggable without crashing everything
                err_path = wav_path.with_suffix(".stt_error.txt")
                err_path.write_text(f"ERROR transcribing {wav_path.name}:\n{e}\n", encoding=encoding)
                continue

            txt_path.write_text(text + "\n", encoding=encoding)
            written.append(txt_path)

        return written


if __name__ == "__main__":
    # Example usage:
    # Ensure the folder contains 16kHz mono PCM16 WAVs (typical "cleaned audio" output).
    txt_files = VoskSpeechToText.transcribe_folder_to_adjacent_txt(
        folder_path="cleaned_audio",
        model_dir="vosk-model-small-en-us-0.15",
        recursive=False,
        overwrite=False,
    )
    print(f"Wrote {len(txt_files)} transcript files.")
