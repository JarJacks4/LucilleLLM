"""
LucilleLLM - Soundscape Audio Setup Script

Generates 17 royalty-free placeholder audio files programmatically
using numpy and uploads them to Google Cloud Storage.

Audio is generated as WAV and converted to MP3 via pydub.

Usage:
    python scripts/setup_audio.py --bucket lucille-soundscapes
    python scripts/setup_audio.py --generate-only
    python scripts/setup_audio.py --bucket lucille-soundscapes --upload-only
    python scripts/setup_audio.py --bucket lucille-soundscapes --verify

Dependencies:
    pip install numpy soundfile pydub
    ffmpeg must be installed for MP3 conversion
"""

import argparse
import logging
import os
import sys

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Audio parameters
SAMPLE_RATE = 44100
DURATION_SECONDS = 60
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "audio_output")


# ── Helper Functions ──────────────────────────────────────────


def _pink_noise(n_samples: int) -> np.ndarray:
    """Generate pink noise using the Voss-McCartney algorithm."""
    n_rows = 16
    array = np.empty((n_samples, n_rows))
    array.fill(np.nan)
    array[0, :] = np.random.random(n_rows)
    array[:, 0] = np.random.random(n_samples)

    cols = np.random.geometric(0.5, n_samples)
    cols[cols >= n_rows] = 0
    rows = np.random.randint(n_samples, size=n_samples)

    for i in range(n_samples):
        col = cols[i]
        if col < n_rows:
            array[i, col] = np.random.random()

    # Forward-fill NaN values
    for col in range(n_rows):
        mask = np.isnan(array[:, col])
        idx = np.where(~mask, np.arange(n_samples), 0)
        np.maximum.accumulate(idx, out=idx)
        array[:, col] = array[idx, col]

    result = np.nansum(array, axis=1)
    result -= np.mean(result)
    max_val = np.max(np.abs(result))
    if max_val > 0:
        result /= max_val
    return result


def _sine_wave(freq: float, duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a sine wave at given frequency."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def _fade_in_out(audio: np.ndarray, fade_samples: int = 4410) -> np.ndarray:
    """Apply fade-in and fade-out to avoid clicks."""
    result = audio.copy()
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    result[:fade_samples] *= fade_in
    result[-fade_samples:] *= fade_out
    return result


def _normalize(audio: np.ndarray, peak: float = 0.8) -> np.ndarray:
    """Normalize audio to a peak amplitude."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio * (peak / max_val)
    return audio


# ── Audio Generators ──────────────────────────────────────────


def gen_nature_rain() -> np.ndarray:
    """Steady rainfall: pink noise."""
    n = SAMPLE_RATE * DURATION_SECONDS
    noise = _pink_noise(n) * 0.6
    # Add subtle low rumble (distant thunder)
    rumble = _sine_wave(40, DURATION_SECONDS) * 0.05
    t = np.linspace(0, DURATION_SECONDS, n, endpoint=False)
    # Occasional thunder swells
    for offset in [12, 28, 45]:
        swell = np.exp(-0.5 * ((t - offset) / 1.5) ** 2) * 0.15
        rumble += swell * _sine_wave(35, DURATION_SECONDS)
    return _fade_in_out(_normalize(noise + rumble))


def gen_nature_ocean() -> np.ndarray:
    """Ocean waves: amplitude-modulated pink noise with wave rhythm."""
    n = SAMPLE_RATE * DURATION_SECONDS
    noise = _pink_noise(n) * 0.5
    t = np.linspace(0, DURATION_SECONDS, n, endpoint=False)
    # Wave rhythm: slow amplitude modulation (~0.1 Hz, ~10s waves)
    wave_env = 0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t))
    ocean = noise * wave_env
    # Add low-frequency swell
    swell = _sine_wave(60, DURATION_SECONDS) * 0.08 * wave_env
    return _fade_in_out(_normalize(ocean + swell))


def gen_nature_forest() -> np.ndarray:
    """Forest ambiance: gentle noise with bird-like chirps."""
    n = SAMPLE_RATE * DURATION_SECONDS
    base = _pink_noise(n) * 0.2
    t = np.linspace(0, DURATION_SECONDS, n, endpoint=False)
    # Add random chirp bursts
    chirps = np.zeros(n)
    rng = np.random.RandomState(42)
    for _ in range(30):
        start = rng.randint(0, n - SAMPLE_RATE)
        freq = rng.uniform(2000, 5000)
        dur = rng.uniform(0.05, 0.15)
        chirp_len = int(dur * SAMPLE_RATE)
        chirp_t = np.linspace(0, dur, chirp_len, endpoint=False)
        chirp = np.sin(2 * np.pi * freq * chirp_t) * 0.1
        chirp *= np.exp(-chirp_t / (dur * 0.3))  # Quick decay
        end = min(start + chirp_len, n)
        chirps[start:end] += chirp[:end - start]
    return _fade_in_out(_normalize(base + chirps))


def gen_nature_stream() -> np.ndarray:
    """Mountain stream: bandpass-filtered white noise."""
    n = SAMPLE_RATE * DURATION_SECONDS
    white = np.random.randn(n)
    # Simple bandpass via rolling average + high-pass
    kernel_size = 20
    kernel = np.ones(kernel_size) / kernel_size
    low_passed = np.convolve(white, kernel, mode="same")
    # Add slight modulation for flowing effect
    t = np.linspace(0, DURATION_SECONDS, n, endpoint=False)
    mod = 0.7 + 0.3 * np.sin(2 * np.pi * 0.3 * t)
    stream = low_passed * mod * 0.5
    return _fade_in_out(_normalize(stream))


def gen_ambient_cafe() -> np.ndarray:
    """Coffee shop: low pink noise with random subtle clicks."""
    n = SAMPLE_RATE * DURATION_SECONDS
    base = _pink_noise(n) * 0.15
    # Random clicks (cup clinks)
    clicks = np.zeros(n)
    rng = np.random.RandomState(7)
    for _ in range(80):
        pos = rng.randint(0, n - 500)
        click = rng.randn(500) * 0.03
        click *= np.exp(-np.linspace(0, 10, 500))
        clicks[pos:pos + 500] += click
    return _fade_in_out(_normalize(base + clicks))


def gen_ambient_fireplace() -> np.ndarray:
    """Fireplace: low crackle impulses with pink noise base."""
    n = SAMPLE_RATE * DURATION_SECONDS
    base = _pink_noise(n) * 0.1
    # Crackle impulses
    crackles = np.zeros(n)
    rng = np.random.RandomState(13)
    for _ in range(200):
        pos = rng.randint(0, n - 1000)
        imp_len = rng.randint(100, 1000)
        impulse = rng.randn(imp_len) * rng.uniform(0.02, 0.08)
        impulse *= np.exp(-np.linspace(0, 5, imp_len))
        end = min(pos + imp_len, n)
        crackles[pos:end] += impulse[:end - pos]
    # Low warm hum
    hum = _sine_wave(80, DURATION_SECONDS) * 0.03
    return _fade_in_out(_normalize(base + crackles + hum))


def gen_ambient_library() -> np.ndarray:
    """Library quiet: near-silence with ultra-quiet pink noise."""
    n = SAMPLE_RATE * DURATION_SECONDS
    base = _pink_noise(n) * 0.04
    # Very occasional subtle sounds
    rng = np.random.RandomState(21)
    for _ in range(10):
        pos = rng.randint(0, n - 2000)
        rustle = rng.randn(2000) * 0.01
        rustle *= np.exp(-np.linspace(0, 8, 2000))
        base[pos:pos + 2000] += rustle
    return _fade_in_out(_normalize(base, peak=0.3))


def gen_meditation_bowls() -> np.ndarray:
    """Singing bowls: layered sine waves at bowl frequencies with slow decay."""
    n = SAMPLE_RATE * DURATION_SECONDS
    t = np.linspace(0, DURATION_SECONDS, n, endpoint=False)
    result = np.zeros(n)
    bowl_freqs = [396, 528, 639]
    # Ring bowls at intervals
    for i, freq in enumerate(bowl_freqs):
        for strike_time in range(i * 3, DURATION_SECONDS, 12):
            # Decaying sine
            mask = t >= strike_time
            decay = np.exp(-(t - strike_time) * 0.3) * mask
            result += np.sin(2 * np.pi * freq * t) * decay * 0.2
            # Add harmonic
            result += np.sin(2 * np.pi * freq * 2 * t) * decay * 0.05
    return _fade_in_out(_normalize(result))


def gen_meditation_breath() -> np.ndarray:
    """Breath guide: slow sine sweep mimicking inhale/hold/exhale rhythm."""
    n = SAMPLE_RATE * DURATION_SECONDS
    t = np.linspace(0, DURATION_SECONDS, n, endpoint=False)
    result = np.zeros(n)
    # 4s inhale, 4s hold, 6s exhale = 14s cycle
    cycle_len = 14.0
    for cycle_start in np.arange(0, DURATION_SECONDS, cycle_len):
        phase = t - cycle_start
        # Inhale (0-4s): rising frequency 200->400 Hz
        inhale_mask = (phase >= 0) & (phase < 4)
        freq_rise = 200 + 50 * phase
        inhale = np.sin(2 * np.pi * freq_rise * t) * inhale_mask * 0.15
        inhale *= np.clip(phase / 0.5, 0, 1) * inhale_mask  # Fade in
        result += inhale
        # Hold (4-8s): steady tone at 400 Hz
        hold_mask = (phase >= 4) & (phase < 8)
        result += np.sin(2 * np.pi * 350 * t) * hold_mask * 0.1
        # Exhale (8-14s): falling frequency 400->150 Hz
        exhale_mask = (phase >= 8) & (phase < 14)
        freq_fall = 350 - 30 * (phase - 8)
        exhale = np.sin(2 * np.pi * freq_fall * t) * exhale_mask * 0.12
        exhale *= np.clip((14 - phase) / 1.0, 0, 1) * exhale_mask  # Fade out
        result += exhale
    return _fade_in_out(_normalize(result))


def gen_meditation_chimes() -> np.ndarray:
    """Wind chimes: random high-frequency pings with reverb-like decay."""
    n = SAMPLE_RATE * DURATION_SECONDS
    result = np.zeros(n)
    t = np.linspace(0, DURATION_SECONDS, n, endpoint=False)
    rng = np.random.RandomState(99)
    chime_freqs = [800, 1200, 1600, 2000, 2400, 3200]
    for _ in range(40):
        strike = rng.uniform(0, DURATION_SECONDS - 2)
        freq = rng.choice(chime_freqs)
        mask = t >= strike
        decay = np.exp(-(t - strike) * 2.0) * mask
        result += np.sin(2 * np.pi * freq * t) * decay * 0.08
        # Add subtle harmonic
        result += np.sin(2 * np.pi * freq * 1.5 * t) * decay * 0.02
    # Soft breeze background
    breeze = _pink_noise(n) * 0.05
    return _fade_in_out(_normalize(result + breeze))


def gen_meditation_om() -> np.ndarray:
    """Om resonance: low drone at 136.1Hz (Om frequency) with harmonics."""
    n = SAMPLE_RATE * DURATION_SECONDS
    t = np.linspace(0, DURATION_SECONDS, n, endpoint=False)
    om_freq = 136.1  # Traditional Om frequency
    # Fundamental
    result = np.sin(2 * np.pi * om_freq * t) * 0.4
    # Harmonics
    result += np.sin(2 * np.pi * om_freq * 2 * t) * 0.15
    result += np.sin(2 * np.pi * om_freq * 3 * t) * 0.08
    result += np.sin(2 * np.pi * om_freq * 4 * t) * 0.04
    # Slow amplitude modulation (breathing rhythm)
    mod = 0.6 + 0.4 * np.sin(2 * np.pi * 0.08 * t)
    result *= mod
    return _fade_in_out(_normalize(result))


def gen_binaural_alpha() -> np.ndarray:
    """Alpha waves: L=200Hz, R=210Hz stereo (10Hz binaural beat)."""
    n = SAMPLE_RATE * DURATION_SECONDS
    t = np.linspace(0, DURATION_SECONDS, n, endpoint=False)
    left = np.sin(2 * np.pi * 200 * t) * 0.5
    right = np.sin(2 * np.pi * 210 * t) * 0.5
    stereo = np.column_stack([left, right])
    return _fade_in_out_stereo(stereo)


def gen_binaural_theta() -> np.ndarray:
    """Theta waves: L=200Hz, R=206Hz stereo (6Hz binaural beat)."""
    n = SAMPLE_RATE * DURATION_SECONDS
    t = np.linspace(0, DURATION_SECONDS, n, endpoint=False)
    left = np.sin(2 * np.pi * 200 * t) * 0.5
    right = np.sin(2 * np.pi * 206 * t) * 0.5
    stereo = np.column_stack([left, right])
    return _fade_in_out_stereo(stereo)


def gen_binaural_delta() -> np.ndarray:
    """Delta waves: L=200Hz, R=202Hz stereo (2Hz binaural beat)."""
    n = SAMPLE_RATE * DURATION_SECONDS
    t = np.linspace(0, DURATION_SECONDS, n, endpoint=False)
    left = np.sin(2 * np.pi * 200 * t) * 0.5
    right = np.sin(2 * np.pi * 202 * t) * 0.5
    stereo = np.column_stack([left, right])
    return _fade_in_out_stereo(stereo)


def _fade_in_out_stereo(
    stereo: np.ndarray, fade_samples: int = 4410
) -> np.ndarray:
    """Apply fade-in and fade-out to stereo audio."""
    result = stereo.copy()
    fade_in = np.linspace(0, 1, fade_samples).reshape(-1, 1)
    fade_out = np.linspace(1, 0, fade_samples).reshape(-1, 1)
    result[:fade_samples] *= fade_in
    result[-fade_samples:] *= fade_out
    return result


def gen_music_piano() -> np.ndarray:
    """Soft piano: simple chord progressions using sine harmonics."""
    n = SAMPLE_RATE * DURATION_SECONDS
    result = np.zeros(n)
    t = np.linspace(0, DURATION_SECONDS, n, endpoint=False)
    # Simple chord progression: C-Am-F-G
    chords = [
        [261.6, 329.6, 392.0],   # C major
        [220.0, 261.6, 329.6],   # A minor
        [174.6, 220.0, 261.6],   # F major
        [196.0, 246.9, 293.7],   # G major
    ]
    chord_dur = 4.0  # 4 seconds per chord
    for i, chord in enumerate(chords * (DURATION_SECONDS // (len(chords) * int(chord_dur)))):
        start_time = i * chord_dur
        if start_time >= DURATION_SECONDS:
            break
        for freq in chord:
            mask = (t >= start_time) & (t < start_time + chord_dur)
            # Piano-like decay
            phase = t - start_time
            env = np.exp(-phase * 0.8) * mask
            # Fundamental + harmonics (piano timbre)
            note = np.sin(2 * np.pi * freq * t) * 0.15
            note += np.sin(2 * np.pi * freq * 2 * t) * 0.05
            note += np.sin(2 * np.pi * freq * 3 * t) * 0.02
            result += note * env
    return _fade_in_out(_normalize(result))


def gen_music_ambient_electronic() -> np.ndarray:
    """Ambient electronic: slowly evolving pad with detuned sines and LFO."""
    n = SAMPLE_RATE * DURATION_SECONDS
    t = np.linspace(0, DURATION_SECONDS, n, endpoint=False)
    result = np.zeros(n)
    # Layered detuned pad
    base_freqs = [110, 165, 220, 330]
    for freq in base_freqs:
        # Slightly detuned pair for chorus effect
        result += np.sin(2 * np.pi * freq * t) * 0.1
        result += np.sin(2 * np.pi * (freq + 0.5) * t) * 0.1
    # LFO amplitude modulation
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t)
    result *= lfo
    # Slow filter sweep (simulated by crossfading harmonics)
    sweep = 0.5 + 0.5 * np.sin(2 * np.pi * 0.02 * t)
    high_harmonics = np.zeros(n)
    for freq in base_freqs:
        high_harmonics += np.sin(2 * np.pi * freq * 3 * t) * 0.03
    result += high_harmonics * sweep
    return _fade_in_out(_normalize(result))


def gen_music_acoustic() -> np.ndarray:
    """Acoustic guitar: Karplus-Strong plucked string synthesis."""
    n = SAMPLE_RATE * DURATION_SECONDS
    result = np.zeros(n)
    # Simple fingerpicking pattern: E2, B2, G3, D3, B2, G3
    pattern_freqs = [82.4, 123.5, 196.0, 146.8, 123.5, 196.0]
    note_dur = 0.5  # seconds per note
    pattern_dur = len(pattern_freqs) * note_dur

    for cycle_start in np.arange(0, DURATION_SECONDS, pattern_dur):
        for j, freq in enumerate(pattern_freqs):
            pluck_time = cycle_start + j * note_dur
            if pluck_time >= DURATION_SECONDS:
                break
            # Karplus-Strong synthesis
            period = int(SAMPLE_RATE / freq)
            noise = np.random.randn(period) * 0.3
            # Build pluck sound
            pluck_samples = int(3.0 * SAMPLE_RATE)  # 3s decay
            pluck = np.zeros(pluck_samples)
            pluck[:period] = noise
            for k in range(period, pluck_samples):
                pluck[k] = 0.996 * 0.5 * (pluck[k - period] + pluck[k - period + 1])
            # Place in output
            start_idx = int(pluck_time * SAMPLE_RATE)
            end_idx = min(start_idx + pluck_samples, n)
            actual_len = end_idx - start_idx
            result[start_idx:end_idx] += pluck[:actual_len] * 0.4

    return _fade_in_out(_normalize(result))


# ── Generator Registry ────────────────────────────────────────

GENERATORS = {
    "nature_rain": gen_nature_rain,
    "nature_ocean": gen_nature_ocean,
    "nature_forest": gen_nature_forest,
    "nature_stream": gen_nature_stream,
    "ambient_cafe": gen_ambient_cafe,
    "ambient_fireplace": gen_ambient_fireplace,
    "ambient_library": gen_ambient_library,
    "meditation_bowls": gen_meditation_bowls,
    "meditation_breath": gen_meditation_breath,
    "meditation_chimes": gen_meditation_chimes,
    "meditation_om": gen_meditation_om,
    "binaural_alpha": gen_binaural_alpha,
    "binaural_theta": gen_binaural_theta,
    "binaural_delta": gen_binaural_delta,
    "music_piano": gen_music_piano,
    "music_ambient_electronic": gen_music_ambient_electronic,
    "music_acoustic": gen_music_acoustic,
}


# ── File I/O ──────────────────────────────────────────────────


def save_wav(audio: np.ndarray, filepath: str) -> None:
    """Save audio array to WAV file."""
    sf.write(filepath, audio, SAMPLE_RATE)
    logger.info(f"  WAV saved: {filepath}")


def convert_to_mp3(wav_path: str, mp3_path: str) -> bool:
    """Convert WAV to MP3 using pydub (requires ffmpeg)."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate="128k")
        logger.info(f"  MP3 saved: {mp3_path}")
        return True
    except ImportError:
        logger.warning(
            "pydub not installed — skipping MP3 conversion. "
            "Install with: pip install pydub"
        )
        return False
    except Exception as e:
        logger.error(f"  MP3 conversion failed: {e}")
        return False


# ── Main Workflows ────────────────────────────────────────────


def generate_all(output_dir: str = OUTPUT_DIR) -> list:
    """
    Generate all 17 soundscape audio files.

    Returns:
        List of generated MP3 file paths (or WAV if MP3 conversion unavailable).
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    for soundscape_id, generator in GENERATORS.items():
        logger.info(f"Generating: {soundscape_id}")
        audio = generator()

        wav_path = os.path.join(output_dir, f"{soundscape_id}.wav")
        mp3_path = os.path.join(output_dir, f"{soundscape_id}.mp3")

        save_wav(audio, wav_path)

        if convert_to_mp3(wav_path, mp3_path):
            generated_files.append(mp3_path)
            # Clean up WAV after successful MP3 conversion
            os.remove(wav_path)
        else:
            generated_files.append(wav_path)

    logger.info(f"Generated {len(generated_files)} audio files in {output_dir}")
    return generated_files


def upload_all(bucket_name: str, prefix: str = "soundscapes/",
               source_dir: str = OUTPUT_DIR) -> int:
    """
    Upload all generated audio files to GCS.

    Returns:
        Number of files successfully uploaded.
    """
    try:
        from google.cloud import storage as gcs
    except ImportError:
        logger.error("google-cloud-storage not installed. pip install google-cloud-storage")
        return 0

    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    uploaded = 0

    for filename in os.listdir(source_dir):
        if not (filename.endswith(".mp3") or filename.endswith(".wav")):
            continue
        local_path = os.path.join(source_dir, filename)
        # Use .mp3 extension in GCS regardless
        blob_name = f"{prefix}{os.path.splitext(filename)[0]}.mp3"
        blob = bucket.blob(blob_name)

        logger.info(f"Uploading: {local_path} -> gs://{bucket_name}/{blob_name}")
        blob.upload_from_filename(local_path)
        blob.content_type = "audio/mpeg"
        blob.patch()
        uploaded += 1

    logger.info(f"Uploaded {uploaded} files to gs://{bucket_name}/{prefix}")
    return uploaded


def verify_all(bucket_name: str, prefix: str = "soundscapes/") -> dict:
    """
    Verify all 17 soundscape audio files exist in GCS.

    Returns:
        Dict of {soundscape_id: exists_bool}.
    """
    try:
        from google.cloud import storage as gcs
    except ImportError:
        logger.error("google-cloud-storage not installed.")
        return {}

    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    results = {}

    for soundscape_id in GENERATORS:
        blob_name = f"{prefix}{soundscape_id}.mp3"
        blob = bucket.blob(blob_name)
        exists = blob.exists()
        results[soundscape_id] = exists
        status = "OK" if exists else "MISSING"
        logger.info(f"  {soundscape_id}: {status}")

    found = sum(1 for v in results.values() if v)
    logger.info(f"Verification: {found}/{len(results)} files found")
    return results


# ── CLI ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate and upload LucilleLLM soundscape audio files"
    )
    parser.add_argument(
        "--bucket", type=str, default="",
        help="GCS bucket name (required for upload/verify)"
    )
    parser.add_argument(
        "--prefix", type=str, default="soundscapes/",
        help="GCS path prefix (default: soundscapes/)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Local output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Only generate audio files locally, don't upload"
    )
    parser.add_argument(
        "--upload-only", action="store_true",
        help="Only upload existing files, don't generate"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Only verify files exist in GCS"
    )

    args = parser.parse_args()

    if args.verify:
        if not args.bucket:
            logger.error("--bucket is required for --verify")
            sys.exit(1)
        verify_all(args.bucket, args.prefix)
        return

    if args.upload_only:
        if not args.bucket:
            logger.error("--bucket is required for --upload-only")
            sys.exit(1)
        upload_all(args.bucket, args.prefix, args.output_dir)
        return

    # Generate
    files = generate_all(args.output_dir)
    logger.info(f"Generation complete: {len(files)} files")

    if args.generate_only:
        return

    # Upload if bucket provided
    if args.bucket:
        upload_all(args.bucket, args.prefix, args.output_dir)
        verify_all(args.bucket, args.prefix)
    else:
        logger.info(
            "No --bucket specified. Files generated locally only. "
            "Use --bucket to upload to GCS."
        )


if __name__ == "__main__":
    main()
