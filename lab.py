"""Deterministic synthetic endpoint benchmark. No models, network, or API keys."""
from pathlib import Path
import csv
import webrtcvad
import wave
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SR, HOP = 16000, 320  # 20 ms frames


def synthesize():
    """Band-limited noise bursts; labels describe activity, not human speech."""
    rng = np.random.default_rng(2027)
    cases = [
        ("clean", .12, .002, .16), ("hesitation", .12, .002, .38),
        ("long_pause", .12, .002, .70), ("quiet", .025, .002, .22),
        ("noisy", .10, .018, .24), ("very_quiet", .012, .003, .18),
        ("pause_noise", .08, .010, .52), ("fading", .06, .003, .28),
    ]
    (ROOT / "clips").mkdir(exist_ok=True)
    for name, amp, noise, pause in cases:
        end = 1.6 + pause
        n = int((end + 1.4) * SR)
        t = np.arange(n) / SR
        raw = rng.normal(size=n)
        spectrum = np.fft.rfft(raw)
        freq = np.fft.rfftfreq(n, 1 / SR)
        spectrum[(freq < 180) | (freq > 3400)] = 0
        carrier = np.fft.irfft(spectrum, n)
        carrier /= np.std(carrier)
        mask = ((t >= .30) & (t < .90)) | ((t >= .90 + pause) & (t < end))
        envelope = .65 + .35 * np.sin(2 * np.pi * 5 * t) ** 2
        if name == "fading":
            envelope *= np.maximum(.12, 1 - t / (end + .1))
        audio = amp * carrier * mask * envelope + noise * rng.normal(size=n)
        pcm = (np.clip(audio, -1, 1) * 32767).astype('<i2')
        with wave.open(str(ROOT / "clips" / (name + ".wav")), "wb") as wav:
            wav.setparams((1, 2, SR, 0, "NONE", "not compressed"))
            wav.writeframes(pcm.tobytes())
        yield name, pcm, end


def endpoint(activity, hangover):
    """First endpoint after activity: streaming decision, never revised."""
    started, silence = False, 0
    for i, active in enumerate(activity):
        if active:
            started, silence = True, 0
        elif started:
            silence += 1
            if silence >= hangover:
                return (i + 1) * HOP / SR
    return None


def main():
    rows = []
    for name, pcm, truth in synthesize():
        frames = pcm[:len(pcm) // HOP * HOP].reshape(-1, HOP)
        rms = np.sqrt(np.mean((frames.astype(float) / 32768) ** 2, axis=1))
        # Initial 300 ms is known calibration silence for this controlled lab.
        floor = np.median(rms[:15])
        vad = webrtcvad.Vad(2)
        strategies = {"fixed_200ms": (rms > .02, 10),
                      "adaptive_400ms": (rms > max(.004, 3 * floor), 20),
                      "webrtc_400ms": ([vad.is_speech(f.tobytes(), SR) for f in frames], 20)}
        for strategy, (activity, hangover) in strategies.items():
            decision = endpoint(activity, hangover)
            early = decision is not None and decision < truth
            rows.append(dict(clip=name, strategy=strategy,
                             latency_ms=None if decision is None or early else round(1000 * (decision - truth)),
                             premature_cutoff=int(early), missed=int(decision is None)))
    with (ROOT / "results.csv").open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("clip             strategy             latency_ms  cutoff  missed")
    for r in rows:
        print(f"{r['clip']:16} {r['strategy']:20} {str(r['latency_ms']):>10} {r['premature_cutoff']:7} {r['missed']:7}")
    names = list(dict.fromkeys(r['strategy'] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, metric, title in zip(axes, ['latency_ms', 'premature_cutoff', 'missed'],
                                 ['Mean successful latency (ms)', 'Premature cutoff rate (%)', 'Missed endpoint rate (%)']):
        values = []
        for name in names:
            data = [r[metric] for r in rows if r['strategy'] == name and r[metric] is not None]
            values.append(np.mean(data) * (100 if metric != 'latency_ms' else 1) if data else 0)
        ax.bar(names, values, color=['#4273b8', '#24a589', '#edaa43'])
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Synthetic endpointing lab: waiting reduces interruptions')
    fig.savefig(ROOT / "comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
