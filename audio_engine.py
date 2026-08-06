"""
audio_engine.py — Real-time binaural audio engine for SpatialSense (v2).

Converts YOLO detections + depth grid into spatially positioned sound sources
rendered on headphones using ILD (Interaural Level Difference) and ITD
(Interaural Time Delay) binaural spatialization.

v2 improvements over v1
───────────────────────
  • Warm triangle-wave tones via additive synthesis (not harsh pure sines)
  • Hann-windowed pulse bursts — zero-click, smooth attack/release
  • Proximity correctly maps higher depth → closer (was inverted in v1)
  • Depth range derived from grid + objects (single-object case handled)
  • EMA smoothing on all source parameters — no jumps between frames
  • Fade-in / fade-out lifecycle for appearing/disappearing sources
  • Pentatonic frequency mapping — simultaneous tones always consonant
  • Distance-dependent brightness via harmonic count (close=bright, far=muffled)
  • Proximity field replaces bare 80 Hz drone with a musical orientation tone
  • Per-source gain normalised by 1/sqrt(N) to prevent clipping at high counts

Threading model
───────────────
  Vision thread  → engine.push(AudioFrame) → _queue (maxsize=2, drops stale)
  Processor thread: _queue.get() → _process_frame() → updates _sources list
  sounddevice callback: snapshot _sources → synthesise stereo PCM

Usage
─────
  engine = AudioEngine()
  engine.start()
  engine.push(frame)   # each vision frame
  engine.stop()
"""

import math
import threading
import queue as _queue
import numpy as np
import sounddevice as sd

from audio_types import AudioFrame, DetectedObject

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE = 44100
BLOCK_SIZE  = 512           # ~11.6 ms per audio callback block

MAX_SOURCES = 4             # Perceptual cap on simultaneous positioned tones

# Threat score weights (sum to 1.0)
W_PROX = 0.55
W_CENT = 0.30
W_AREA = 0.15

# Pulse cadence: threat ∈ [0, 1] → interval ∈ [SLOW, FAST] seconds
PULSE_SLOW_SEC = 1.80       # Low threat — gentle, infrequent
PULSE_FAST_SEC = 0.18       # High threat — rapid warning

# Hann burst duration (seconds) — tuned for a pleasant "pip" that's long
# enough to perceive pitch but short enough to feel pulsed
BURST_DUR_SEC = 0.10

# ── Pentatonic frequency palette ─────────────────────────────────────────────
#
# All tones drawn from a C-major pentatonic scale (C D E G A) across octaves.
# Any combination of these sounds consonant — no dissonance when 4 sources
# play simultaneously.

CLASS_FREQ: dict[str, float] = {
    # People — warm mid-range
    "person":       392.0,   # G4
    # Vehicles — low, ominous
    "car":          130.8,   # C3
    "truck":        130.8,
    "bus":          130.8,
    "motorcycle":   164.8,   # E3
    "bicycle":      196.0,   # G3
    # Animals — bright, alert
    "cat":          659.3,   # E5
    "dog":          587.3,   # D5
    "bird":         784.0,   # G5
    "horse":        523.3,   # C5
    # Furniture / static obstacles — neutral
    "chair":        440.0,   # A4
    "dining table": 440.0,
    "couch":        440.0,
    "bench":        440.0,
    "potted plant": 329.6,   # E4
    "tv":           329.6,
    # Default
    "_default":     293.7,   # D4
}

# Proximity field tone (ambient orientation indicator)
FIELD_FREQ     = 220.0      # A3 — warm, unobtrusive tone for depth navigation
FIELD_MAX_GAIN = 0.40       # Clear, audible spatial tone for nearest obstacle sector

# 3×3 grid → spatial angles for the proximity field
_GRID_AZ = [[-60, 0, 60], [-60, 0, 60], [-60, 0, 60]]
_GRID_EL = [[ 25, 25, 25], [ 0,  0,  0], [-25, -25, -25]]

# EMA smoothing factor for source parameters (per process_frame call at ~30 Hz)
# α=0.25 → 90% convergence in ~9 frames (~300 ms) — smooth but responsive
SMOOTH_ALPHA = 0.25

# Fade-in / fade-out duration in seconds
FADE_SEC = 0.08             # 80 ms crossfade

# ── Source state ──────────────────────────────────────────────────────────────

class _Source:
    """
    One spatial audio source with smooth lifecycle management.

    All parameters are EMA-smoothed every frame by the processor thread.
    The `life` field controls fade-in (0→1) and fade-out (1→0).
    """
    __slots__ = (
        "label", "freq",
        # Current (smoothed) spatial parameters
        "azimuth", "elevation", "proximity", "threat",
        # Targets (set each frame, smoothed toward)
        "_tgt_az", "_tgt_el", "_tgt_prox", "_tgt_threat",
        # Tone synthesis state (persistent across blocks)
        "tone_phase", "pulse_phase",
        # Lifecycle
        "life",         # 0.0 = silent/new → 1.0 = fully audible
        "dying",        # True = fading out, removed when life ≤ 0
    )

    def __init__(self, label: str, freq: float,
                 az: float, el: float, prox: float, threat: float):
        self.label       = label
        self.freq        = freq
        self.azimuth     = az
        self.elevation   = el
        self.proximity   = prox
        self.threat      = threat
        self._tgt_az     = az
        self._tgt_el     = el
        self._tgt_prox   = prox
        self._tgt_threat = threat
        self.tone_phase  = 0.0
        self.pulse_phase = 0.0
        self.life        = 0.0   # starts silent, fades in
        self.dying       = False

    def set_target(self, az: float, el: float, prox: float, threat: float):
        """Set the next-frame target values (will be smoothed toward)."""
        self._tgt_az     = az
        self._tgt_el     = el
        self._tgt_prox   = prox
        self._tgt_threat = threat
        self.dying       = False   # revived if previously dying

    def smooth_step(self):
        """EMA-smooth all parameters toward their targets. Call once per frame."""
        a = SMOOTH_ALPHA
        self.azimuth   += a * (self._tgt_az     - self.azimuth)
        self.elevation += a * (self._tgt_el     - self.elevation)
        self.proximity += a * (self._tgt_prox   - self.proximity)
        self.threat    += a * (self._tgt_threat  - self.threat)


# ── DSP helpers ───────────────────────────────────────────────────────────────

def _pulse_period(threat: float) -> float:
    """Return pulse period in samples for the given threat ∈ [0, 1]."""
    sec = PULSE_SLOW_SEC + threat * (PULSE_FAST_SEC - PULSE_SLOW_SEC)
    return max(sec * SAMPLE_RATE, 1.0)


def _hann_burst_envelope(
    frames: int,
    pulse_phase: float,
    pulse_period: float,
) -> np.ndarray:
    """
    Hann-windowed burst envelope — zero-click, smooth attack and release.

    Each burst is a full Hann window (raised cosine: 0 → 1 → 0) so there
    is NO discontinuity at burst boundaries.  The burst duration is fixed;
    the gap between bursts encodes threat level (shorter gap = more urgent).
    """
    burst_samples = int(BURST_DUR_SEC * SAMPLE_RATE)
    burst_samples = min(burst_samples, int(pulse_period))  # can't exceed period

    phase_arr = (pulse_phase + np.arange(frames, dtype=np.float64)) % pulse_period
    in_burst  = phase_arr < burst_samples

    # Hann window: 0.5 * (1 - cos(2π * t / N))
    hann = np.where(
        in_burst,
        0.5 * (1.0 - np.cos(2.0 * np.pi * phase_arr / burst_samples)),
        0.0,
    )
    return hann.astype(np.float32)


def _triangle_tone(
    frames: int,
    freq: float,
    phase: float,
    n_harmonics: int,
) -> np.ndarray:
    """
    Bandlimited triangle wave via additive synthesis.

    Triangle waves contain only odd harmonics with amplitude ∝ 1/n².
    They sound warm and rounded — much more pleasant than a pure sine
    while remaining clearly pitched.

    `n_harmonics` controls brightness:
      - 2–3: muffled (simulates distance/air absorption)
      - 6–8: bright and clear (close object)
    """
    t = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    signal = np.zeros(frames, dtype=np.float64)

    for k in range(n_harmonics):
        n    = 2 * k + 1                    # odd harmonics: 1, 3, 5, 7 …
        sign = 1.0 if (k % 2 == 0) else -1.0
        harmonic_freq = n * freq

        # Stop before Nyquist to avoid aliasing
        if harmonic_freq >= SAMPLE_RATE * 0.45:
            break

        signal += sign * np.sin(2.0 * np.pi * harmonic_freq * t + n * phase) / (n * n)

    # Normalise: theoretical peak of a triangle wave from this series is π²/8
    signal *= 8.0 / (np.pi * np.pi)
    return signal.astype(np.float32)


def _spatialize(
    mono: np.ndarray,
    azimuth: float,
    elevation: float,
    proximity: float,
) -> np.ndarray:
    """
    ILD + ITD binaural spatialization.

    ILD  — sqrt power-law panning
    ITD  — Woodworth model: up to ±29 samples (~0.66 ms)
    Elev — broadband attenuation for sub-horizon sources
    Dist — proximity drives output gain (floor 0.12)
    """
    az_clamp = float(np.clip(azimuth, -90.0, 90.0))
    az_rad   = math.radians(az_clamp)

    # ILD: Interaural Level Difference
    pan = (az_clamp + 90.0) / 180.0              # 0=left … 1=right
    g_l = math.sqrt(max(1.0 - pan, 0.0))
    g_r = math.sqrt(max(pan, 0.0))

    # Elevation shadow (up to –25% for sub-horizontal)
    el_gain = 1.0 + min(float(elevation), 0.0) / 30.0 * 0.25

    # Proximity gain (closer = louder, floor keeps far objects barely audible)
    prox_clamped = float(np.clip(proximity, 0.0, 1.0))
    dist_gain = 0.12 + 0.88 * prox_clamped

    total = el_gain * dist_gain

    # ITD: Interaural Time Delay — Woodworth model
    ITD_MAX = 29   # samples at 44100 Hz ≈ 0.66 ms
    delay = int(math.sin(az_rad) * ITD_MAX)

    N    = len(mono)
    left  = np.empty(N, dtype=np.float32)
    right = np.empty(N, dtype=np.float32)

    if delay > 0:          # source to the right → left ear hears it later
        left[:delay] = 0.0
        left[delay:] = mono[:-delay] * (g_l * total)
        right[:]     = mono * (g_r * total)
    elif delay < 0:        # source to the left → right ear hears it later
        d = -delay
        right[:d] = 0.0
        right[d:] = mono[:-d] * (g_r * total)
        left[:]   = mono * (g_l * total)
    else:
        left[:]  = mono * (g_l * total)
        right[:] = mono * (g_r * total)

    return np.column_stack([left, right])


# ── Audio Engine ──────────────────────────────────────────────────────────────

class AudioEngine:
    """
    Non-blocking binaural audio engine with smooth lifecycle management.
    """

    def __init__(self, volume: float = 1.0) -> None:
        self._sources: list[_Source] = []
        self._volume = max(0.0, volume)

        # Proximity field state (ambient orientation indicator from depth grid)
        self._field_az    = 0.0
        self._field_el    = 0.0
        self._field_gain  = 0.0
        self._field_phase = 0.0
        self._field_tgt_az   = 0.0
        self._field_tgt_el   = 0.0
        self._field_tgt_gain = 0.0

        self._lock    = threading.Lock()
        self._queue: _queue.Queue[AudioFrame] = _queue.Queue(maxsize=2)
        self._running = False

        # Precompute the per-block fade increment
        # Each audio block is BLOCK_SIZE / SAMPLE_RATE seconds.
        # We want life to ramp 0→1 in FADE_SEC seconds.
        self._fade_per_block = (BLOCK_SIZE / SAMPLE_RATE) / max(FADE_SEC, 1e-6)

        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=2,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=self._audio_callback,
            latency="high",  # 'high' buffer prevents CoreAudio underflow abort on macOS under Metal/ANE GPU load
        )
        self._proc_thread = threading.Thread(
            target=self._processor_loop,
            name="audio-processor",
            daemon=True,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._proc_thread.start()
        self._stream.start()
        print("[Audio] Binaural engine started — put on headphones.")

    def stop(self) -> None:
        self._running = False
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        print("[Audio] Engine stopped.")

    def push(self, frame: AudioFrame) -> None:
        """Non-blocking push — drops oldest if full (always use freshest data)."""
        try:
            self._queue.put_nowait(frame)
        except _queue.Full:
            try:
                self._queue.get_nowait()
            except _queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except _queue.Full:
                pass

    # ── Processor thread ──────────────────────────────────────────────────────

    def _processor_loop(self) -> None:
        while self._running:
            try:
                frame = self._queue.get(timeout=0.1)
            except _queue.Empty:
                continue
            try:
                self._process_frame(frame)
            except Exception as exc:
                print(f"[Audio] Processor error: {exc}")

    def _process_frame(self, frame: AudioFrame) -> None:
        objects = frame.objects
        grid    = frame.grid_matrix

        # ── Establish the scene depth range from BOTH objects AND grid ────────
        # This prevents the single-object edge case from collapsing the range.
        all_depths: list[float] = []
        for o in objects:
            if not math.isnan(o.depth):
                all_depths.append(o.depth)
                
        for v in grid.flat:
            if v > 0 and not math.isnan(v):
                all_depths.append(float(v))

        if all_depths:
            d_max = max(all_depths)
            d_min = min(all_depths)
            d_rng = max(d_max - d_min, 1e-6)
        else:
            d_max = d_min = d_rng = 1.0

        # ── Score every detected object ──────────────────────────────────────
        scored: list[tuple[float, float, float, float, str]] = []

        for obj in objects:
            # Proximity: higher depth value = closer (after main.py inversion)
            if math.isnan(obj.depth):
                prox = 0.0
            else:
                prox = (obj.depth - d_min) / d_rng

            # Centrality: 1.0 if bbox centre == frame centre
            dx   = abs(obj.cx - 0.5)
            dy   = abs(obj.cy - 0.5)
            cent = 1.0 - math.sqrt(dx * dx + dy * dy) / (0.5 * math.sqrt(2.0))

            # Angular size
            area = float(np.clip(obj.bbox_area_frac, 0.0, 1.0))

            threat = W_PROX * prox + W_CENT * cent + W_AREA * area
            if math.isnan(threat):
                threat = 0.0

            # Spatial angle from bbox centre
            az = (obj.cx - 0.5) * 180.0      # –90 (left) … +90 (right)
            el = (0.5 - obj.cy) *  60.0      # +30 (top)  … –30 (bottom)

            scored.append((threat, az, el, prox, obj.label))

        # Keep top-N by threat
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:MAX_SOURCES]

        # ── Proximity field from grid ────────────────────────────────────────
        # Safely compute g_max ignoring NaNs
        valid_grid = grid[~np.isnan(grid)]
        g_max = float(valid_grid.max()) if valid_grid.size > 0 else 0.0
        
        if g_max > 0:
            flat_idx = int(np.nanargmax(grid))
            ri, ci   = divmod(flat_idx, 3)
            
            # Avoid division by near-zero range when depth is uniform across sectors
            if d_rng > 1e-3:
                field_prox = float(grid[ri, ci] - d_min) / d_rng
            else:
                field_prox = 0.5  # Neutral fallback when all sectors have equal depth
                
            field_prox = float(np.clip(field_prox, 0.0, 1.0))
            if math.isnan(field_prox):
                field_prox = 0.5
            
            f_az   = float(_GRID_AZ[ri][ci])
            f_el   = float(_GRID_EL[ri][ci])
            
            # Dynamic frequency mapping: closer = higher pitch, height (top vs bottom) shifts pitch
            # Base range: 220Hz (A3 - far) -> 352Hz (F4 - close)
            # Height offset: Top row = +12%, Mid = 0%, Bottom = -12%
            height_mult = 1.12 if ri == 0 else (0.88 if ri == 2 else 1.0)
            f_freq = (220.0 + 132.0 * field_prox) * height_mult

            # Smooth gain curve: gentle floor (0.25) so it never silent-drops, swells for close obstacles
            f_gain = FIELD_MAX_GAIN * (0.25 + 0.75 * field_prox)
        else:
            f_az = 0.0
            f_el = 0.0
            f_freq = FIELD_FREQ
            f_gain = FIELD_MAX_GAIN * 0.25

        # ── Update source pool (under lock) ──────────────────────────────────
        with self._lock:
            existing = {s.label: s for s in self._sources if not s.dying}

            active_labels: set[str] = set()
            for threat, az, el, prox, label in top:
                active_labels.add(label)
                freq = CLASS_FREQ.get(label, CLASS_FREQ["_default"])

                if label in existing:
                    src = existing[label]
                    src.set_target(az, el, prox, threat)
                    src.smooth_step()
                else:
                    src = _Source(label, freq, az, el, prox, threat)
                    self._sources.append(src)

            # Mark sources no longer in top-N for fade-out
            for src in self._sources:
                if src.label not in active_labels and not src.dying:
                    src.dying = True

            # Remove fully faded-out sources
            self._sources = [s for s in self._sources if s.life > 0.0 or not s.dying]

            # Smooth the proximity field targets
            self._field_tgt_az   = f_az
            self._field_tgt_el   = f_el
            self._field_tgt_gain = f_gain

            a = SMOOTH_ALPHA
            self._field_az   += a * (self._field_tgt_az   - self._field_az)
            self._field_el   += a * (self._field_tgt_el   - self._field_el)
            self._field_gain += a * (self._field_tgt_gain - self._field_gain)
            # Smooth frequency transitions
            self._field_freq = getattr(self, "_field_freq", FIELD_FREQ) + a * (f_freq - getattr(self, "_field_freq", FIELD_FREQ))

    # ── sounddevice callback ──────────────────────────────────────────────────

    def _audio_callback(self, outdata: np.ndarray, frames: int,
                        _time_info, _status) -> None:
        """
        Real-time audio callback. Must never block.

        Synthesis flow per source:
          1. Triangle-wave tone (additive, brightness ~ proximity)
          2. Hann-windowed pulse envelope (cadence ~ threat)
          3. ILD + ITD spatialization
          4. Life-based fade gain (smooth appear/disappear)
        """
        try:
            mixed = np.zeros((frames, 2), dtype=np.float32)
            fade_delta = self._fade_per_block

            # ── Snapshot & advance state (brief lock) ─────────────────────────
            with self._lock:
                snaps: list[tuple] = []
                n_active = 0

                for s in self._sources:
                    # Advance life (fade-in or fade-out)
                    if s.dying:
                        s.life = max(s.life - fade_delta, 0.0)
                    else:
                        s.life = min(s.life + fade_delta, 1.0)

                    if s.life <= 0.0:
                        continue   # fully faded out, skip synthesis

                    n_active += 1
                    pp = _pulse_period(s.threat)
                    snaps.append((
                        s.freq, s.azimuth, s.elevation, s.proximity,
                        s.threat, s.tone_phase, s.pulse_phase, pp, s.life,
                    ))

                    # Advance tone phase
                    s.tone_phase = (
                        s.tone_phase + 2.0 * math.pi * s.freq * frames / SAMPLE_RATE
                    ) % (2.0 * math.pi)
                    # Advance pulse phase
                    s.pulse_phase = (s.pulse_phase + frames) % pp

                # Proximity field snapshot
                f_az    = self._field_az
                f_el    = self._field_el
                f_gain  = self._field_gain
                f_freq  = getattr(self, "_field_freq", FIELD_FREQ)
                f_phase = self._field_phase
                self._field_phase = (
                    self._field_phase
                    + 2.0 * math.pi * f_freq * frames / SAMPLE_RATE
                ) % (2.0 * math.pi)

            # ── Per-source gain normalisation ─────────────────────────────────
            # 1/sqrt(N) prevents N sources from being N× louder than 1 source
            source_norm = 1.0 / math.sqrt(max(n_active, 1))

            # ── Synthesise object sources (outside lock) ──────────────────────
            for (freq, az, el, prox, threat,
                 tone_phase, pulse_phase, pp, life) in snaps:

                # Brightness: close objects get more harmonics (brighter/sharper),
                # far objects get fewer (muffled, simulates air absorption)
                n_harmonics = max(2, int(2 + prox * 6))   # 2 (far) … 8 (close)

                mono = _triangle_tone(frames, freq, tone_phase, n_harmonics)

                # Hann-windowed pulse envelope
                env = _hann_burst_envelope(frames, pulse_phase, pp)
                mono = mono * env

                # Source gain: threat level × life (fade) × normalisation
                gain = float(np.clip(threat, 0.15, 1.0)) * life * source_norm

                mixed += _spatialize(mono, az, el, prox) * gain

            # ── Proximity field (outside lock) ────────────────────────────────
            if f_gain > 0.001:
                # Warm marimba/pad synthesis (fundamental + warm harmonics)
                field_tone = _triangle_tone(frames, f_freq, f_phase, 3)
                mixed += _spatialize(field_tone, f_az, f_el, 0.4) * f_gain

            # ── Apply Volume Multiplier ────────────────────────────────────────
            mixed *= self._volume

            # ── Soft Limiter (Smooth Tanh Saturation - zero digital clipping) ─
            peak = float(np.max(np.abs(mixed)))
            limit = 0.85 * max(1.0, self._volume)
            if peak > limit:
                mixed = np.tanh(mixed / limit) * limit

            outdata[:] = mixed

        except Exception as exc:
            print(f"[Audio Callback Error] {exc}")
            outdata[:] = 0.0