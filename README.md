# Audio QC Tool

Streamlit app for reviewing conversation audio quality and transcript behavior from a Google Drive folder.

The app expects one conversation package containing:

- One mixed WAV file.
- Separate speaker-channel WAV files when available.
- One AssemblyAI JSON transcript.
- One Riverside TXT transcript.

Riverside TXT is used only to map readable speaker names onto AssemblyAI speaker labels such as `A`, `B`, and `C`. Timing metrics are calculated from AssemblyAI JSON and/or audio VAD outputs, depending on the metric.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The current local app is normally served at:

```text
http://localhost:8501
```

## Input Pairing

The app downloads supported files from the Drive link and pairs them as one conversation when the folder has one JSON and one TXT. It supports `.wav`, `.json`, and `.txt`.

Speaker names are resolved by reading Riverside speaker headers such as:

```text
marcos.f@turing.com (00:01.374)
Speaker A - Marcos
[00:01:23] Marcos: text
Marcos (00:01.374)
```

Email-style names are cleaned into display names. For example, `marcos.f@turing.com` becomes `Marcos F`.

## Core Timing Rules

AssemblyAI times are normalized from milliseconds to seconds.

For word-region calculations, the app uses a strict turn split rule:

```text
If the same speaker has an internal pause greater than 0.2 seconds, split the speech into a new region.
If the pause is 0.2 seconds or less, keep it inside the same region.
```

This 0.2 second rule is used for overlap, monologues, and cadence. Density follows the reference analyzer behavior described below.

## Audio QC: SNR And Silence Floor

Audio QC is calculated for each WAV using Silero VAD.

The app:

1. Reads the WAV.
2. Converts stereo/multichannel audio to mono.
3. Resamples to `16000 Hz`, except `8000 Hz` files stay at `8000 Hz`.
4. Runs Silero VAD to identify speech intervals.
5. Builds a speech mask from VAD intervals.
6. Splits audio samples into speech and non-speech samples.
7. Calculates RMS dBFS for speech and non-speech.

The formulas are:

```text
speech_level_dbfs = 20 * log10(RMS(speech_samples))
silence_floor_dbfs = 20 * log10(RMS(non_speech_samples))
snr_vad_db = speech_level_dbfs - silence_floor_dbfs
```

The silence floor is expected to be a negative dBFS value. Lower is quieter and better.

### Audio Thresholds

Client threshold:

```text
SNR >= 20 dB
silence_floor <= -40 dBFS
```

Safe pass threshold:

```text
SNR >= 23 dB
silence_floor <= -43 dBFS
metric confidence is not LOW
```

The safe pass threshold is intentionally stricter because VAD-based SNR can move slightly depending on room noise, breaths, cross-talk, and how much clean silence exists in the file.

### Audio Confidence

The app confidence is based on how much speech and silence Silero VAD found.

```text
HIGH:   at least 10s primary material and 5s secondary material
MEDIUM: at least 3s primary material and 1s secondary material
LOW:    less than that, or audio QC failed
```

For SNR, speech is the primary material and silence is the secondary material. For silence floor, silence is the primary material and speech is the secondary material.

If SNR or silence floor is within 1 dB of the threshold, confidence is capped down to MEDIUM because the decision is borderline.

## Density

Density checks whether speaking time is reasonably balanced across participants.

Speaker names come from Riverside, but the timing comes from AssemblyAI.

The app follows the reference analyzer approach:

```text
For each AssemblyAI utterance:
  duration = utterance.end - utterance.start

For each speaker:
  speaker_time = sum(duration for that speaker's utterances)
```

If the JSON has no AssemblyAI `utterances`, the app falls back to word-duration sum:

```text
speaker_time = sum(word.end - word.start for each word from that speaker)
```

The displayed speaking time is formatted as:

```text
mm:ss
```

The density percentage is:

```text
speaking_pct_of_duration = speaker_time / conversation_duration * 100
```

Where:

```text
conversation_duration = max(word.end) - min(word.start)
```

### Density Thresholds

For `N` speakers:

```text
expected_pct = 100 / N
min_allowed_pct = 0.5 * expected_pct
```

For exactly 2 speakers:

```text
max_allowed_pct = 1.5 * expected_pct
```

For 3 or more speakers:

```text
max_allowed_pct = 2.0 * expected_pct
```

Density status:

```text
PASS   if every speaker is within min_allowed_pct and max_allowed_pct
REVIEW if any speaker is outside that range
```

## Overlap Ratio

Overlap means two or more speakers are active at the same time.

The app can calculate overlap from two sources:

1. Separate speaker-channel audio using Silero VAD.
2. AssemblyAI word-level speaker regions.

If separate speaker-channel audio produces a non-zero overlap ratio, the app uses that as the final overlap because separate channels are usually stronger evidence of true simultaneous speech. Otherwise it uses AssemblyAI word-region overlap.

### Transcript-Based Overlap

For each speaker, the app builds speech regions from AssemblyAI words:

```text
same speaker + gap <= 0.2s  => same region
same speaker + gap > 0.2s   => new region
```

Then it performs an interval sweep across all speaker regions:

```text
If active_speaker_count >= 2, that time is overlap.
```

The app separately tracks:

```text
two_speaker_overlap
three_speaker_overlap
four_plus_speaker_overlap
```

### Audio-Channel Overlap

For separate speaker WAV files, the app runs Silero VAD on each speaker channel and treats each VAD speech interval as that speaker being active.

The same interval sweep is then used:

```text
If 2+ speaker-channel VAD intervals are active at once, that is overlap.
```

### Overlap Formula

The app uses total detected speech time as the denominator:

```text
overlap_ratio_pct = overlap_duration / total_non_silence_time * 100
```

Where:

```text
total_non_silence_time = union duration of all detected speech intervals
```

This is slightly different from dividing by full recording length. It focuses the score on speaking behavior instead of long silence before/after the session.

### Overlap Golden Standard

For natural multi-speaker conversation, the app treats some overlap as healthy.

Naturalness scoring gives the best overlap score around:

```text
7% to 15% overlap
```

The practical target is:

```text
around 10% overlap
```

Zero overlap is suspicious for natural conversation unless the recording is highly structured. Extremely high overlap can indicate cross-talk, bad diarization, or separate channels bleeding into each other.

## Monologues

A monologue is a continuous same-speaker speech region longer than:

```text
60 seconds
```

The app builds word-level speech regions using the 0.2 second internal pause rule, then counts any region with:

```text
duration > 60 seconds
```

Golden standard:

```text
0 monologues over 60 seconds is best.
```

Naturalness scoring subtracts 8 points per monologue from the monologue component.

## Cadence

Cadence means the rhythm and spontaneity of the conversation. It is not just speed. It asks whether the exchange sounds like a real conversation with back-and-forth movement rather than long scripted blocks.

The app calculates cadence from AssemblyAI word-level turns.

It measures:

- Number of turns.
- Speaker switches.
- Speaker switches per minute.
- Backchannels such as `yeah`, `right`, `ok`, `uh`, `hmm`.
- Cutoff/interruption hints such as trailing `-`, `/`, or `...`.
- Filled pauses such as `um`, `uh`, `hmm`.
- Disfluency hints such as repeated words or phrases like `you know`, `sort of`, `kind of`.
- Paralinguistic hints such as laughter, sighs, coughs, breathing markers.
- Outsized silences over 5 seconds between turns.
- Maximum turn duration.

Cadence status is:

```text
PASS if:
  speaker_switches_per_min >= 2.0
  and at least one natural cue exists
  and outsized_silence_count == 0

MANUAL REVIEW otherwise
```

Natural cues include backchannels, cutoffs, filled pauses, or disfluency hints.

Golden standard:

```text
At least 2 speaker switches per minute
Some natural short responses or disfluencies
No long empty gaps above 5 seconds
No unusually long one-speaker blocks
```

Cadence is intentionally a heuristic. It should guide listening review, not replace it.

## Natural Conversation Feel

Natural conversation feel is a composite score from 0 to 100. It estimates how natural the mixed conversation is likely to sound after listening.

It uses five components:

```text
Density balance                 25 points
Monologues                      20 points
Overlap                         20 points
Mixed audio SNR/silence         20 points
Cadence and spontaneity         15 points
Total                          100 points
```

### Density Component

```text
25 points if density_status == PASS
10 points otherwise
```

### Monologue Component

```text
monologue_score = max(0, 20 - monologue_count * 8)
```

### Overlap Component

```text
0% overlap          => 3 points
7% to 15% overlap   => 20 points
below 7% overlap    => max(5, 8 + overlap_ratio * 1.5)
above 15% overlap   => max(4, 20 - (overlap_ratio - 15) * 0.8)
```

### Audio Component

```text
20 points if safe_pass_status == SAFE PASS
16 points if client_threshold_status == PASS
8 points  if audio_qc_status == OK but thresholds do not pass
4 points  otherwise
```

### Cadence Component

```text
cadence_score =
  speaker_switches_per_min * 1.2
  + up to 3 points for backchannels
  + up to 2 points for cutoff hints
  + up to 3 points for filled pauses/disfluencies
  + up to 2 points for paralinguistic cues

maximum cadence_score = 15
```

Long silences reduce cadence:

```text
subtract up to 6 points for outsized silences
```

### Naturalness Status

```text
PASS          score >= 75
MANUAL REVIEW score >= 60 and < 75
NEEDS REVIEW  score < 60
```

Golden standard:

```text
Balanced density
No monologues above 60s
Overlap around 10%
SNR and silence floor pass
Good back-and-forth cadence
No long empty pauses
```

## Confidence Percentages

These percentages are engineering confidence estimates, not lab-certified accuracy numbers. They describe how much trust to put in the metric when the expected inputs are present and clean.

They should be treated as a QA guide:

```text
90%+  strong automated signal
75-89% useful automated signal, spot-check if borderline
60-74% heuristic, manual review recommended
below 60% weak signal, manual review required
```

| Metric | Best-Case Confidence | Normal Confidence | Low Confidence Case | Notes |
|---|---:|---:|---:|---|
| SNR using Silero VAD | 90% | 80-85% | 45-60% | Strong when there is enough clean speech and enough silence. Lower when there is music, heavy cross-talk, or very little silence. |
| Silence floor using Silero VAD | 90% | 80-85% | 45-60% | Strong when VAD finds real non-speech regions. Lower if room tone contains quiet speech bleed or if there is almost no silence. |
| Speaker name mapping | 85% | 70-80% | 50-60% | Uses Riverside names and text similarity against AssemblyAI labels. Lower when speakers say very similar words or transcript text is sparse. |
| Density | 92% | 85-90% | 60-75% | Strong when AssemblyAI utterances and speaker labels are reliable. Falls back to word-duration sum if utterances are missing. |
| Transcript overlap | 80% | 70-75% | 50-65% | Depends on diarization and word timestamps. Very short interruptions can be missed or over-smoothed. |
| Separate-channel audio overlap | 90% | 85-90% | 60-75% | Strongest overlap signal when each speaker WAV is truly isolated. Lower if channels bleed or VAD misses quiet speech. |
| Monologue count | 85% | 75-85% | 60-70% | Good when word timestamps and speaker labels are stable. Sensitive to the 0.2s split rule. |
| Cadence | 70% | 60-70% | 45-55% | Heuristic based on switches and text cues. Good for triage, not a substitute for listening. |
| Natural conversation feel | 65% | 55-65% | 40-50% | Composite heuristic. Useful for ranking and review decisions, but final judgment should include human listening. |

## Manual Review Guidance

Use manual review when:

- SNR is below 20 dB.
- Silence floor is above -40 dBFS.
- Audio confidence is LOW.
- Density is REVIEW.
- Overlap is 0% but the audio sounds overlapping.
- Overlap is very high and may be channel bleed.
- Any monologue over 60 seconds appears.
- Cadence is MANUAL REVIEW.
- Natural conversation feel is below 75.

For manual listening, check:

- Is speech easy to understand?
- Are quiet sections actually quiet?
- Is there background hum, fan noise, keyboard noise, or room echo?
- Do speaker channels leak into each other?
- Do speakers naturally respond to each other?
- Are there long solo stretches?
- Does the conversation sound spontaneous or scripted?

## Known Limitations

- SNR and silence floor depend on Silero VAD quality.
- Speaker name mapping depends on Riverside transcript format and text similarity.
- AssemblyAI diarization errors will affect density, overlap, monologues, and cadence.
- Cadence and naturalness are heuristic scores, not objective truth.
- The app gives QA signals; final acceptance should include manual listening for borderline or client-critical files.

