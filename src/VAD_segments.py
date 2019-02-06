"""
Interface for PY-WEB
This code is modified from the example in https://github.com/wiseman/py-webrtcvad/blob/master/example.py
"""

import collections
import contextlib
import numpy as np
import sys
import librosa
import wave

import webrtcvad
from configuration import get_config

# get arguments from parser
config = get_config() 

def read_wave(path, sr):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    Assumes sample width == 2
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
    data, _ = librosa.load(path, sr)
    assert len(data.shape) == 1
    assert sr in (8000, 16000, 32000, 48000)
    return data, pcm_data
    
class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small amount of silence or the beginnings/endings of speech around the voiced frames.
    Input:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Output: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                start = ring_buffer[0][0].timestamp
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield (start, frame.timestamp + frame.duration)
                ring_buffer.clear()
                voiced_frames = []
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield (start, frame.timestamp + frame.duration)

def VAD_chunk(aggressiveness, path):
    """ Divide an audio wave to short speech segments.
        Input: aggressiveness - an integer between 0 and 3
               path - audio file path
        Output: list of time touples one per segment
                list of speech segments
    """
    audio, byte_audio = read_wave(path, config.sr)
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = frame_generator(20, byte_audio, config.sr)
    frames = list(frames)
    times = vad_collector(config.sr, 20, 200, vad, frames)
    speech_times = []
    speech_segs = []
    for time in times:
        start_time = np.round(time[0],decimals=2)
        end_time = np.round(time[1],decimals=2)
        speech_times.append([start_time, end_time])
        speech_segs.append(audio[int(start_time*config.sr):int(end_time*config.sr)])
    return speech_times, speech_segs
