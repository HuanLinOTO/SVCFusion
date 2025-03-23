"""
Developed by: C_Zim(Chai🍊)
Date: 25/2/26
Version: 0.3
Description:
    此项目仅供娱乐，不得用于商业用途
    这只是个模板，请根据自己的需求进行修改
    请确保你有一定的混音知识和经验，否则只会越改越差
"""

import tempfile
from pedalboard import (
    Pedalboard,
    Mix,
    Gain,
    HighpassFilter,
    PeakFilter,
    HighShelfFilter,
    Delay,
    Invert,
    Compressor,
    Reverb,
    Limiter,
)
from pedalboard.io import AudioFile
import numpy

import warnings
import librosa
import soundfile


warnings.filterwarnings("ignore")


class TimeCalculator:
    def __init__(self, inst_path):
        y, sr = librosa.load(inst_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(int(tempo), 0)
        if bpm >= 100:
            bpm = bpm / 2
        self.basic_time = 60000 / bpm
        self.times = {
            "pre_delay": self.reverb_pre_delay(),
            "release": self.compressor_release(),
        }

    def _calculate_time(self, times):
        stop = 0
        for time in times:
            half_time = time / 2
            times.append(half_time)
            stop += 1
            if stop >= 15:
                break
        syns = times[:]
        return syns

    def _select_time(
        self, time_lists, standard_value, standard_range, double_mode=False
    ):
        if min(time_lists) >= standard_range:
            if double_mode:
                return standard_value * 2
            else:
                return standard_value
        else:
            min_num = float("inf")
            for time_list in time_lists:
                diff = abs(time_list - standard_value)
                if diff < min_num:
                    min_num = diff
                    closest_num = time_list
            if double_mode:
                return closest_num * 2
            else:
                return closest_num

    def _note(self, rate, mode):
        if mode == 0:
            note = self.basic_time * rate
            dot = note * 1.5
            trip = note * 2 / 3
            bases = [note, dot, trip]
            fulls = self._calculate_time(bases)
        elif mode == 1:
            note = self.basic_time / rate
            dot = note * 1.5
            trip = note * 2 / 3
            bases = [note, dot, trip]
            fulls = self._calculate_time(bases)
        sorted_times = sorted(fulls)
        return sorted_times

    def reverb_pre_delay(self):
        pre_delay_raws = self._note(8, 1)
        pre_delays = [round(pre_delay, 2) for pre_delay in pre_delay_raws]
        roomER = self._select_time(pre_delays, 0.6, 1, True)
        roomLR = self._select_time(pre_delays, 2, 4, True)
        plate = self._select_time(pre_delays, 10, 20, True)
        hall = self._select_time(pre_delays, 20, 40, True)
        return (
            roomER,
            roomLR,
            plate,
            hall,
        )

    def compressor_release(self):
        release_raws = self._note(2, 0)
        releases = [round(release, 1) for release in release_raws]
        fast = self._select_time(releases, 100, 200)
        medium = self._select_time(releases, 350, 500)
        slow = self._select_time(releases, 500, 1000)
        limiter = self._select_time(releases, 450, 800)
        return (
            fast,
            medium,
            slow,
            limiter,
        )


def load(path, sample_rate):
    with AudioFile(path).resampled_to(sample_rate) as audio:
        data = audio.read(audio.frames)
    return data


def vocal(voc_input, release=300, fb=180):
    bv = Pedalboard(
        [
            Gain(voc_input),
            HighpassFilter(230),
            PeakFilter(2700, -2, 1),
            HighShelfFilter(20000, -2, 1.8),
            Gain(1),
            PeakFilter(1400, 3, 1.15),
            PeakFilter(8500, 2.5, 1),
            Gain(-1),
            Mix(
                [
                    Gain(0),
                    Pedalboard([Invert(), Compressor(-30, 3.2, 40, fb), Gain(-40)]),
                ]
            ),
            Compressor(-18, 2.5, 19, release),
            Gain(0),
        ]
    )

    return bv


def reverb(
    revb_gain,
    s=5,
    m=25,
    l=50,
    d=200,
):
    delay = Pedalboard(
        [
            Gain(-20),
            Delay(d / 8, 0, 1),
            Gain(-12),
        ]
    )

    short = Pedalboard(
        [
            Gain(-20),
            Delay(s / 1000, 0, 1),
            Reverb(0.2, 0.35, 1, 0, 1, 0),
            Gain(-12),
        ]
    )

    medium = Pedalboard(
        [
            Gain(-16),
            Delay(m / 1000, 0.3, 1),
            Reverb(0.45, 0.55, 1, 0, 1, 0),
            Gain(-19),
        ]
    )

    long = Pedalboard(
        [Gain(-12), Delay(l / 1000, 0.6, 1), Reverb(0.6, 0.7, 1, 0, 1, 0), Gain(-23)]
    )

    br = Pedalboard(
        [
            Mix(
                [
                    short,
                    medium,
                    long,
                    delay,
                ]
            ),
            PeakFilter(1450, -4, 1.83),
            PeakFilter(2300, 5, 0.51),
            Gain(revb_gain),
        ]
    )

    return br


def instrument(headroom):
    inst = Pedalboard([Gain(headroom)])
    return inst


def master(comp_rel=500, lim_rel=400):
    mast = Pedalboard(
        [Compressor(-10, 1.6, 10, comp_rel), Limiter(-3, lim_rel), Gain(-0.5)]
    )

    return mast


def combine(vocal, revb, inst):
    min_length = min(vocal.shape[1], inst.shape[1])
    voc_new = vocal[:, :min_length]
    revb_new = revb[:, :min_length]
    inst_new = inst[:, :min_length]
    combined = voc_new + inst_new + revb_new
    return combined


def out_put(path, audio, samplerate):
    soundfile.write(path, audio.T, samplerate=samplerate, format="flac")


def automix(
    voc_path: str,
    inst_path: str,
    sample_rate: int = 44100,
    revb_gain: int = 0,
    headroom: int = -8,
    voc_input: int = -4,
):
    ts = TimeCalculator(voc_path).times
    predelay = ts["pre_delay"]
    release = ts["release"]

    voc = load(voc_path, sample_rate)
    inst = load(inst_path, sample_rate)

    fx_voc = vocal(voc_input, release[1], release[0])
    fx_revb = reverb(revb_gain, predelay[0], predelay[2], predelay[3], predelay[1])
    fx_inst = instrument(headroom)
    fx_master = master(release[3], release[2])
    eff_voc = fx_voc(voc, sample_rate)
    stereo = numpy.tile(eff_voc, (2, 1))

    revb = fx_revb(stereo, sample_rate)
    eff_inst = fx_inst(inst, sample_rate)
    combined = combine(eff_voc, revb, eff_inst)
    output = fx_master(combined, sample_rate)

    output_path = tempfile.mktemp(suffix=".flac")

    out_put(output_path, output, sample_rate)
    return output_path
