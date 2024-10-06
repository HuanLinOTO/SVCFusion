import os
import numpy as np
import random
import librosa
import torch
import argparse
import shutil
from ddspsvc.logger import utils
from tqdm import tqdm
from ddspsvc.ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from ddspsvc.diffusion.vocoder import Vocoder
from ddspsvc.logger.utils import traverse_dir


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="path to the config file"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set",
    )
    parser.add_argument(
        "-f",
        "--filelist",
        type=str,
        default=None,
        required=False,
        help="path to filelist",
    )
    # flag
    parser.add_argument(
        "--flag",
        type=str,
        default="",
        required=False,
        help="flag",
    )
    return parser.parse_args(args=args, namespace=namespace)


def preprocess(
    filelist: list[str],
    f0_extractor,
    volume_extractor,
    mel_extractor,
    units_encoder,
    sample_rate,
    hop_size,
    device="cuda",
    use_pitch_aug=False,
    flag: str = "",
):
    for filepath in filelist:
        path_srcdir = os.path.dirname(filepath)
        # srcdir的上层
        path = os.path.dirname(os.path.dirname(path_srcdir))
        path_unitsdir = os.path.join(path, "units")
        path_f0dir = os.path.join(path, "f0")
        path_volumedir = os.path.join(path, "volume")
        path_augvoldir = os.path.join(path, "aug_vol")
        path_meldir = os.path.join(path, "mel")
        path_augmeldir = os.path.join(path, "aug_mel")
        path_skipdir = os.path.join(path, "skip")
        path_durationdir = os.path.join(path, "duration")

        path_pitchaugdict = os.path.join(path, f"pitch_aug_dict_{flag}.npy")

        # pitch augmentation dictionary
        pitch_aug_dict = {}

        if os.path.exists(path_pitchaugdict) and pitch_aug_dict == {}:
            print("Load pitch augmentation dictionary from:", path_pitchaugdict)
            pitch_aug_dict = np.load(path_pitchaugdict, allow_pickle=True).item()

        # run
        def process(file):
            binfile = file + ".npy"
            path_srcfile = os.path.join(path_srcdir, file)
            path_unitsfile = os.path.join(path_unitsdir, binfile)
            path_f0file = os.path.join(path_f0dir, binfile)
            path_volumefile = os.path.join(path_volumedir, binfile)
            path_augvolfile = os.path.join(path_augvoldir, binfile)
            path_melfile = os.path.join(path_meldir, binfile)
            path_augmelfile = os.path.join(path_augmeldir, binfile)
            path_skipfile = os.path.join(path_skipdir, file)
            path_duration = os.path.join(path_durationdir, file + ".txt")

            # load audio
            audio, _ = librosa.load(path_srcfile, sr=sample_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)

            audio_t = torch.from_numpy(audio).float().to(device)
            audio_t = audio_t.unsqueeze(0)

            if not os.path.exists(path_duration):
                # extract duration
                os.makedirs(os.path.dirname(path_duration), exist_ok=True)
                duration = librosa.get_duration(filename=path_srcfile, sr=sample_rate)
                with open(path_duration, "w") as f:
                    f.write(str(duration))
            else:
                duration = float(open(path_duration).read())

            if not os.path.exists(path_f0file):
                # extract volume
                volume = volume_extractor.extract(audio)
            else:
                volume = np.load(path_volumefile, allow_pickle=True)

            # extract mel and volume augmentaion
            if mel_extractor is not None:
                if not os.path.exists(path_melfile):
                    mel_t = mel_extractor.extract(audio_t, sample_rate)
                    mel = mel_t.squeeze().to("cpu").numpy()
                else:
                    mel = np.load(path_melfile, allow_pickle=True)

                max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
                max_shift = min(1, np.log10(1 / max_amp))
                log10_vol_shift = random.uniform(-1, max_shift)
                if use_pitch_aug:
                    if file in pitch_aug_dict:
                        keyshift = pitch_aug_dict[file]
                    else:
                        keyshift = random.uniform(-5, 5)
                else:
                    keyshift = 0

                if not os.path.exists(path_augmelfile):
                    aug_mel_t = mel_extractor.extract(
                        audio_t * (10**log10_vol_shift), sample_rate, keyshift=keyshift
                    )
                    aug_mel = aug_mel_t.squeeze().to("cpu").numpy()
                else:
                    aug_mel = np.load(path_augmelfile, allow_pickle=True)

                if not os.path.exists(path_augvolfile):
                    aug_vol = volume_extractor.extract(audio * (10**log10_vol_shift))
                else:
                    aug_vol = np.load(path_augvolfile, allow_pickle=True)

            if not os.path.exists(path_unitsfile):
                # units encode
                units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
                units = units_t.squeeze().to("cpu").numpy()
            else:
                units = np.load(path_unitsfile, allow_pickle=True)

            if not os.path.exists(path_f0file):
                # extract f0
                f0 = f0_extractor.extract(audio, uv_interp=False)
            else:
                f0 = np.load(path_f0file)

            uv = f0 == 0
            if len(f0[~uv]) > 0:
                # interpolate the unvoiced f0
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

                # save npy
                os.makedirs(os.path.dirname(path_unitsfile), exist_ok=True)
                np.save(path_unitsfile, units)
                os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
                np.save(path_f0file, f0)
                os.makedirs(os.path.dirname(path_volumefile), exist_ok=True)
                np.save(path_volumefile, volume)
                if mel_extractor is not None:
                    pitch_aug_dict[file] = keyshift
                    os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
                    np.save(path_melfile, mel)
                    os.makedirs(os.path.dirname(path_augmelfile), exist_ok=True)
                    np.save(path_augmelfile, aug_mel)
                    os.makedirs(os.path.dirname(path_augvolfile), exist_ok=True)
                    np.save(path_augvolfile, aug_vol)
                    np.save(path_pitchaugdict, pitch_aug_dict)
            else:
                print("\n[Error] F0 extraction failed: " + path_srcfile)
                os.makedirs(os.path.dirname(path_skipfile), exist_ok=True)
                shutil.move(path_srcfile, os.path.dirname(path_skipfile))
                print("This file has been moved to " + path_skipfile)

        # print("Preprocess the audio clips in :", path_srcdir)

        process(os.path.basename(filepath))
        print("[!!]")

    print("Done")
    # exit(0)


if __name__ == "__main__":
    # parse commands
    cmd = parse_args()

    device = cmd.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # load config
    args = utils.load_config(cmd.config)
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size

    extensions = args.data.extensions

    # initialize f0 extractor
    f0_extractor = F0_Extractor(
        args.data.f0_extractor,
        args.data.sampling_rate,
        args.data.block_size,
        args.data.f0_min,
        args.data.f0_max,
    )

    # initialize volume extractor
    volume_extractor = Volume_Extractor(args.data.block_size)

    # initialize mel extractor
    mel_extractor = None
    use_pitch_aug = False
    if args.model.type in [
        "Diffusion",
        "DiffusionNew",
        "DiffusionFast",
        "RectifiedFlow",
    ]:
        mel_extractor = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
        if (
            mel_extractor.vocoder_sample_rate != sample_rate
            or mel_extractor.vocoder_hop_size != hop_size
        ):
            mel_extractor = None
            print("Unmatch vocoder parameters, mel extraction is ignored!")
        elif args.model.use_pitch_aug:
            use_pitch_aug = True

    # initialize units encoder
    if args.data.encoder == "cnhubertsoftfish":
        cnhubertsoft_gate = args.data.cnhubertsoft_gate
    else:
        cnhubertsoft_gate = 10
    units_encoder = Units_Encoder(
        args.data.encoder,
        args.data.encoder_ckpt,
        args.data.encoder_sample_rate,
        args.data.encoder_hop_size,
        cnhubertsoft_gate=cnhubertsoft_gate,
        device=device,
    )

    # preprocess training set
    filelist_path = cmd.filelist
    filelist = open(filelist_path, "r", encoding="utf-8").read().splitlines()

    preprocess(
        filelist=filelist,
        f0_extractor=f0_extractor,
        volume_extractor=volume_extractor,
        mel_extractor=mel_extractor,
        units_encoder=units_encoder,
        sample_rate=sample_rate,
        hop_size=hop_size,
        device=device,
        use_pitch_aug=use_pitch_aug,
        flag=cmd.flag,
    )
    exit(0)
