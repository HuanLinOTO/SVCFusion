from pathlib import Path


def get_audio_info(source: Path):
    total_num = 0
    result = {}

    for spk in (source / "audio").iterdir():
        # print(spk, calc_wavs_num_from_dir(spk))
        num = len(list(dir.glob("*.wav")))
        total_num += num

        result[spk.name] = num

    return total_num, result


def main():
    audio_per_chunk = 10000

    source = Path("data/train")

    total_num = 0


if __name__ == "__main__":
    main()
