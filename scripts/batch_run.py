#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path


IDEA_PAIRS: list[tuple[str, str]] = [
    ("a somber, vibrant oil painting of a funeral inside a church", "an oil painting of a young bride in white satin"),
    ("a somber, vibrant oil painting of a funeral procession in rain", "an oil painting of an old fisherman with tired eyes"),
    ("an oil painting of mourners lighting candles in a chapel", "an oil painting of a violinist playing alone at dawn"),
    ("an oil painting of a wake in a candlelit farmhouse", "an oil painting of a widow holding folded letters"),
    ("an oil painting of a deserted harbor at blue hour", "an oil painting of a weathered sea captain in silence"),
    ("an oil painting of a storm over a cliffside church", "an oil painting of a choir singer with trembling hands"),
    ("an oil painting of a funeral feast in a village hall", "an oil painting of a baker covered in flour and grief"),
    ("an oil painting of mourners beneath black umbrellas", "an oil painting of a florist arranging white lilies"),
    ("an oil painting of a hospital corridor at sunset", "an oil painting of a nurse staring out a rainy window"),
    ("an oil painting of a cemetery under autumn leaves", "an oil painting of a young priest with ink-stained fingers"),
    ("an oil painting of a dim tavern after bad news", "an oil painting of a sailor clutching a wedding ring"),
    ("an oil painting of a crumbling theater after final applause", "an oil painting of an aging ballerina in quiet makeup"),
    ("an oil painting of a burned farmhouse at twilight", "an oil painting of a farmer holding a rescued lamb"),
    ("an oil painting of a moonlit bridge over dark water", "an oil painting of a ferryman with a gentle stare"),
    ("an oil painting of a mountain village after an avalanche", "an oil painting of a carpenter repairing a child sled"),
    ("an oil painting of a wake beside a piano", "an oil painting of a composer with red eyes and steady posture"),
    ("an oil painting of a courtroom after a harsh verdict", "an oil painting of a mother standing unwavering"),
    ("an oil painting of a chapel choir in winter", "an oil painting of a grieving tenor with clasped hands"),
    ("an oil painting of a crowded station during farewell", "an oil painting of a soldier reading a final telegram"),
    ("an oil painting of a flooded town square at dusk", "an oil painting of a bookseller salvaging soaked poetry"),
    ("an oil painting of a memorial wall covered in flowers", "an oil painting of a child in a black coat holding a dove"),
    ("an oil painting of a monastery garden after rain", "an oil painting of a monk with a cracked lantern"),
    ("an oil painting of a shattered wedding hall", "an oil painting of a bride removing her veil in silence"),
    ("an oil painting of a lonely ferris wheel at night", "an oil painting of a carnival worker with melancholic smile"),
    ("an oil painting of a winter chapel lit by amber candles", "an oil painting of a gravedigger with compassionate eyes"),
    ("an oil painting of a moonlit wheat field before a storm", "an oil painting of a shepherdess clutching a rosary"),
    ("an oil painting of a balcony above a grieving crowd", "an oil painting of a mayor trying not to cry"),
    ("an oil painting of a family kitchen after farewell", "an oil painting of a grandmother folding black dresses"),
    ("an oil painting of a smoky jazz club at closing time", "an oil painting of a trumpet player in deep sorrow"),
    ("an oil painting of a chapel aisle lined with lilies", "an oil painting of twin sisters holding each other"),
    ("an oil painting of a fishing village in heavy fog", "an oil painting of a lighthouse keeper with prayer beads"),
    ("an oil painting of a border crossing at dawn", "an oil painting of a refugee teacher holding worn textbooks"),
    ("an oil painting of a funeral drum circle", "an oil painting of a drummer with a tearful smile"),
    ("an oil painting of a city rooftop after thunder", "an oil painting of a poet in soaked velvet coat"),
    ("an oil painting of a train platform covered in roses", "an oil painting of a porter comforting a child"),
    ("a charcoal-style cinematic still of mourners in candlelight", "a cinematic portrait of a widower clutching a framed photo"),
    ("a moody black-and-white film still of a rainy cemetery", "a black-and-white portrait of a priest with trembling lips"),
    ("a dramatic photographic scene of a chapel wake", "a dramatic portrait photo of a seamstress in funeral black"),
    ("a cinematic scene of stormy shoreline memorial candles", "a cinematic portrait of a coast guard captain in grief"),
    ("a desaturated documentary-style village memorial", "a documentary portrait of a baker serving mourners"),
    ("a gothic digital painting of a cathedral requiem", "a digital portrait of a choir soloist with wet eyelashes"),
    ("a painterly acrylic-style scene of a moonlit vigil", "an acrylic-style portrait of a young nurse with folded hands"),
    ("a noir film frame of a midnight funeral motorcade", "a noir portrait of a detective haunted by loss"),
    ("a watercolor scene of white lilies floating on dark river", "a watercolor portrait of a ferryman with gentle eyes"),
    ("a sepia-toned cinematic town square memorial", "a sepia portrait of a teacher carrying old letters"),
    ("an oil painting of a cathedral bell tower at dusk", "an oil painting of a bell-ringer with solemn expression"),
    ("an oil painting of mourners sharing bread after burial", "an oil painting of a blacksmith washing soot from his face"),
    ("an oil painting of a funeral in a sunlit countryside chapel", "an oil painting of a bride with a wilting bouquet"),
    ("an oil painting of a silent harbor with folded flags", "an oil painting of a young deckhand staring at the tide"),
    ("an oil painting of villagers carrying lanterns through fog", "an oil painting of a florist wearing mourning gloves"),
]


def parse_prompt_file(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|||" not in line:
            raise ValueError(f"Invalid line (missing '|||'): {line}")
        prompt_a, prompt_b = [x.strip() for x in line.split("|||", 1)]
        if not prompt_a or not prompt_b:
            raise ValueError(f"Invalid line (empty prompt): {line}")
        pairs.append((prompt_a, prompt_b))
    if not pairs:
        raise ValueError("No prompt pairs found in file")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run many visual-anagram generations sequentially")
    parser.add_argument("--prompt_file", type=Path, default=None, help="Text file lines: prompt_a ||| prompt_b")
    parser.add_argument("--auto_ideas", type=int, default=0, help="Generate this many built-in prompt pairs")
    parser.add_argument("--count", type=int, default=20, help="Number of images to generate")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/batch"))
    parser.add_argument("--seed_start", type=int, default=100)
    parser.add_argument("--preset", default="dreamshaper8")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model_family", default="auto")
    parser.add_argument("--view_a", default="identity")
    parser.add_argument("--view_b", default="vflip")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=6.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--python", default="python")
    args = parser.parse_args()

    if args.prompt_file is None and args.auto_ideas <= 0:
        raise ValueError("Provide --prompt_file or set --auto_ideas > 0")

    pairs: list[tuple[str, str]] = []
    if args.prompt_file is not None:
        pairs.extend(parse_prompt_file(args.prompt_file))
    if args.auto_ideas > 0:
        rng = random.Random(args.seed_start)
        ideas = IDEA_PAIRS.copy()
        rng.shuffle(ideas)
        repeat_count = (args.auto_ideas + len(ideas) - 1) // len(ideas)
        pairs.extend((ideas * repeat_count)[: args.auto_ideas])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.count):
        prompt_a, prompt_b = pairs[idx % len(pairs)]
        seed = args.seed_start + idx
        out = args.out_dir / f"anagram_{idx:03d}.png"
        out_grid = args.out_dir / f"anagram_{idx:03d}_grid.png"

        cmd = [
            args.python,
            "-m",
            "src.run",
            "--preset",
            args.preset,
            "--model_family",
            args.model_family,
            "--prompt_a",
            prompt_a,
            "--prompt_b",
            prompt_b,
            "--view_a",
            args.view_a,
            "--view_b",
            args.view_b,
            "--steps",
            str(args.steps),
            "--guidance",
            str(args.guidance),
            "--seed",
            str(seed),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--out",
            str(out),
            "--out_grid",
            str(out_grid),
        ]
        if args.model:
            cmd.extend(["--model", args.model])

        print(f"[{idx + 1}/{args.count}] seed={seed} -> {out.name}")
        subprocess.run(cmd, check=True)

    print(f"Done. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
