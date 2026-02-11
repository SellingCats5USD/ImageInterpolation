#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path


IDEA_PAIRS: list[tuple[str, str]] = [
    ("an oil painting of a lighthouse in a storm", "an oil painting of an old sailor portrait"),
    ("a cinematic photo of a neon city street in rain", "a cinematic photo of a masked violinist"),
    ("a watercolor painting of koi fish in a garden pond", "a watercolor painting of a dragon curled in clouds"),
    ("a surreal desert with giant hourglasses", "a surreal close-up portrait of a clockmaker"),
    ("a baroque cathedral interior lit by candles", "a regal baroque portrait of a queen"),
    ("an astronaut walking through a frozen forest", "a detailed portrait of a white wolf with stars in fur"),
    ("a fantasy castle floating above the sea", "a fantasy portrait of a sea witch with glowing eyes"),
    ("a retro 80s arcade room", "a retro 80s synthwave musician portrait"),
    ("a tranquil japanese temple in autumn", "a samurai portrait with red maple leaves"),
    ("a dramatic volcanic landscape at dusk", "a dramatic portrait of a blacksmith covered in ash"),
    ("a cozy bookstore with warm lamp light", "a cozy portrait of a novelist with ink-stained hands"),
    ("a biomechanical city skyline", "a biomechanical android portrait"),
    ("an enchanted mushroom forest at night", "an enchanted portrait of a forest guardian"),
    ("an underwater palace with rays of light", "an underwater empress portrait with pearl crown"),
    ("a moody victorian train station", "a victorian detective portrait"),
    ("a snowy alpine village at sunrise", "a mountaineer portrait with frost on beard"),
    ("a cyberpunk rooftop garden", "a cyberpunk botanist portrait"),
    ("a galaxy swirling inside a glass bottle", "a mystical astronomer portrait"),
    ("a steampunk airship over canyons", "a steampunk captain portrait"),
    ("a gothic library of forbidden books", "a gothic sorcerer portrait"),
    ("a watercolor meadow with wild horses", "a watercolor portrait of a horse whisperer"),
    ("a post-apocalyptic overgrown city", "a survivor portrait with cracked goggles"),
    ("an art nouveau cafe scene", "an art nouveau singer portrait"),
    ("a moonlit beach with bioluminescent waves", "a moonlit portrait of a pearl diver"),
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
    parser.add_argument("--preset", default="sdxl")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model_family", default="auto")
    parser.add_argument("--view_a", default="identity")
    parser.add_argument("--view_b", default="vflip")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=6.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
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
