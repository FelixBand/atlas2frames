#!/usr/bin/env python3
"""
atlas2sheet.py - Rebuild Adobe Animate texture-atlas animations into
frame-by-frame PNGs.

Input : a folder containing Animation.json + spritemap1.json/png, spritemap2.json/png ...
        (or any parent folder; every Animation.json found below it is processed)
Output: per animation -> <name>_sheet.png, <name>_sheet.json, and optionally frames/<name>/0000.png ...

Requires: pip install pillow numpy

Examples
    python atlas2sheet.py "path\\to\\goldp1\\atlas" -o out --list
    python atlas2sheet.py "path\\to\\goldp1\\atlas" -o out --frames
    python atlas2sheet.py "path\\to\\characters" -o out --global-bounds --scale 1
"""
import argparse
import json
import math
import re
import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np
from PIL import Image

# Long-form key names (non-optimized exports) -> short names used internally.
LONG = {
    "ANIMATION": "AN", "SYMBOL_DICTIONARY": "SD", "Symbols": "S", "SYMBOL_name": "SN",
    "TIMELINE": "TL", "LAYERS": "L", "Layer_name": "LN", "Layer_type": "LT",
    "Clipped_by": "Clpb", "FRAMES": "FR", "index": "I", "duration": "DU",
    "elements": "E", "SYMBOL_Instance": "SI", "ATLAS_SPRITE_instance": "ASI",
    "Matrix3D": "M3D", "symbolType": "ST", "firstFrame": "FF", "loop": "LP",
    "transformationPoint": "TRP", "Instance_Name": "IN", "color": "C",
    "name": "N", "STAGE_INSTANCE": "STI", "metadata": "MD", "framerate": "FRT",
    # some exports spell these differently
    "Frames": "FR", "Clipped_by": "Clpb",
    # long-form colour effects
    "mode": "M", "alphaMultiplier": "AM", "brightness": "BRT",
    "tintColor": "TC", "tintMultiplier": "TM",
    "redMultiplier": "RM", "greenMultiplier": "GM", "blueMultiplier": "BM",
    "redOffset": "RO", "greenOffset": "GO", "blueOffset": "BO", "alphaOffset": "AO",
}

# Long-form string values -> the short codes used internally.
VALUES = {
    "ST": {"graphic": "G", "movieclip": "MC", "button": "B"},
    "LP": {"loop": "LP", "playonce": "PO", "singleframe": "SF"},
    "M": {"alpha": "CA", "brightness": "CBRT", "tint": "T", "advanced": "AD"},
}


def normalize(o):
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            k = LONG.get(k, k)
            v = normalize(v)
            if isinstance(v, str) and k in VALUES:
                v = VALUES[k].get(v.lower(), v)
            out[k] = v
        return out
    if isinstance(o, list):
        return [normalize(v) for v in o]
    return o


def load_json(p):
    with open(p, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def parse_matrix(m):
    if m is None:
        return np.eye(3)
    if isinstance(m, dict):  # long-form Matrix3D
        a, b, c, d = m.get("m00", 1), m.get("m01", 0), m.get("m10", 0), m.get("m11", 1)
        tx, ty = m.get("m30", 0), m.get("m31", 0)
    elif len(m) == 16:       # column-major 4x4
        a, b, c, d, tx, ty = m[0], m[1], m[4], m[5], m[12], m[13]
    else:                    # [a b c d tx ty]
        a, b, c, d, tx, ty = m[:6]
    return np.array([[a, c, tx], [b, d, ty], [0, 0, 1]], dtype=float)


IDENT_FX = ((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0))


def parse_fx(c):
    """Colour effect -> (multipliers RGBA 0..1, offsets RGBA 0..255)."""
    if not c:
        return IDENT_FX
    mode = c.get("M")
    if mode in ("CA", "AA"):
        return ((1, 1, 1, c.get("AM", 1)), (0, 0, 0, 0))
    if mode in ("CBRA", "CBRT"):
        b = c.get("BRT", 0)
        if b >= 0:
            return ((1 - b,) * 3 + (1,), (255 * b,) * 3 + (0,))
        return ((1 + b,) * 3 + (1,), (0, 0, 0, 0))
    if mode == "T":
        tc = c.get("TC", "#000000").lstrip("#")
        tm = c.get("TM", 0)
        r, g, b = (int(tc[i:i + 2], 16) for i in (0, 2, 4))
        return ((1 - tm,) * 3 + (1,), (r * tm, g * tm, b * tm, 0))
    if mode == "AD":
        return ((c.get("RM", 1), c.get("GM", 1), c.get("BM", 1), c.get("AM", 1)),
                (c.get("RO", 0), c.get("GO", 0), c.get("BO", 0), c.get("AO", 0)))
    return IDENT_FX


def combine_fx(parent, child):
    if parent == IDENT_FX:
        return child
    if child == IDENT_FX:
        return parent
    mp, ap = np.array(parent[0]), np.array(parent[1])
    mc, ac = np.array(child[0]), np.array(child[1])
    return (tuple(mc * mp), tuple(ac * mp + ap))


def apply_fx(tile, fx):
    mul, add = np.array(fx[0], np.float32), np.array(fx[1], np.float32)
    arr = np.asarray(tile).astype(np.float32) * mul + add
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


class Atlas:
    def __init__(self, folder, rot_cw=False):
        self.folder = Path(folder)
        data = normalize(load_json(self.folder / "Animation.json"))
        self.an = data.get("AN", {})
        self.symbols = {s["SN"]: s for s in data.get("SD", {}).get("S", [])}
        self.fps = data.get("MD", {}).get("FRT", 24)
        self.sprites = {}
        self._starts = {}
        self._len = {}
        self.warned = set()
        for jf in sorted(self.folder.glob("spritemap*.json")):
            png = jf.with_suffix(".png")
            if not png.exists():
                print(f"  ! missing {png.name}", file=sys.stderr)
                continue
            sheet = Image.open(png).convert("RGBA")
            for entry in load_json(jf)["ATLAS"]["SPRITES"]:
                s = entry["SPRITE"]
                crop = sheet.crop((s["x"], s["y"], s["x"] + s["w"], s["y"] + s["h"]))
                if s.get("rotated"):
                    # Animate stores rotated sprites turned 90 deg clockwise; undo that.
                    crop = crop.transpose(Image.ROTATE_270 if rot_cw else Image.ROTATE_90)
                self.sprites[s["name"]] = crop
        if not self.sprites:
            raise SystemExit(f"No spritemap*.json found in {self.folder}")

    def warn(self, key, msg):
        if key not in self.warned:
            self.warned.add(key)
            print(f"  ! {msg}", file=sys.stderr)

    # ---- timelines ------------------------------------------------------
    def main_timeline(self):
        if "TL" in self.an:
            return self.an["TL"]
        sym = self.symbols.get(self.an.get("SN"))
        if sym:
            return sym["TL"]
        raise SystemExit("Could not locate the main timeline in Animation.json")

    
    def tl_len(self, tl):
        k = id(tl)

        if k not in self._len:
            ends = (
                f.get("I", 0) + f.get("DU", 1)
                for layer in tl.get("L", [])
                if not self.is_helper_layer(layer)
                for f in layer.get("FR", [])
            )
            self._len[k] = max(ends, default=1)

        return max(1, self._len[k])


    def frame_at(self, layer, t):
        frs = layer.get("FR", [])
        if not frs:
            return None
        starts = self._starts.get(id(layer))
        if starts is None:
            starts = self._starts[id(layer)] = [f["I"] for f in frs]
        i = bisect_right(starts, t) - 1
        if i < 0:
            return None
        f = frs[i]
        return f if t < f["I"] + f["DU"] else None

    def labels(self, tl):
        out = []
        for layer in tl["L"]:
            for f in layer.get("FR", []):
                if f.get("N"):
                    out.append((f["N"], f["I"], f["I"] + f["DU"]))
        return sorted(out, key=lambda x: x[1])

    # ---- flattening -----------------------------------------------------
    def collect(self, tl, t, M, fx, out, depth=0):
        """Append (sprite_name, 3x3 matrix, fx) draws for timeline tl at time t."""
        if depth > 32:
            return
        for layer in reversed(tl["L"]):  # first layer = top, so draw last
            if self.is_helper_layer(layer):
                continue
            if layer.get("LT") in ("Clp", "Clipper"):
                self.warn("mask", "mask layers found: masks are NOT applied (clipped art drawn unmasked)")
                continue
            frame = self.frame_at(layer, t)
            if not frame:
                continue
            for e in frame.get("E", []):
                if "ASI" in e:
                    a = e["ASI"]
                    matrix = a.get("MX", a.get("M3D"))
                    out.append((a["N"], M @ parse_matrix(matrix), fx))
                elif "SI" in e:
                    si = e["SI"]
                    sym = self.symbols.get(si["SN"])
                    if not sym:
                        self.warn("sym:" + si["SN"], f"missing symbol {si['SN']}")
                        continue
                    stl = sym["TL"]
                    n = self.tl_len(stl)
                    st, ff, lp = si.get("ST", "G"), si.get("FF", 0), si.get("LP", "LP")
                    local = t - frame["I"]
                    if st == "G":
                        if lp == "SF":
                            sf = ff
                        elif lp == "PO":
                            sf = min(ff + local, n - 1)
                        else:
                            sf = (ff + local) % n
                    else:  # movie clip / button: loops on the parent's clock
                        sf = t % n
                    
                    # Compose the symbol transform with its registration-point offset.
                    si_matrix = parse_matrix(si.get("M3D"))

                    trp = si.get("TRP") or {}
                    if isinstance(trp, dict):
                        px = float(trp.get("x", 0))
                        py = float(trp.get("y", 0))
                    elif isinstance(trp, (list, tuple)) and len(trp) >= 2:
                        px, py = float(trp[0]), float(trp[1])
                    else:
                        px = py = 0.0

                    # Move the symbol's registration point to its local origin.
                    pivot_offset = np.array([
                        [1.0, 0.0, -px],
                        [0.0, 1.0, -py],
                        [0.0, 0.0, 1.0],
                    ])

                    matrix = si.get("MX", si.get("M3D"))

                    self.collect(
                        stl,
                        sf,
                        M @ parse_matrix(matrix),
                        combine_fx(fx, parse_fx(si.get("C"))),
                        out,
                        depth + 1,
                    )


    def draws_bounds(self, draws):
        xs, ys = [], []
        for name, M, _ in draws:
            spr = self.sprites.get(name)
            if spr is None:
                self.warn("spr:" + name, f"sprite '{name}' not found in spritemaps")
                continue
            w, h = spr.size
            for px, py in ((0, 0), (w, 0), (0, h), (w, h)):
                v = M @ np.array([px, py, 1.0])
                xs.append(v[0]); ys.append(v[1])
        if not xs:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    # ---- rasterising ----------------------------------------------------
    def render(self, draws, W, H, ox, oy, scale):
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        S = np.array([[scale, 0, -ox], [0, scale, -oy], [0, 0, 1.0]])
        for name, M, fx in draws:
            spr = self.sprites.get(name)
            if spr is None:
                continue
            F = S @ M
            w, h = spr.size
            pts = [F @ np.array([px, py, 1.0]) for px, py in ((0, 0), (w, 0), (0, h), (w, h))]
            x0 = max(int(math.floor(min(p[0] for p in pts))), 0)
            y0 = max(int(math.floor(min(p[1] for p in pts))), 0)
            x1 = min(int(math.ceil(max(p[0] for p in pts))), W)
            y1 = min(int(math.ceil(max(p[1] for p in pts))), H)
            if x1 <= x0 or y1 <= y0:
                continue
            try:
                inv = np.linalg.inv(F)
            except np.linalg.LinAlgError:
                continue  # collapsed (zero-scale) sprite
            data = (inv[0, 0], inv[0, 1], inv[0, 0] * x0 + inv[0, 1] * y0 + inv[0, 2],
                    inv[1, 0], inv[1, 1], inv[1, 0] * x0 + inv[1, 1] * y0 + inv[1, 2])
            tile = spr.transform((x1 - x0, y1 - y0), Image.AFFINE, data, resample=Image.BICUBIC)
            if fx != IDENT_FX:
                tile = apply_fx(tile, fx)
            canvas.alpha_composite(tile, (x0, y0))
        return canvas

    def is_helper_layer(self, layer):
        return (
            layer.get("LN", "").strip().casefold()
            == "makingsurethisisallincluded".casefold()
        )

def safe(name):
    # Use the final component of names such as "_EXPORT/Mono down".
    name = str(name).replace("\\", "/").rsplit("/", 1)[-1]

    # Turn spaces and punctuation into underscores.
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return name.strip("_").lower() or "anim"


def process(folder, outdir, args):
    print(f"\n== {folder}")
    atlas = Atlas(folder, rot_cw=args.rot_cw)
    main = atlas.main_timeline()

    
    # Discover animations to export: (name, timeline, start, end)
    anims = []

    if args.symbol:
        # Explicitly requested symbol.
        sym = atlas.symbols.get(args.symbol)
        if not sym:
            raise SystemExit(
                f"Symbol '{args.symbol}' not found. "
                f"Available: {', '.join(atlas.symbols)}"
            )
        anims.append((
            args.symbol,
            sym["TL"],
            0,
            atlas.tl_len(sym["TL"]),
        ))

    elif args.whole:
        # Explicitly export the main timeline.
        anims.append((
            atlas.an.get("SN") or "main",
            main,
            0,
            atlas.tl_len(main),
        ))

    else:
        # Prefer named exported symbols, such as "_EXPORT/Mono down".
        for symbol_name, sym in sorted(
            atlas.symbols.items(),
            key=lambda item: item[0].casefold(),
        ):
            normalized_name = symbol_name.replace("\\", "/")

            if not normalized_name.casefold().startswith("_export/"):
                continue

            tl = sym.get("TL")
            if not isinstance(tl, dict):
                continue

            anims.append((
                symbol_name,
                tl,
                0,
                atlas.tl_len(tl),
            ))

        # Fall back to frame labels if no exported symbols exist.
        if not anims:
            labels = atlas.labels(main)

            if labels:
                seen = {}
                for label, start, end in labels:
                    seen[label] = seen.get(label, 0) + 1
                    name = (
                        label if seen[label] == 1
                        else f"{label}_{seen[label]}"
                    )
                    anims.append((name, main, start, end))
            else:
                anims.append((
                    atlas.an.get("SN") or "main",
                    main,
                    0,
                    atlas.tl_len(main),
                ))

    if args.only:
        anims = [
            a for a in anims
            if any(
                term.casefold() in a[0].casefold()
                or term.casefold() in safe(a[0]).casefold()
                for term in args.only
            )
        ]

    if args.list:
        for name, _, start, end in anims:
            print(
                f"  {safe(name):30s} "
                f"frames {start}-{end - 1} ({end - start} frames)"
            )
        return

    # flatten every frame into draw lists
    flat = {}
    for n, tl, s, e in anims:
        flat[n] = [(lambda t: (lambda o: (atlas.collect(tl, t, np.eye(3), IDENT_FX, o), o)[1])([]))(t)
                   for t in range(s, e)]
    bounds = {n: [atlas.draws_bounds(d) for d in frames] for n, frames in flat.items()}

    def union(bs):
        bs = [b for b in bs if b]
        if not bs:
            return None
        return (min(b[0] for b in bs), min(b[1] for b in bs),
                max(b[2] for b in bs), max(b[3] for b in bs))

    gb = union([b for bl in bounds.values() for b in bl]) if args.global_bounds else None
    outdir.mkdir(parents=True, exist_ok=True)

    
    
    used_names = set()

    for n, _, s, e in anims:
        b = gb or union(bounds[n])

        if not b:
            print(f"  - {safe(n)}: empty, skipped")
            continue

        sc = args.scale

        # Calculate a consistent canvas for every frame of this animation.
        ox = math.floor(b[0] * sc) - args.pad
        oy = math.floor(b[1] * sc) - args.pad
        right = math.ceil(b[2] * sc) + args.pad
        bottom = math.ceil(b[3] * sc) + args.pad

        W = max(1, right - ox)
        H = max(1, bottom - oy)

        # Create a clean, unique folder name.
        base = safe(n)
        candidate = base
        suffix = 2

        while candidate.casefold() in used_names:
            candidate = f"{base}_{suffix}"
            suffix += 1

        used_names.add(candidate.casefold())

        # One folder per animation; filenames include the animation name.
        fd = outdir / candidate
        fd.mkdir(parents=True, exist_ok=True)

        for i, draws in enumerate(flat[n]):
            frame = atlas.render(draws, W, H, ox, oy, sc)
            frame.save(fd / f"{candidate}{i:04d}.png")

        print(
            f"  + {candidate}: {len(flat[n])} frames "
            f"({W}x{H}) -> {fd}"
        )


def main():
    ap = argparse.ArgumentParser(description="Rebuild Animate texture-atlas animations into sprite sheets.")
    ap.add_argument("input", help="atlas folder, or a parent folder to search recursively")
    ap.add_argument("-o", "--out", default="out", help="output folder (default: out)")
    ap.add_argument("--list", action="store_true", help="list animations and exit")
    ap.add_argument("--only", nargs="+", help="only animations whose name contains one of these")
    ap.add_argument("--whole", action="store_true", help="export the whole main timeline as one animation")
    ap.add_argument("--symbol", help="export a specific symbol's timeline by name")
    ap.add_argument("--frames", action="store_true", help="also save individual frame PNGs")
    ap.add_argument("--scale", type=float, default=1.0, help="output scale (default 1.0)")
    ap.add_argument("--pad", type=int, default=2, help="padding around frames in px (default 2)")
    ap.add_argument("--global-bounds", action="store_true",
                    help="use one cell size/origin for ALL animations (keeps the character anchored)")
    ap.add_argument("--rot-cw", action="store_true",
                    help="flip the un-rotation direction if rotated sprites come out sideways")
    args = ap.parse_args()

    root = Path(args.input)
    found = sorted({p.parent for p in root.rglob("Animation.json")})
    if not found:
        raise SystemExit("No Animation.json found under that path.")
    out = Path(args.out)
    for folder in found:
        sub = out
        if len(found) > 1:
            rel = folder.relative_to(root).parts
            # drop a trailing 'atlas' folder so you get e.g. out/goldp1/
            rel = rel[:-1] if rel and rel[-1].lower() == "atlas" and len(rel) > 1 else rel
            sub = out / "_".join(rel) if rel else out
        process(folder, sub, args)


if __name__ == "__main__":
    main()