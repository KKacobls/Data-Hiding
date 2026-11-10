#!/usr/bin/env python3
"""TPE-DNA teaching demo: 16x16 E-region RLE/DE/LSB embedding.

Paper-inspired reversible information-embedding stage only, NOT full DNA encryption.
Teaching-only RLE packing and externally supplied class map are declared in README.
Requires Python 3.9+ standard library only.
"""
import argparse
import collections
import json
import math
import random
from pathlib import Path

SIDE = 16
DEFAULT_SEED = 20261110
SPECIAL = [(120, 120), (185, 182), (225, 200), (250, 10), (0, 255), (120, 120)]


def ceil_half(x):
    return -(-x // 2)


def pixels(avg, diff):
    return (avg + ceil_half(diff), avg - diff // 2)


def change_pair(x, y, bit, method):
    avg, diff = (x + y) // 2, x - y
    newdiff = 2 * diff + bit if method == "DE" else 2 * (diff // 2) + bit
    return pixels(avg, newdiff)


def safe(pair):
    return all(0 <= n <= 255 for n in pair)


def category(x, y):
    avg, diff = (x + y) // 2, x - y
    if all(safe(change_pair(x, y, bit, "DE")) for bit in (0, 1)):
        if diff in (0, -1):
            return "EZ"
        if all(safe(pixels(avg, 4 * diff + 2 * a + b))
               for a in (0, 1) for b in (0, 1)):
            return "EN1"
        return "EN2"
    if all(safe(change_pair(x, y, bit, "LSB")) for bit in (0, 1)):
        return "CN"
    return "NC"


def e_pairs():
    return [(r, c) for r in range(1, 15) for c in range(1, 15, 2)]


def a_pixels():
    return [(r, c) for r in range(16) for c in range(16)
            if r in (0, 15) or c in (0, 15)]


def get(m, pos):
    r, c = pos
    return m[r][c], m[r][c + 1]


def put(m, pos, pair):
    r, c = pos
    m[r][c], m[r][c + 1] = pair


def generate(seed):
    rng = random.Random(seed)
    m = [[0] * SIDE for _ in range(SIDE)]
    # Nonuniform 4..63 values, preserving MSB2="00" for compressible A data.
    for r, c in a_pixels():
        m[r][c] = rng.randint(4, 63)
    for i, pos in enumerate(e_pairs()):
        if i < len(SPECIAL):
            pair = SPECIAL[i]
        else:
            r, c = pos
            avg = round(124 + 45 * math.sin((r + 1) * .45)
                        + 37 * math.cos((c + 2) * .58) + rng.randint(-21, 21))
            avg = max(46, min(207, avg))
            candidates = list(range(-18, -1)) + list(range(2, 19))
            rng.shuffle(candidates)
            pair = next((pixels(avg, h) for h in candidates
                         if safe(pixels(avg, h)) and category(*pixels(avg, h)) == "EN1"), None)
            if pair is None:
                raise AssertionError("Could not construct safe EN1 pair")
        put(m, pos, pair)
    return m


def make_aux(m, categories):
    marker = "".join("1" if c in ("EZ", "EN1") else "0" for c in categories)
    lsb = "".join(str((get(m, pos)[0] - get(m, pos)[1]) & 1)
                  for pos, c in zip(e_pairs(), categories) if c in ("EN2", "CN"))
    msb = "".join(f"{m[r][c]:08b}"[:2] for r, c in a_pixels())
    return marker + lsb + msb, marker, lsb, msb


def rle(raw):
    result = []
    for char in raw:
        digit = int(char)
        if result and result[-1][0] == digit:
            result[-1] = (digit, result[-1][1] + 1)
        else:
            result.append((digit, 1))
    return result


def pack(groups):
    if any(n < 1 or n > 127 for bit, n in groups):
        raise ValueError("One RLE count exceeds the 7-bit format")
    return "".join(f"{bit:1b}{n:07b}" for bit, n in groups)


def unpack(payload, count):
    if len(payload) != count * 8:
        raise ValueError("Bad RLE payload size")
    return "".join(payload[i] * int(payload[i + 1:i + 8], 2)
                   for i in range(0, len(payload), 8))


def embed(m, categories, bits):
    out = [row[:] for row in m]
    rows = []
    used = 0
    for index, (pos, cls) in enumerate(zip(e_pairs(), categories), 1):
        orig = get(m, pos)
        bit = None
        new = orig
        if cls != "NC" and used < len(bits):
            bit = int(bits[used])
            used += 1
            new = change_pair(*orig, bit, "DE" if cls in ("EZ", "EN1") else "LSB")
            assert safe(new)
            put(out, pos, new)
        rows.append(dict(group=index, row=pos[0] + 1, column=pos[1] + 1,
                         original=list(orig), category=cls, bit=bit, embedded=list(new)))
    assert used == len(bits), f"Capacity too small: {used}/{len(bits)}"
    return out, rows


def recover(pre, classes, nbits, ngroups):
    # Important: class map is supplied externally in this teaching version.
    bits, processed = "", []
    for index, (pos, cls) in enumerate(zip(e_pairs(), classes)):
        if len(bits) == nbits:
            break
        if cls == "NC":
            continue
        diff = get(pre, pos)[0] - get(pre, pos)[1]
        bits += str(diff & 1)
        processed.append((index, cls, diff))
    assert len(bits) == nbits
    raw = unpack(bits, ngroups)
    n = len(classes)
    n_lsb = sum(c in ("EN2", "CN") for c in classes)
    marker = raw[:n]
    original_lsb = iter(map(int, raw[n:n + n_lsb]))
    original_msb = raw[n + n_lsb:]
    assert len(original_msb) == 2 * len(a_pixels())
    assert marker == "".join("1" if c in ("EZ", "EN1") else "0" for c in classes)
    restored = [row[:] for row in pre]
    for index, cls, newdiff in processed:
        pos = e_pairs()[index]
        avg = sum(get(pre, pos)) // 2
        olddiff = (newdiff // 2 if cls in ("EZ", "EN1")
                   else 2 * (newdiff // 2) + next(original_lsb))
        put(restored, pos, pixels(avg, olddiff))
    assert "".join(f"{restored[r][c]:08b}"[:2] for r, c in a_pixels()) == original_msb
    return restored, bits, raw


def tsv(matrix):
    return "".join("\t".join(map(str, row)) + "\n" for row in matrix)


def execute(seed=DEFAULT_SEED, out=None):
    original = generate(seed)
    classes = [category(*get(original, pos)) for pos in e_pairs()]
    assert classes[:6] == ["EZ", "EN1", "EN2", "CN", "NC", "EZ"]
    raw, marker, lsb, msb = make_aux(original, classes)
    runs = rle(raw)
    payload = pack(runs)
    pre, rows = embed(original, classes, payload)
    restored, extracted, decoded = recover(pre, classes, len(payload), len(runs))
    assert extracted == payload
    assert decoded == raw
    assert restored == original
    changed = sum(original[r][c] != pre[r][c] for r in range(SIDE) for c in range(SIDE))
    report = {
        "seed": seed,
        "stage": "reversible E-region embedding demo (DNA encryption NOT implemented)",
        "original_min": min(map(min, original)), "original_max": max(map(max, original)),
        "original_unique_values": len(set(sum(original, []))),
        "modified_pixel_count": changed,
        "categories": dict(collections.Counter(classes)),
        "raw_bits": len(raw),
        "rle_runs": [list(x) for x in runs],
        "embedded_bits": len(payload),
        "capacity_bits": sum(c != "NC" for c in classes),
        "binary_payload": payload,
        "first_6_pairs": rows[:6],
        "recovery_equal_original": True,
        "external_info": ["RLE 1+7 bit format and group count",
                          "payload length",
                          "original 98-class map: required by THIS teaching decoder"],
        "paper_exact": False
    }
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for name, matrix in [("original.tsv", original), ("pre_encrypted.tsv", pre),
                             ("restored.tsv", restored)]:
            (out / name).write_text(tsv(matrix), encoding="utf-8")
        (out / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "pairs.tsv").write_text(
            "group\trow\tcolumn\toriginal\tclass\tbit\tembedded\n" +
            "".join(f"{d['group']}\t{d['row']}\t{d['column']}\t"
                    f"{','.join(map(str, d['original']))}\t{d['category']}\t"
                    f"{'' if d['bit'] is None else d['bit']}\t"
                    f"{','.join(map(str, d['embedded']))}\n" for d in rows),
            encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()
    print(json.dumps(execute(args.seed, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
