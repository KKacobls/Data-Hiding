from __future__ import annotations

import base64
import bisect
import io
import re
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

# =====================================================
# 位元工具
# =====================================================
def int_to_bits(n, length):
    return [int(b) for b in format(n, f"0{length}b")]


def bits_to_int(bits):
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val


def _ensure_even_matrix(matrix):
    matrix = [row[:] for row in matrix]
    rows, cols = len(matrix), len(matrix[0])
    if rows % 2 == 1:
        matrix = matrix[:-1]
        rows -= 1
    if cols % 2 == 1:
        matrix = [row[:-1] for row in matrix]
        cols -= 1
    return matrix, rows, cols  # 偶數裁剪後的矩陣、列數、欄數


def _flatten_pairs(matrix):
    flat = []
    for row in matrix:
        flat.extend(row)
    pairs = [(int(flat[i]), int(flat[i + 1])) for i in range(0, len(flat), 2)]
    return flat, pairs  # flat: 一維像素 | pairs: 兩兩配對 (A,B)


def is_strict_safe(a: int, b: int) -> bool:
    """嚴格安全（整數 DE 解析邊界，O(1)）：藏 bit 0 與 1 皆不溢位。

    暴力驗證 256×256 全組合與下式一致：
        A' = L + (2H + D + 1) // 2
        B' = L - (2H + D - 1) // 2
    得：
        3A - 509 <= B <= 3A
        3B - 509 <= A <= 3B
    """
    a, b = int(a), int(b)
    return (3 * a - 509 <= b <= 3 * a) and (3 * b - 509 <= a <= 3 * b)


def _pair_embed_bit_in_range(a: int, b: int, bit: int) -> bool:
    """整數 DE 試算：在封面 (a,b) 藏入 bit 後像素是否仍在 [0,255]。"""
    L = (a + b) // 2
    H = a - b
    H_new = H * 2 + bit
    na = L + ((H_new + 1) // 2)
    nb = L - (H_new // 2)
    return 0 <= na <= 255 and 0 <= nb <= 255


def _pair_expandable(a, b) -> bool:
    a, b = int(a), int(b)
    return _pair_embed_bit_in_range(a, b, 0)


def _pair_expandable_strict(a, b) -> bool:
    """嚴格安全：解析邊界 O(1)（略保守於 _pair_embed_bit 雙 bit 試算 ~0.19%）。"""
    return is_strict_safe(a, b)


def _pair_expandable_strict_bitwise(a: int, b: int) -> bool:
    """嚴格安全：逐 bit 整數試算（與實際嵌入公式一致，較慢）。"""
    a, b = int(a), int(b)
    return _pair_embed_bit_in_range(a, b, 0) and _pair_embed_bit_in_range(a, b, 1)


def _de_compatibility_grade(ratio: float) -> str:
    if ratio >= 0.95:
        return "極佳 ⭐⭐⭐⭐⭐"
    if ratio >= 0.70:
        return "良好 ⭐⭐⭐⭐"
    if ratio >= 0.30:
        return "勉強可用 ⭐⭐"
    return "不建議使用 ❌"


def _ms_since(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def build_loc_map(pairs):
    """回傳 loc_map（1=可藏 bit 0）與溢位組別（1-based）。承載區嵌入另以 strict 槽位跳過弱配對。"""
    loc_map = []
    overflow_pairs = []
    for idx, (a, b) in enumerate(pairs):
        if _pair_expandable(a, b):
            loc_map.append(1)
        else:
            loc_map.append(0)
            overflow_pairs.append(idx + 1)
    return loc_map, overflow_pairs


def loc_map_from_overflow_indices(pair_total: int, overflow_indices: list[int]) -> list[int]:
    overflow_set = set(overflow_indices)
    return [0 if (i + 1) in overflow_set else 1 for i in range(pair_total)]


def secret_to_bits(secret_text: str) -> list[int]:
    bits = []
    for byte in secret_text.encode("utf-8"):
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_bytes(secret_bits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(secret_bits), 8):
        if i + 8 > len(secret_bits):
            break
        out.append(bits_to_int(secret_bits[i : i + 8]))
    return bytes(out)


# =====================================================
# 封包 V4：[16 magic] + [8 IB] + [32 n] + [n×IB 溢位區 I_ovf]
#         + [8 L] + [L×8 檔名] + [16 W] + [W×IB 弱跳區 I_weak]
#         + [32 M] + [DATA] + [32 CRC]
# IB = INDEX_BITS = max(1, bit_length(N))，N=像素對總數；I_ovf 與 I_weak 共用 IB，筆數分別為 n、W
# Magic 同時表示「已隱寫」與演算法；CRC32 僅對 DATA bytes（zlib.crc32）
# =====================================================

STEGO_MAGIC: dict[int, dict] = {
    0xDE01: {
        "id": "DE",
        "label": "差值擴展 DE",
        "version": "V4",
    },
    # 未來可擴充，例如：
    # 0xAB02: {"id": "OTHER", "label": "其他演算法", "version": "V1"},
}

PACKET_MAGIC = 0xDE01  # 目前 DE 預設 magic（= STEGO_MAGIC 其中一項）


def get_stego_algorithm(magic: int) -> dict | None:
    """依 16-bit magic 查詢演算法資訊；None 表示非已知隱寫封包。"""
    return STEGO_MAGIC.get(magic)


def is_stego_magic(magic: int) -> bool:
    return magic in STEGO_MAGIC


def compute_index_bits(pair_total: int) -> int:
    """最大 1-based 溢位索引為 pair_total，所需 bit 數 = bit_length(N)。"""
    if pair_total <= 0:
        return 1
    return max(1, pair_total.bit_length())


def normalize_extension(ext: str) -> str:
    """從副檔名（含或不含點）取得小寫 ASCII 副檔名。"""
    ext = (ext or "txt").strip().lstrip(".").lower()
    ext = re.sub(r"[^a-z0-9_+-]", "", ext)
    return ext[:255] if ext else "txt"


def normalize_filename(filename: str) -> str:
    """完整檔案名稱（UTF-8，不含路徑），最長 255 bytes。"""
    name = (filename or "secret.txt").strip()
    name = name.replace("\\", "/").split("/")[-1]
    if not name:
        name = "secret.txt"
    raw = name.encode("utf-8")
    if len(raw) > 255:
        raw = raw[:255]
        while raw and (raw[-1] & 0xC0) == 0x80:
            raw = raw[:-1]
    return raw.decode("utf-8", errors="replace") or "secret.txt"


def filename_field_bits(filename: str) -> tuple[int, list[int]]:
    """回傳 (L, 檔案名稱 bit 流)。"""
    name = normalize_filename(filename)
    raw = name.encode("utf-8")
    return len(raw), int_to_bits(int.from_bytes(raw, "big"), len(raw) * 8)


def filename_from_bits(name_bits: list[int]) -> str:
    if not name_bits:
        return "secret.txt"
    raw = bits_to_bytes(name_bits)
    return normalize_filename(raw.decode("utf-8", errors="replace"))


def extension_from_filename(filename: str) -> str:
    return normalize_extension(Path(normalize_filename(filename)).suffix)


TEXT_VIEWABLE_EXTENSIONS = frozenset({
    "txt", "md", "json", "py", "js", "ts", "html", "htm", "css", "xml", "csv",
    "log", "ini", "cfg", "yml", "yaml", "sh", "bat", "ps1", "c", "cpp", "h",
    "java", "go", "rs", "sql", "rb", "php", "vue", "tsx", "jsx", "toml",
})

IMAGE_VIEWABLE_EXTENSIONS = frozenset({
    "png", "jpg", "jpeg", "gif", "webp", "bmp",
})


def classify_extracted_payload(secret_bytes: bytes, filename: str) -> dict:
    """解碼結果分類：是否可預覽文字／圖片、建議檔名與 MIME。"""
    import mimetypes

    filename = normalize_filename(filename)
    ext = extension_from_filename(filename)
    mime, _ = mimetypes.guess_type(filename)
    mime = mime or "application/octet-stream"
    viewable_text = False
    text_preview = ""
    viewable_image = ext in IMAGE_VIEWABLE_EXTENSIONS

    if ext in TEXT_VIEWABLE_EXTENSIONS and b"\x00" not in secret_bytes[:8192]:
        try:
            text_preview = secret_bytes.decode("utf-8")
            viewable_text = True
        except UnicodeDecodeError:
            viewable_text = False

    return {
        "extension": ext,
        "filename": filename,
        "mime": mime,
        "size": len(secret_bytes),
        "viewable_text": viewable_text,
        "viewable_image": viewable_image,
        "text": text_preview if viewable_text else "",
    }


def build_bitstream(
    overflow_indices: list[int],
    secret_bytes: bytes,
    pair_total: int,
    filename: str = "secret.txt",
    weak_skip_indices: list[int] | None = None,
) -> list[int]:
    secret_bits = []
    for byte in secret_bytes:
        for i in range(7, -1, -1):
            secret_bits.append((byte >> i) & 1)
    crc = zlib.crc32(secret_bytes) & 0xFFFFFFFF
    index_bits = compute_index_bits(pair_total)
    weak_skip_indices = sorted(weak_skip_indices or [])

    stream = int_to_bits(PACKET_MAGIC, 16)
    stream.extend(int_to_bits(index_bits, 8))
    stream.extend(int_to_bits(len(overflow_indices), 32))
    for idx in overflow_indices:
        if idx >= (1 << index_bits):
            raise ValueError(
                f"溢位索引 {idx} 超出 INDEX_BITS={index_bits} 可表示範圍（N={pair_total}）"
            )
        stream.extend(int_to_bits(idx, index_bits))
    l_len, name_bits = filename_field_bits(filename)
    stream.extend(int_to_bits(l_len, 8))
    stream.extend(name_bits)
    stream.extend(int_to_bits(len(weak_skip_indices), 16))
    for idx in weak_skip_indices:
        if idx >= (1 << index_bits):
            raise ValueError(
                f"弱配對跳過索引 {idx} 超出 INDEX_BITS={index_bits} 可表示範圍"
            )
        stream.extend(int_to_bits(idx, index_bits))
    stream.extend(int_to_bits(len(secret_bits), 32))
    stream.extend(secret_bits)
    stream.extend(int_to_bits(crc, 32))
    return stream


def bitstream_fixed_overhead(
    overflow_count: int,
    pair_total: int,
    filename: str = "secret.txt",
    weak_skip_count: int = 0,
) -> int:
    """不含 DATA 的固定開銷（含 CRC 欄位）。"""
    return packet_header_bits(
        overflow_count, pair_total, filename, weak_skip_count=weak_skip_count
    ) + 32


def packet_header_bits(
    overflow_count: int,
    pair_total: int,
    filename: str = "secret.txt",
    *,
    weak_skip_count: int = 0,
) -> int:
    """封包標頭（Magic + INDEX_BITS + n + 索引 + L + 檔名 + W + M），不含 DATA 與 CRC。"""
    index_bits = compute_index_bits(pair_total)
    l_len, _ = filename_field_bits(filename)
    return (
        16 + 8 + 32 + index_bits * overflow_count + 8 + l_len * 8 + 16
        + index_bits * weak_skip_count + 32
    )


def packet_indices_end(index_bits: int, overflow_count: int) -> int:
    return 56 + index_bits * overflow_count


def packet_field_label(
    bit_pos: int,
    header_reserved: int,
    indices_end: int,
    name_end: int,
) -> str:
    """嵌入流 bit 落在封包哪一段（用於 UI）。"""
    if bit_pos < 16:
        return "Magic"
    if bit_pos < 24:
        return "INDEX_BITS"
    if bit_pos < 56:
        return "n"
    if bit_pos < indices_end:
        return "索引清單"
    if bit_pos < indices_end + 8:
        return "L"
    if bit_pos < name_end:
        return "檔案名稱"
    if bit_pos < header_reserved:
        return "M"
    return "承載區"


def build_overflow_embed_map(
    overflow_pairs: list[int],
    index_bits: int,
    header_reserved: int,
    pair_total: int | None = None,
    pairs=None,
    filename: str = "secret.txt",
) -> list[dict]:
    """每個溢位點：封包內索引欄位 bit 區間 + 圖像嵌入流 bit 位置 + 所屬區段。"""
    if not overflow_pairs:
        return []
    overflow_set = set(overflow_pairs)
    if pair_total is None:
        pair_total = max(overflow_pairs)
    indices_end = packet_indices_end(index_bits, len(overflow_pairs))
    l_len, _ = filename_field_bits(filename)
    name_end = indices_end + 8 + l_len * 8

    payload_embed_at: dict[int, int] = {}
    payload_safe = 0
    if pairs is not None:
        loc_map = loc_map_from_overflow_indices(pair_total, overflow_pairs)
        payload_slot_set = set(
            payload_embed_slot_indices_strict(pairs, loc_map, header_reserved)
        )
    else:
        payload_slot_set = None

    for pair_idx in range(header_reserved, pair_total):
        pair_num = pair_idx + 1
        if pair_num in overflow_set:
            payload_embed_at[pair_num] = header_reserved + payload_safe
            continue
        if payload_slot_set is not None and pair_idx not in payload_slot_set:
            continue
        payload_safe += 1

    items = []
    for i, pair_num in enumerate(overflow_pairs):
        pair_idx = pair_num - 1
        idx_start = 56 + i * index_bits
        if pair_idx < header_reserved:
            embed_bit = pair_idx
            zone = "header"
        else:
            embed_bit = payload_embed_at[pair_num]
            zone = "payload"
        items.append({
            "pair": pair_num,
            "embed_bit": embed_bit,
            "zone": zone,
            "field": packet_field_label(embed_bit, header_reserved, indices_end, name_end),
            "index_entry": i + 1,
            "index_bit_start": idx_start,
            "index_bit_end": idx_start + index_bits - 1,
        })
    return items


def count_continuous_safe_pairs(pairs) -> int:
    """從第 0 組起，連續有多少組可通過藏 1 嚴格試算。"""
    count = 0
    for a, b in pairs:
        if _pair_expandable_strict(a, b):
            count += 1
        else:
            break
    return count


def validate_header_region(
    pairs,
    overflow_count: int,
    *,
    filename: str = "secret.txt",
    raise_on_fail: bool = True,
) -> tuple[int, int]:
    """檢查連續安全區是否容納得下封包標頭。回傳 (continuous_safe, header_bits)。"""
    continuous_safe = count_continuous_safe_pairs(pairs)
    header_bits = packet_header_bits(overflow_count, len(pairs), filename)
    if continuous_safe < header_bits:
        msg = (
            f"從第 0 組開始，連續安全配對只有 {continuous_safe} 組，"
            f"但封包標頭需要 {header_bits} bits。"
            f"這張圖的開頭區域對比太強，不適合 DE，請換一張圖。"
        )
        if raise_on_fail:
            raise ValueError(msg)
    return continuous_safe, header_bits


def extract_bits_sequential(pairs, count: int) -> list[int]:
    """從第 0 組起依序取 bit（標頭區，不跳過溢位）。"""
    return extract_bits_from_pairs(pairs, 0, count)


def extract_bits_from_pairs(pairs, start: int, count: int) -> list[int]:
    """從第 start 組起連續取 count 個 bit（標頭區）。"""
    bits = []
    for i in range(start, start + count):
        a, b = pairs[i]
        bits.append((int(a) - int(b)) & 1)
    return bits


def count_payload_embed_slots(pairs, loc_map: list[int], header_reserved: int) -> int:
    """承載區可寫入 bit 的配對數：跳過溢位索引與僅能藏 0 的弱配對。"""
    return len(payload_embed_slot_indices_strict(pairs, loc_map, header_reserved))


def compute_capacity_profile(
    pair_total: int,
    overflow_pairs: list[int],
    loc_map: list[int],
    pairs,
    continuous_safe: int,
    header_reserved: int,
) -> dict:
    """三道篩選：門檻 → 寬鬆承載 → 嚴格承載，回傳可保證的 DATA 容量。"""
    safe_pairs = sum(loc_map)
    gate_continuous = continuous_safe >= header_reserved
    overflow_in_header = any(p <= header_reserved for p in overflow_pairs)
    gate_header_clear = not overflow_in_header
    capacity_ok = gate_continuous and gate_header_clear

    payload_loose_slots = sum(loc_map[header_reserved:])
    payload_strict_slots = count_payload_embed_slots(pairs, loc_map, header_reserved)

    cap_loose_data_bits = max(0, payload_loose_slots - 32)
    cap_strict_data_bits = max(0, payload_strict_slots - 32)
    cap_formula_data_bits = max(0, safe_pairs - header_reserved - 32)

    gate_reasons: list[str] = []
    if not gate_continuous:
        gate_reasons.append("continuous_safe_lt_header")
    if overflow_in_header:
        gate_reasons.append("overflow_in_header_zone")

    if not capacity_ok:
        max_data_bits = 0
        limiting_factor = "gate"
    else:
        max_data_bits = min(cap_loose_data_bits, cap_strict_data_bits)
        if max_data_bits <= 0:
            limiting_factor = "none"
        elif cap_strict_data_bits <= cap_loose_data_bits:
            limiting_factor = "strict_payload"
        else:
            limiting_factor = "loose_payload"

    return {
        "capacity_ok": capacity_ok,
        "gate_continuous_safe": gate_continuous,
        "gate_header_no_overflow": gate_header_clear,
        "overflow_in_header_zone": overflow_in_header,
        "capacity_gate_reasons": gate_reasons,
        "cap_T": pair_total,
        "cap_O": len(overflow_pairs),
        "cap_safe_pairs": safe_pairs,
        "cap_C": continuous_safe,
        "cap_H": header_reserved,
        "cap_payload_loose_slots": payload_loose_slots,
        "cap_payload_strict_slots": payload_strict_slots,
        "cap_loose_data_bits": cap_loose_data_bits,
        "cap_strict_data_bits": cap_strict_data_bits,
        "cap_formula_data_bits": cap_formula_data_bits,
        "max_data_bits": max_data_bits,
        "max_text_bytes": max_data_bits // 8,
        "limiting_factor": limiting_factor,
    }


def resolve_strict_payload_capacity(
    pairs,
    loc_map: list[int],
    overflow_pairs: list[int],
    pair_total: int,
    filename: str,
    *,
    payload_bytes: int = 0,
    strict_walk_cache: StrictWalkCache | None = None,
) -> dict:
    """
    公式計算 strict 承載：標頭 H 與 W 表随 payload 長度迭代至穩定（不用二分搜尋）。
    payload_bytes=0 → 可保證 DATA 上界；>0 → 該長度下的槽位预算與是否 fit。
    """
    n = len(overflow_pairs)
    name_norm = normalize_filename(filename)
    header0 = packet_header_bits(n, pair_total, name_norm, weak_skip_count=0)
    strict0 = count_payload_embed_slots(pairs, loc_map, header0)
    cache = strict_walk_cache or build_strict_walk_cache(pairs, loc_map)

    if payload_bytes > 0:
        payload_needed = payload_bytes * 8 + 32
        header_reserved, weak_skips = resolve_packet_header_and_weak_skips(
            pairs,
            loc_map,
            n,
            pair_total,
            name_norm,
            payload_needed,
            strict_walk_cache=cache,
            need_weak_indices=False,
        )
        strict_slots = count_payload_embed_slots(pairs, loc_map, header_reserved)
        data_bits = max(0, strict_slots - 32)
        return {
            "header_reserved_bits": header_reserved,
            "weak_skip_count": cache.weak_skip_count(header_reserved, payload_needed),
            "payload_strict_slots": strict_slots,
            "payload_needed_bits": payload_needed,
            "max_data_bits": data_bits,
            "max_text_bytes": data_bits // 8,
            "formula_fits": payload_needed <= strict_slots,
        }

    b_values: list[int] = []
    b = max(0, (strict0 - 32) // 8)
    header_reserved = header0
    weak_count = 0
    strict_slots = strict0
    hw_iterations = 0
    for _ in range(8):
        hw_iterations += 1
        b_values.append(b)
        payload_needed = b * 8 + 32
        header_reserved, _ = resolve_packet_header_and_weak_skips(
            pairs,
            loc_map,
            n,
            pair_total,
            name_norm,
            payload_needed,
            strict_walk_cache=cache,
            need_weak_indices=False,
        )
        weak_count = cache.weak_skip_count(header_reserved, payload_needed)
        strict_slots = count_payload_embed_slots(pairs, loc_map, header_reserved)
        b_new = max(0, (strict_slots - 32) // 8)
        if b_new == b:
            break
        b = b_new
    b_final = min(b_values)
    if b_final != b:
        payload_needed = b_final * 8 + 32
        header_reserved, _ = resolve_packet_header_and_weak_skips(
            pairs,
            loc_map,
            n,
            pair_total,
            name_norm,
            payload_needed,
            strict_walk_cache=cache,
            need_weak_indices=False,
        )
        weak_count = cache.weak_skip_count(header_reserved, payload_needed)
        strict_slots = count_payload_embed_slots(pairs, loc_map, header_reserved)
    data_bits = b_final * 8
    return {
        "max_text_bytes_guaranteed": b_final,
        "max_data_bits_guaranteed": data_bits,
        "guaranteed_header_bits": header_reserved,
        "guaranteed_weak_skip_count": weak_count,
        "guaranteed_payload_strict_slots": strict_slots,
        "header_at_guaranteed": header0,
        "strict_slots_at_h0": strict0,
        "guaranteed_hw_iterations": hw_iterations,
        "loc_map_bits": pair_total,
    }


def _reverse_pair_pixels(a: int, b: int) -> tuple[int, int]:
    """DE 逆運算：由隱寫配對還原嵌入前一層的封面像素。"""
    L = (int(a) + int(b)) // 2
    H = int(a) - int(b)
    bit = H & 1
    H_c = (H - bit) // 2
    return L + ((H_c + 1) // 2), L - (H_c // 2)


def _stego_from_cover_with_bit(cover_a: int, cover_b: int, bit: int) -> tuple[int, int]:
    """由封面像素與 bit 模擬 DE 嵌入後的隱寫像素。"""
    a, b = int(cover_a), int(cover_b)
    L = (a + b) // 2
    H = a - b
    H_new = H * 2 + bit
    return L + ((H_new + 1) // 2), L - (H_new // 2)


def _cover_pixels_for_payload_slot_decode(stego_a: int, stego_b: int) -> tuple[int, int]:
    """解碼端還原封面像素，用於重建承載槽清單。"""
    sa, sb = int(stego_a), int(stego_b)
    if _pair_expandable_strict(sa, sb):
        return sa, sb
    if not _pair_expandable(sa, sb):
        return _reverse_pair_pixels(sa, sb)
    ra, rb = _reverse_pair_pixels(sa, sb)
    if _pair_expandable_strict(ra, rb):
        for bit in (0, 1):
            if _stego_from_cover_with_bit(ra, rb, bit) == (sa, sb):
                return ra, rb
    return sa, sb


def _cover_for_slot_check(stego_a: int, stego_b: int) -> tuple[int, int]:
    """還原封面像素（還原圖等用途）。"""
    return _cover_pixels_for_payload_slot_decode(stego_a, stego_b)


def payload_embed_slot_indices(
    pairs,
    loc_map: list[int],
    header_reserved: int,
) -> list[int]:
    """承載區嵌入槽（0-based）：跳 loc_map=0 的溢位（含弱配對）。"""
    return [
        idx
        for idx in range(header_reserved, len(pairs))
        if loc_map[idx] == 1
    ]


def payload_embed_slot_indices_strict(
    pairs,
    loc_map: list[int],
    header_reserved: int,
) -> list[int]:
    """嚴格承載槽：跳溢位 + 弱配對（僅 0/1 皆可藏）。"""
    return [
        idx
        for idx in range(header_reserved, len(pairs))
        if loc_map[idx] == 1 and _pair_expandable_strict_bitwise(*pairs[idx])
    ]


def payload_weak_skip_indices(
    pairs,
    loc_map: list[int],
    header_reserved: int,
    payload_needed: int,
) -> list[int]:
    """strict 嵌入 walk 前綴中跳過的弱配對（1-based），解碼時排除 ghost maybe。"""
    weak_skips: list[int] = []
    strict_count = 0
    for idx in range(header_reserved, len(pairs)):
        if loc_map[idx] != 1:
            continue
        if _pair_expandable_strict_bitwise(*pairs[idx]):
            strict_count += 1
            if strict_count >= payload_needed:
                break
        else:
            weak_skips.append(idx + 1)
    return weak_skips


class StrictWalkCache:
    """承載區 strict walk 预計算：O(n) 建表，O(log n) 查 weak 跳過數。"""

    __slots__ = ("_pair_idxs", "_prefix_strict", "_prefix_weak")

    def __init__(self, pairs, loc_map: list[int]):
        pair_idxs: list[int] = []
        prefix_strict: list[int] = []
        prefix_weak: list[int] = []
        strict_count = 0
        weak_count = 0
        for idx, loc in enumerate(loc_map):
            if loc != 1:
                continue
            pair_idxs.append(idx)
            if _pair_expandable_strict_bitwise(*pairs[idx]):
                strict_count += 1
            else:
                weak_count += 1
            prefix_strict.append(strict_count)
            prefix_weak.append(weak_count)
        self._pair_idxs = pair_idxs
        self._prefix_strict = prefix_strict
        self._prefix_weak = prefix_weak

    def weak_skip_count(self, header_reserved: int, payload_needed: int) -> int:
        if payload_needed <= 0 or not self._pair_idxs:
            return 0
        start = bisect.bisect_left(self._pair_idxs, header_reserved)
        if start >= len(self._pair_idxs):
            return 0
        base_strict = self._prefix_strict[start - 1] if start > 0 else 0
        base_weak = self._prefix_weak[start - 1] if start > 0 else 0
        target = base_strict + payload_needed
        if self._prefix_strict[-1] < target:
            return self._prefix_weak[-1] - base_weak
        end = bisect.bisect_left(self._prefix_strict, target, lo=start)
        return self._prefix_weak[end] - base_weak


def build_strict_walk_cache(pairs, loc_map: list[int]) -> StrictWalkCache:
    return StrictWalkCache(pairs, loc_map)


def resolve_packet_header_and_weak_skips(
    pairs,
    loc_map: list[int],
    overflow_count: int,
    pair_total: int,
    filename: str,
    payload_needed: int,
    *,
    strict_walk_cache: StrictWalkCache | None = None,
    need_weak_indices: bool = True,
) -> tuple[int, list[int]]:
    """標頭長度與弱配對跳過表互相依賴時迭代至穩定。"""
    header_reserved = packet_header_bits(overflow_count, pair_total, filename, weak_skip_count=0)
    weak_count = 0
    for _ in range(8):
        if strict_walk_cache is not None:
            weak_count = strict_walk_cache.weak_skip_count(header_reserved, payload_needed)
            new_header = packet_header_bits(
                overflow_count, pair_total, filename, weak_skip_count=weak_count
            )
        else:
            weak_skips = payload_weak_skip_indices(
                pairs, loc_map, header_reserved, payload_needed
            )
            weak_count = len(weak_skips)
            new_header = packet_header_bits(
                overflow_count, pair_total, filename, weak_skip_count=weak_count
            )
        if new_header == header_reserved:
            break
        header_reserved = new_header
    if need_weak_indices:
        weak_skips = payload_weak_skip_indices(
            pairs, loc_map, header_reserved, payload_needed
        )
    else:
        weak_skips = []
    return header_reserved, weak_skips


def _classify_payload_pair_decode(sa: int, sb: int) -> str:
    """解碼端配對分類：yes=確定承載槽，maybe=需選擇，no=非承載（與嵌入 strict 掃描一致）。"""
    return _classify_payload_pair_walk(sa, sb)


def _classify_payload_pair_walk(sa: int, sb: int) -> str:
    """依 stego 像素分類（嵌入端 strict 掃描的解碼鏡像）。"""
    sa, sb = int(sa), int(sb)
    if _pair_expandable_strict_bitwise(sa, sb):
        return "yes"
    if not _pair_expandable(sa, sb):
        ra, rb = _reverse_pair_pixels(sa, sb)
        if _pair_expandable_strict_bitwise(ra, rb):
            return "yes"
        return "no"
    ra, rb = _reverse_pair_pixels(sa, sb)
    if not _pair_expandable_strict_bitwise(ra, rb):
        return "no"
    if _stego_from_cover_with_bit(ra, rb, 1) == (sa, sb):
        return "maybe"
    if _stego_from_cover_with_bit(ra, rb, 0) == (sa, sb):
        return "maybe"
    return "no"


def _is_ghost_maybe_pair(
    sa: int,
    sb: int,
    cover_a: int,
    cover_b: int,
) -> bool:
    """弱配對幽靈槽：封面未改、strict 嵌入會跳過，解碼 walk 卻像 maybe。"""
    if _classify_payload_pair_walk(sa, sb) != "maybe":
        return False
    sa, sb, ca, cb = int(sa), int(sb), int(cover_a), int(cover_b)
    if (sa, sb) != (ca, cb):
        return False
    recon = _cover_pixels_for_payload_slot_decode(sa, sb)
    rev = _reverse_pair_pixels(sa, sb)
    return recon == rev and recon != (ca, cb)


def _payload_decode_stream(
    stego_pairs,
    loc_map: list[int],
    header_reserved: int,
    payload_needed: int,
    cover_pairs=None,
    weak_skip_indices: list[int] | None = None,
) -> list[tuple[int, str]]:
    """承載區解碼 walk：yes 必收；maybe 為候選；跳過 ghost weak（有封面時）。"""
    weak_skip_set = set(weak_skip_indices or [])
    stream: list[tuple[int, str]] = []
    for idx in range(header_reserved, len(stego_pairs)):
        if loc_map[idx] != 1:
            continue
        if (idx + 1) in weak_skip_set:
            continue
        sa, sb = stego_pairs[idx]
        kind = _classify_payload_pair_walk(sa, sb)
        if kind == "no":
            continue
        if kind == "maybe" and cover_pairs is not None:
            ca, cb = cover_pairs[idx]
            if _is_ghost_maybe_pair(sa, sb, ca, cb):
                continue
        stream.append((idx, kind))
        if len(stream) >= payload_needed:
            break
    return stream


def _payload_bits_from_slots(stego_pairs, slots: list[int]) -> list[int]:
    return [(int(stego_pairs[idx][0]) - int(stego_pairs[idx][1])) & 1 for idx in slots]


def _payload_crc_ok(stego_pairs, slots: list[int], data_len: int) -> bool:
    payload_needed = data_len + 32
    if len(slots) < payload_needed:
        return False
    out = bytearray()
    byte = 0
    bits_in_byte = 0
    for i in range(payload_needed):
        idx = slots[i]
        bit = (int(stego_pairs[idx][0]) - int(stego_pairs[idx][1])) & 1
        byte = (byte << 1) | bit
        bits_in_byte += 1
        if bits_in_byte == 8:
            out.append(byte)
            byte = 0
            bits_in_byte = 0
    if bits_in_byte:
        out.append(byte << (8 - bits_in_byte))
    secret_bytes = bytes(out[: data_len // 8])
    crc_read = int.from_bytes(out[data_len // 8 : data_len // 8 + 4], "big")
    crc_calc = zlib.crc32(secret_bytes) & 0xFFFFFFFF
    return crc_calc == crc_read


def _slots_from_prefix_mask(
    prefix: list[tuple[int, str]],
    maybe_indices: list[int],
    mask: int,
    payload_needed: int,
) -> list[int]:
    slots: list[int] = []
    mi = 0
    for idx, kind in prefix:
        if kind == "include":
            slots.append(idx)
        elif kind == "maybe":
            if mask & (1 << mi):
                slots.append(idx)
            mi += 1
        if len(slots) >= payload_needed:
            break
    return slots[:payload_needed]


def _resolve_strict_payload_slots(
    stego_pairs,
    loc_map: list[int],
    header_reserved: int,
    data_len: int,
    cover_pairs=None,
    weak_skip_indices: list[int] | None = None,
) -> list[int]:
    """承載槽清單（與 strict 嵌入一致）。stego-only 走嵌入順序 walk + CRC 回溯。"""
    payload_needed = data_len + 32
    if cover_pairs is not None:
        slots = payload_embed_slot_indices_strict(cover_pairs, loc_map, header_reserved)
        if len(slots) < payload_needed:
            raise ValueError(
                f"承載區不足：需要 {payload_needed} bits，"
                f"strict 槽僅 {len(slots)} 組"
            )
        return slots[:payload_needed]

    return _resolve_strict_payload_slots_stego(
        stego_pairs,
        loc_map,
        header_reserved,
        data_len,
        cover_pairs=cover_pairs,
        weak_skip_indices=weak_skip_indices,
    )


def _resolve_strict_payload_slots_stego(
    stego_pairs,
    loc_map: list[int],
    header_reserved: int,
    data_len: int,
    *,
    cover_pairs=None,
    weak_skip_indices: list[int] | None = None,
    max_combinations: int = 5_000_000,
) -> list[int]:
    """stego-only：嵌入順序 walk；yes 必收、maybe 用 CRC 組合搜尋。"""
    from itertools import combinations
    import math

    payload_needed = data_len + 32
    stream = _payload_decode_stream(
        stego_pairs,
        loc_map,
        header_reserved,
        payload_needed,
        cover_pairs=cover_pairs,
        weak_skip_indices=weak_skip_indices,
    )

    yes_indices = [idx for idx, kind in stream if kind == "yes"]
    maybe_indices = [idx for idx, kind in stream if kind == "maybe"]
    need_from_maybe = payload_needed - len(yes_indices)

    if need_from_maybe < 0:
        slots = yes_indices[:payload_needed]
        if _payload_crc_ok(stego_pairs, slots, data_len):
            return slots
        raise ValueError("承載區 CRC 校驗失敗（strict 解碼）")

    if need_from_maybe == 0:
        slots = yes_indices[:payload_needed]
        if _payload_crc_ok(stego_pairs, slots, data_len):
            return slots
        raise ValueError("承載區 CRC 校驗失敗（strict 解碼）")

    if len(maybe_indices) < need_from_maybe:
        raise ValueError(
            f"承載區不足：需要 {payload_needed} bits，"
            f"yes={len(yes_indices)} maybe={len(maybe_indices)}"
        )

    c_n = math.comb(len(maybe_indices), need_from_maybe)
    if c_n > max_combinations:
        raise ValueError(
            f"承載區 maybe 組合過多（C({len(maybe_indices)},{need_from_maybe})={c_n}），"
            f"請縮小 DATA"
        )

    for combo in combinations(maybe_indices, need_from_maybe):
        chosen = set(combo)
        slots: list[int] = []
        for idx, kind in stream:
            if kind == "yes" or idx in chosen:
                slots.append(idx)
            if len(slots) >= payload_needed:
                break
        if len(slots) == payload_needed and _payload_crc_ok(stego_pairs, slots, data_len):
            return slots

    if weak_skip_indices is None:
        prefix = stream[: min(len(stream), payload_needed + 64)]
        maybe_candidates = [idx for idx, kind in prefix if kind == "maybe"]

        def build_slots(chosen: set[int]) -> list[int]:
            slots: list[int] = []
            for idx, kind in prefix:
                if kind == "yes" or idx in chosen:
                    slots.append(idx)
                if len(slots) >= payload_needed:
                    break
            return slots[:payload_needed]

        prefix_yes = sum(1 for _, kind in prefix if kind == "yes")
        min_k = max(0, payload_needed - prefix_yes)
        for k in range(min_k, len(maybe_candidates) + 1):
            if math.comb(len(maybe_candidates), k) > max_combinations:
                continue
            for combo in combinations(maybe_candidates, k):
                slots = build_slots(set(combo))
                if len(slots) == payload_needed and _payload_crc_ok(
                    stego_pairs, slots, data_len
                ):
                    return slots

    raise ValueError("承載區 CRC 校驗失敗（strict 解碼，maybe 組合未對齊）")


def payload_embed_slot_indices_strict_decode(
    stego_pairs,
    overflow_indices: list[int],
    header_reserved: int,
    data_len: int = 0,
    cover_pairs=None,
    weak_skip_indices: list[int] | None = None,
) -> list[int]:
    """解碼端 strict 承載槽（與嵌入端一致）。"""
    loc_map = loc_map_from_overflow_indices(len(stego_pairs), overflow_indices)
    if cover_pairs is not None or data_len > 0:
        return _resolve_strict_payload_slots(
            stego_pairs,
            loc_map,
            header_reserved,
            data_len,
            cover_pairs=cover_pairs,
            weak_skip_indices=weak_skip_indices,
        )
    return payload_embed_slot_indices(stego_pairs, loc_map, header_reserved)


def payload_embed_slot_indices_decode(
    stego_pairs,
    overflow_indices: list[int],
    header_reserved: int,
    data_len: int = 0,
    cover_pairs=None,
    weak_skip_indices: list[int] | None = None,
) -> list[int]:
    """解碼端承載槽（strict，與嵌入一致）。"""
    return payload_embed_slot_indices_strict_decode(
        stego_pairs,
        overflow_indices,
        header_reserved,
        data_len=data_len,
        cover_pairs=cover_pairs,
        weak_skip_indices=weak_skip_indices,
    )


def extract_payload_bits(
    pairs,
    overflow_indices: list[int],
    header_pair_count: int,
    data_len: int = 0,
    cover_pairs=None,
    weak_skip_indices: list[int] | None = None,
) -> list[int]:
    """承載區：strict 槽位讀 bit（與嵌入端 payload_embed_slot_indices_strict 一致）。"""
    loc_map = loc_map_from_overflow_indices(len(pairs), overflow_indices)
    if data_len <= 0:
        slots = payload_embed_slot_indices_decode(
            pairs, overflow_indices, header_pair_count, cover_pairs=cover_pairs
        )
        return _payload_bits_from_slots(pairs, slots)
    slots = _resolve_strict_payload_slots(
        pairs,
        loc_map,
        header_pair_count,
        data_len,
        cover_pairs=cover_pairs,
        weak_skip_indices=weak_skip_indices,
    )
    return _payload_bits_from_slots(pairs, slots)


def _pair_post_embed_pixels(a: int, b: int, bit: int) -> tuple[int, int]:
    L = (a + b) // 2
    H = a - b
    H_new = H * 2 + bit
    na = L + ((H_new + 1) // 2)
    nb = L - (H_new // 2)
    return na, nb


def _debug_embed_fail(
    a: int,
    b: int,
    bit: int,
    pair_idx: int,
    *,
    zone: str = "",
    embed_bit: int | None = None,
) -> str:
    """失敗當下數值對照（is_strict_safe vs 實際 //），供 log / API 顯示。"""
    a, b = int(a), int(b)
    L = (a + b) // 2
    H = a - b
    na, nb = _pair_post_embed_pixels(a, b, bit)
    na_lh = L + (2 * H + bit + 1) // 2
    nb_lh = L - (2 * H + bit - 1) // 2
    H_new = H * 2 + bit
    na_de = L + ((H_new + 1) // 2)
    nb_de = L - (H_new // 2)
    where = zone or f"第 {pair_idx + 1} 組"
    if embed_bit is not None:
        where += f"，嵌入流 bit {embed_bit}"
    return (
        f"{where}\n"
        f"  封面 A={a} B={b}  H={H}  L={L}  要藏 bit={bit}\n"
        f"  嵌入後 A'={na} B'={nb}\n"
        f"  公式 L/H：A'={na_lh} B'={nb_lh}\n"
        f"  公式 H_new：A'={na_de} B'={nb_de}\n"
        f"  is_strict_safe={is_strict_safe(a, b)}"
        f"  embed(D=0)={_pair_embed_bit_in_range(a, b, 0)}"
        f"  embed(D=1)={_pair_embed_bit_in_range(a, b, 1)}"
        f"  loc_expandable(D=0)={_pair_expandable(a, b)}\n"
        f"  邊界 B∈[{3 * a - 509}, {3 * a}]"
        f"  A∈[{3 * b - 509}, {3 * b}]"
        f"  實際 B={b} A={a}"
    )


def _log_embed_fail(detail: str) -> None:
    import logging

    logging.getLogger(__name__).warning("DE 嵌入試算失敗\n%s", detail)


def _embed_fail_message(
    a: int,
    b: int,
    bit: int,
    pair_idx: int,
    *,
    summary: str,
    zone: str = "",
    embed_bit: int | None = None,
) -> str:
    detail = _debug_embed_fail(
        a, b, bit, pair_idx, zone=zone, embed_bit=embed_bit
    )
    _log_embed_fail(detail)
    return f"{summary}\n\n--- 診斷 ---\n{detail}"


def verify_embed_bitstream(
    pairs,
    loc_map: list[int],
    bitstream: list[int],
    header_reserved: int,
) -> tuple[bool, str | None, int | None]:
    """試算整段 bitstream 能否嵌入（不修改像素）。回傳 (ok, 錯誤訊息, 失敗組別 1-based)。"""
    payload_slots = payload_embed_slot_indices_strict(pairs, loc_map, header_reserved)
    payload_needed = len(bitstream) - header_reserved

    if len(payload_slots) < payload_needed:
        return (
            False,
            (
                f"承載區槽位不足：需要 {payload_needed} bits，"
                f"僅 {len(payload_slots)} 組可寫（跳溢位後）。"
                f"請縮小 DATA 或換更大、對比更低的封面。"
            ),
            None,
        )

    for pair_idx in range(header_reserved):
        bit = bitstream[pair_idx]
        a, b = pairs[pair_idx]
        if not _pair_embed_bit_in_range(a, b, bit):
            na, nb = _pair_post_embed_pixels(a, b, bit)
            return (
                False,
                _embed_fail_message(
                    a, b, bit, pair_idx,
                    summary=(
                        f"封包標頭無法寫入：第 {pair_idx + 1} 組藏 bit {bit} 後"
                        f"像素為 ({na}, {nb})，超出 [0,255]。"
                        f"封面開頭對比過強；請換圖或縮短檔名以縮小標頭。"
                    ),
                    zone="封包標頭",
                ),
                pair_idx + 1,
            )

    for i in range(payload_needed):
        pair_idx = payload_slots[i]
        bit = bitstream[header_reserved + i]
        a, b = pairs[pair_idx]
        if not _pair_embed_bit_in_range(a, b, bit):
            na, nb = _pair_post_embed_pixels(a, b, bit)
            embed_bit = header_reserved + i
            return (
                False,
                _embed_fail_message(
                    a, b, bit, pair_idx,
                    summary=(
                        f"承載區嵌入失敗：第 {pair_idx + 1} 組藏 bit {bit} 後"
                        f"像素為 ({na}, {nb})，超出 [0,255]"
                        f"（嵌入流 bit {embed_bit}）。"
                        f"弱配對或 DATA 過大；請縮小 DATA 或更換封面。"
                    ),
                    zone="承載區",
                    embed_bit=embed_bit,
                ),
                pair_idx + 1,
            )

    return True, None, None


def _embed_bit_into_pair(
    flat: list,
    pair_idx: int,
    bit: int,
    *,
    raise_on_overflow: bool = True,
) -> bool:
    i = pair_idx * 2
    a, b = int(flat[i]), int(flat[i + 1])
    if not _pair_embed_bit_in_range(a, b, bit):
        na, nb = _pair_post_embed_pixels(a, b, bit)
        if not _pair_expandable(a, b):
            summary = (
                f"第 {pair_idx + 1} 組在 loc_map 標為安全，"
                f"但藏 bit {bit} 後像素 ({na}, {nb}) 超出 [0,255]。"
            )
        else:
            summary = (
                f"第 {pair_idx + 1} 組弱配對：藏 bit {bit} 後像素 ({na}, {nb})"
                f"超出 [0,255]。請縮小 DATA 或更換封面。"
            )
        msg = _embed_fail_message(
            a, b, bit, pair_idx, summary=summary, zone="實際寫入"
        )
        if raise_on_overflow:
            raise ValueError(msg)
        return False
    L = (a + b) // 2
    H = a - b
    H_new = H * 2 + bit
    na = L + ((H_new + 1) // 2)
    nb = L - (H_new // 2)
    flat[i] = na
    flat[i + 1] = nb
    return True


def _restore_pair(flat: list, pair_idx: int) -> None:
    i = pair_idx * 2
    a, b = int(flat[i]), int(flat[i + 1])
    L = (a + b) // 2
    H = a - b
    H_orig = H // 2
    flat[i] = L + ((H_orig + 1) // 2)
    flat[i + 1] = L - (H_orig // 2)


def parse_bitstream(extracted_bits: list[int]) -> tuple[list[int], list[int], bytes, str]:
    pos = 0
    magic = bits_to_int(extracted_bits[pos : pos + 16])
    algo = get_stego_algorithm(magic)
    if algo is None:
        raise ValueError(
            f"找不到已知隱寫封包（讀到 {magic:#06x}，"
            f"已知 magic：{', '.join(f'{m:#06x}' for m in STEGO_MAGIC)}）。"
            "這通常表示：① 上傳的是封面原圖；② 圖片被 JPEG 等有损格式破壞；"
            "③ 使用了不支援的舊版封包格式。"
        )
    pos += 16

    if pos + 8 > len(extracted_bits):
        raise ValueError("封包不完整：缺少 INDEX_BITS")
    index_bits = bits_to_int(extracted_bits[pos : pos + 8])
    if index_bits < 1 or index_bits > 32:
        raise ValueError(f"INDEX_BITS 不合理：{index_bits}")
    pos += 8

    n = bits_to_int(extracted_bits[pos : pos + 32])
    if n > 50_000_000:
        raise ValueError(f"封包 n 不合理：{n}")
    pos += 32

    indices = []
    for _ in range(n):
        if pos + index_bits > len(extracted_bits):
            raise ValueError("封包不完整：索引區段不足")
        indices.append(bits_to_int(extracted_bits[pos : pos + index_bits]))
        pos += index_bits

    if pos + 8 > len(extracted_bits):
        raise ValueError("封包不完整：缺少檔案名稱長度 L")
    l_len = bits_to_int(extracted_bits[pos : pos + 8])
    pos += 8
    if pos + l_len * 8 > len(extracted_bits):
        raise ValueError("封包不完整：檔案名稱區段不足")
    filename = filename_from_bits(extracted_bits[pos : pos + l_len * 8])
    pos += l_len * 8

    if pos + 16 > len(extracted_bits):
        raise ValueError("封包不完整：缺少弱配對跳過數 W")
    weak_skip_count = bits_to_int(extracted_bits[pos : pos + 16])
    pos += 16
    weak_skips = []
    for _ in range(weak_skip_count):
        if pos + index_bits > len(extracted_bits):
            raise ValueError("封包不完整：弱配對跳過表不足")
        weak_skips.append(bits_to_int(extracted_bits[pos : pos + index_bits]))
        pos += index_bits

    if pos + 32 > len(extracted_bits):
        raise ValueError("封包不完整：缺少 data_len")
    data_len = bits_to_int(extracted_bits[pos : pos + 32])
    if data_len > len(extracted_bits) or data_len > 64 * 1024 * 1024 * 8:
        raise ValueError(f"封包 M 不合理：{data_len} bits（位元流可能未對齊）")
    pos += 32

    if pos + data_len + 32 > len(extracted_bits):
        raise ValueError(f"封包不完整：需要 DATA {data_len} bits + CRC 32 bits")
    secret_bits = extracted_bits[pos : pos + data_len]
    pos += data_len

    crc_read = bits_to_int(extracted_bits[pos : pos + 32])
    secret_bytes = bits_to_bytes(secret_bits)
    crc_calc = zlib.crc32(secret_bytes) & 0xFFFFFFFF
    if crc_calc != crc_read:
        raise ValueError(f"CRC 校驗失敗：計算 {crc_calc:#010x}，封包 {crc_read:#010x}")

    return indices, secret_bits, secret_bytes, filename


def extract_bits_from_safe_pairs(pairs, overflow_indices: list[int]) -> list[int]:
    overflow_set = set(overflow_indices)
    bits = []
    for idx, (a, b) in enumerate(pairs):
        if (idx + 1) in overflow_set:
            continue
        bits.append((int(a) - int(b)) & 1)
    return bits


def iter_bits_from_safe_pairs(pairs, overflow_indices: list[int]):
    """逐 bit 產生，供 magic 前 16 bit 快速拒絕用。"""
    overflow_set = set(overflow_indices)
    for idx, (a, b) in enumerate(pairs):
        if (idx + 1) in overflow_set:
            continue
        yield (int(a) - int(b)) & 1


def _read_head_bits(bit_iter, count: int) -> list[int] | None:
    head = []
    for _ in range(count):
        try:
            head.append(next(bit_iter))
        except StopIteration:
            return None
    return head


def extract_bits_from_expandable_pairs(pairs) -> list[int]:
    """解碼時：從通過 H×2 試算的安全通道依序取 bit。"""
    bits = []
    for a, b in pairs:
        if _pair_expandable(a, b):
            bits.append((int(a) - int(b)) & 1)
    return bits


# =====================================================
# 嵌入
# =====================================================
def embed_DE_core(
    matrix,
    secret_bytes,
    filename: str = "secret.txt",
    raise_on_overflow: bool = True,
):
    matrix, rows, cols = _ensure_even_matrix(matrix)
    flat, pairs = _flatten_pairs(matrix)
    loc_map, overflow_pairs = build_loc_map(pairs)
    overflow_pairs = sorted(overflow_pairs)

    pair_total = len(pairs)

    if isinstance(secret_bytes, bytes):
        data_bytes = secret_bytes
    else:
        data_bytes = secret_bytes.encode("utf-8")

    data_bits = len(data_bytes) * 8
    payload_needed = data_bits + 32
    name_norm = normalize_filename(filename)
    n = len(overflow_pairs)
    header_reserved, weak_skips = resolve_packet_header_and_weak_skips(
        pairs, loc_map, n, pair_total, name_norm, payload_needed
    )
    bitstream = build_bitstream(
        overflow_pairs, data_bytes, pair_total, name_norm, weak_skips
    )
    safe_pairs = sum(loc_map)
    index_bits = compute_index_bits(pair_total)
    fixed_overhead = bitstream_fixed_overhead(
        n, pair_total, name_norm, weak_skip_count=len(weak_skips)
    )
    continuous_safe, _ = validate_header_region(
        pairs, n, filename=name_norm, raise_on_fail=raise_on_overflow
    )
    payload_safe = count_payload_embed_slots(pairs, loc_map, header_reserved)

    logs = [
        f"圖片尺寸（偶數裁剪後）：{rows} x {cols}",
        f"像素對總數 N = {pair_total}，安全配對 = {safe_pairs}",
        f"INDEX_BITS = {index_bits}（動態索引寬度）",
        f"連續安全區（從第 0 組）= {continuous_safe} 組",
        (
            f"封包標頭保留 = {header_reserved} bits"
            f"（magic+idx_bits+n+索引+L+檔名+W+ M）"
        ),
        f"弱配對跳過 W = {len(weak_skips)} 組（strict walk 前綴）",
        f"承載區嚴格槽位 = {payload_safe} 組（標頭之後，0/1 皆可藏）",
        f"溢位點 n = {n}",
        (
            f"封包 = {len(bitstream)} bits"
            f"（標頭 {header_reserved} + DATA {data_bits} + CRC 32）"
        ),
        f"CRC32(DATA) = {zlib.crc32(data_bytes) & 0xFFFFFFFF:#010x}",
    ]

    if overflow_pairs:
        shown = overflow_pairs[:20]
        suffix = f" …共 {len(overflow_pairs)} 組" if len(overflow_pairs) > 20 else ""
        logs.append(f"溢位組別：第 {shown}{suffix} 組")

    if continuous_safe < header_reserved:
        msg = (
            f"連續安全區不足！僅 {continuous_safe} 組，"
            f"封包標頭需 {header_reserved} bits。"
            f"圖片開頭對比太強，不適合 DE。"
        )
        if raise_on_overflow:
            raise ValueError(msg)
        logs.append(f"警告：{msg}")

    if payload_safe < payload_needed:
        msg = (
            f"承載區容量不足！標頭後安全配對={payload_safe}，"
            f"需要 {payload_needed} bits（DATA {data_bits} + CRC 32）"
        )
        if raise_on_overflow:
            raise ValueError(msg)
        logs.append(f"警告：{msg}")

    ok, verify_err, fail_pair = verify_embed_bitstream(
        pairs, loc_map, bitstream, header_reserved
    )
    if not ok:
        if raise_on_overflow:
            raise ValueError(verify_err)
        logs.append(f"警告：{verify_err}")

    header_stream = bitstream[:header_reserved]
    payload_stream = bitstream[header_reserved:]

    new_flat = flat[:]

    for pair_idx in range(header_reserved):
        _embed_bit_into_pair(
            new_flat, pair_idx, header_stream[pair_idx], raise_on_overflow=raise_on_overflow
        )

    payload_slots = payload_embed_slot_indices_strict(pairs, loc_map, header_reserved)

    payload_idx = 0
    for pair_idx in payload_slots:
        if payload_idx >= len(payload_stream):
            break
        _embed_bit_into_pair(
            new_flat,
            pair_idx,
            payload_stream[payload_idx],
            raise_on_overflow=raise_on_overflow,
        )
        payload_idx += 1

    bit_idx = header_reserved + payload_idx
    logs.append(
        f"實際嵌入了 {bit_idx} 個 bit"
        f"（標頭 {header_reserved} + 承載 {payload_idx}）"
    )

    new_matrix = [new_flat[i : i + cols] for i in range(0, len(new_flat), cols)]
    meta = {
        "overflow_pairs": overflow_pairs,
        "overflow_count": n,
        "bits_embedded": bit_idx,
        "safe_pairs": safe_pairs,
        "pair_total": pair_total,
        "required_bits": len(bitstream),
        "header_bits": fixed_overhead,
        "header_reserved_bits": header_reserved,
        "continuous_safe": continuous_safe,
        "payload_safe_pairs": payload_safe,
        "secret_bits": data_bits,
        "crc32": zlib.crc32(data_bytes) & 0xFFFFFFFF,
        "index_bits": index_bits,
        "filename": name_norm,
        "method": "DE_V4",
        "magic": PACKET_MAGIC,
        "weak_skip_count": len(weak_skips),
        "weak_skip_indices": weak_skips,
    }
    return new_matrix, loc_map, logs, meta


# =====================================================
# 提取 + 還原
# =====================================================
def _likely_embedded_pair(a, b) -> bool:
    """stego 配對是否承載了嵌入 bit（可逆推回可擴展的封面）。"""
    if _pair_expandable(a, b):
        return True
    L = (int(a) + int(b)) // 2
    H = int(a) - int(b)
    bit = H & 1
    H_c = (H - bit) // 2
    a_c = L + ((H_c + 1) // 2)
    b_c = L - (H_c // 2)
    return 0 <= a_c <= 255 and 0 <= b_c <= 255 and _pair_expandable(a_c, b_c)


def _overflow_alignment_scan(pairs) -> tuple[list[int], list[int]]:
    """單次掃描：回傳 (refined_skip, expandable_fail)。"""
    expandable_fail = []
    guess: set[int] = set()
    for i, (a, b) in enumerate(pairs):
        if not _pair_expandable(a, b):
            expandable_fail.append(i + 1)
            guess.add(i + 1)
    for idx in list(guess):
        if _likely_embedded_pair(*pairs[idx - 1]):
            guess.discard(idx)
    return sorted(guess), expandable_fail


def _refined_skip_list(pairs) -> list[int]:
    """位元流對齊：跳過原始溢位點，排除嵌入後誤判為溢位的假陽性配對。"""
    refined, _ = _overflow_alignment_scan(pairs)
    return refined


def _decode_skip_candidates(pairs):
    """位元流對齊候選：僅用於讀出封包，最終溢位索引以 CRC 通過的封包為準。"""
    refined, expandable_fail = _overflow_alignment_scan(pairs)
    if refined:
        yield refined
    yield []
    if expandable_fail and expandable_fail != refined:
        yield expandable_fail


def _parse_header_fields(header_bits: list[int]) -> tuple[list[int], int, int, str]:
    """從標頭位元流解析溢位索引、DATA 長度、標頭 bit 數、檔案名稱。"""
    pos = 0
    magic = bits_to_int(header_bits[pos : pos + 16])
    if not is_stego_magic(magic):
        raise ValueError(f"標頭 magic 錯誤：{magic:#06x}")
    pos += 16

    index_bits = bits_to_int(header_bits[pos : pos + 8])
    if index_bits < 1 or index_bits > 32:
        raise ValueError(f"INDEX_BITS 不合理：{index_bits}")
    pos += 8

    n = bits_to_int(header_bits[pos : pos + 32])
    if n > 50_000_000:
        raise ValueError(f"封包 n 不合理：{n}")
    pos += 32

    indices = []
    for _ in range(n):
        if pos + index_bits > len(header_bits):
            raise ValueError("封包不完整：索引區段不足")
        indices.append(bits_to_int(header_bits[pos : pos + index_bits]))
        pos += index_bits

    if pos + 8 > len(header_bits):
        raise ValueError("封包不完整：缺少檔案名稱長度 L")
    l_len = bits_to_int(header_bits[pos : pos + 8])
    pos += 8
    if pos + l_len * 8 > len(header_bits):
        raise ValueError("封包不完整：檔案名稱區段不足")
    filename = filename_from_bits(header_bits[pos : pos + l_len * 8])
    pos += l_len * 8

    if pos + 32 > len(header_bits):
        raise ValueError("封包不完整：缺少 data_len")
    data_len = bits_to_int(header_bits[pos : pos + 32])
    if data_len > 64 * 1024 * 1024 * 8:
        raise ValueError(f"封包 M 不合理：{data_len} bits")
    pos += 32
    header_bit_count = pos
    return indices, data_len, header_bit_count, filename


def _read_packet_header_from_pairs(pairs) -> tuple[list[int], int, int, str, list[int]]:
    """從圖像配對連續讀取封包標頭（至 M 欄位結尾）。"""
    pos = 0

    def read(n: int) -> list[int]:
        nonlocal pos
        if pos + n > len(pairs):
            raise ValueError("封包標頭超出圖片容量")
        bits = extract_bits_from_pairs(pairs, pos, n)
        pos += n
        return bits

    magic = bits_to_int(read(16))
    if not is_stego_magic(magic):
        raise ValueError(f"找不到已知隱寫封包（讀到 {magic:#06x}）")

    index_bits = bits_to_int(read(8))
    if index_bits < 1 or index_bits > 32:
        raise ValueError(f"INDEX_BITS 不合理：{index_bits}")

    n = bits_to_int(read(32))
    if n > 50_000_000:
        raise ValueError(f"封包 n 不合理：{n}")

    indices = []
    for _ in range(n):
        indices.append(bits_to_int(read(index_bits)))

    l_len = bits_to_int(read(8))
    filename = filename_from_bits(read(l_len * 8))

    weak_skip_count = bits_to_int(read(16))
    weak_skips = [bits_to_int(read(index_bits)) for _ in range(weak_skip_count)]

    data_len = bits_to_int(read(32))
    if data_len > 64 * 1024 * 1024 * 8:
        raise ValueError(f"封包 M 不合理：{data_len} bits")

    return indices, data_len, pos, filename, weak_skips


def _extract_packet_bits(pairs, cover_pairs=None) -> list[int]:
    """標頭從第 0 組連續讀；承載區跳過標頭與溢位索引。"""
    if len(pairs) < 64:
        raise ValueError("配對數不足，無法讀取封包")

    indices, data_len, header_reserved, _filename, weak_skips = _read_packet_header_from_pairs(
        pairs
    )
    payload_needed = data_len + 32
    payload_bits = extract_payload_bits(
        pairs,
        indices,
        header_reserved,
        data_len=data_len,
        cover_pairs=cover_pairs,
        weak_skip_indices=weak_skips,
    )
    if len(payload_bits) < payload_needed:
        raise ValueError(
            f"承載區不足：需要 DATA {data_len} bits + CRC 32 bits，"
            f"僅讀到 {len(payload_bits)} bits"
        )
    header_bits = extract_bits_from_pairs(pairs, 0, header_reserved)
    return header_bits + payload_bits[:payload_needed]


def probe_de_packet(matrix) -> tuple[list[int], list[int], bytes, str] | None:
    """快速解析封包：標頭連續讀 + CRC32 驗證（不還原圖像）。"""
    matrix, _, _ = _ensure_even_matrix(matrix)
    _, pairs = _flatten_pairs(matrix)
    try:
        full_bits = _extract_packet_bits(pairs)
        return parse_bitstream(full_bits)
    except ValueError:
        return None


def peek_de_magic(matrix) -> bool:
    """僅讀前 16 bit 是否為已知隱寫 magic（不做 CRC）。"""
    matrix, _, _ = _ensure_even_matrix(matrix)
    _, pairs = _flatten_pairs(matrix)
    if len(pairs) < 16:
        return False
    head = extract_bits_sequential(pairs, 16)
    return is_stego_magic(bits_to_int(head))


def extract_DE_core(matrix, cover_pairs=None):
    """解碼：標頭連續讀 → 解析索引 → 承載區 strict 槽 → 驗 CRC32。"""
    matrix, _, _ = _ensure_even_matrix(matrix)
    _, pairs = _flatten_pairs(matrix)

    try:
        full_bits = _extract_packet_bits(pairs, cover_pairs=cover_pairs)
        indices, secret_bits_raw, secret_bytes, filename = parse_bitstream(full_bits)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    loc_map = loc_map_from_overflow_indices(len(pairs), indices)
    return secret_bytes, loc_map, indices, len(full_bits), filename


def restore_DE_core(
    matrix,
    overflow_indices: list[int],
    bits_embedded: int,
    filename: str = "secret.txt",
    cover_pairs=None,
):
    matrix, _, cols = _ensure_even_matrix(matrix)
    flat, pairs = _flatten_pairs(matrix)
    restored = flat[:]
    _overflow, data_len, header_reserved, _filename, weak_skips = _read_packet_header_from_pairs(
        pairs
    )

    for pair_idx in range(header_reserved):
        _restore_pair(restored, pair_idx)

    payload_bits_to_restore = max(0, bits_embedded - header_reserved)
    payload_slots = payload_embed_slot_indices_decode(
        pairs,
        overflow_indices,
        header_reserved,
        data_len=data_len,
        cover_pairs=cover_pairs,
        weak_skip_indices=weak_skips,
    )
    for pair_idx in payload_slots[:payload_bits_to_restore]:
        _restore_pair(restored, pair_idx)

    return [restored[i : i + cols] for i in range(0, len(restored), cols)]


# =====================================================
# 彩色圖 BGR 拉平（方案 C）：[B 全部] + [G 全部] + [R 全部]
# =====================================================
def is_color_image(img: np.ndarray) -> bool:
    return img.ndim == 3 and img.shape[2] >= 3


def _crop_even_hw(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    if h % 2 == 1:
        img = img[:-1]
        h -= 1
    if w % 2 == 1:
        img = img[:, :-1]
        w -= 1
    return img, h, w  # 裁剪為偶數尺寸後的圖、高、寬


def image_to_de_matrix(img: np.ndarray) -> tuple[list, dict]:
    """將灰階或彩色 BGR 圖轉成 DE 用的 2D matrix 與 metadata。"""
    if is_color_image(img):
        img = img[:, :, :3].copy()
        img, h, w = _crop_even_hw(img)
        flat = np.concatenate([
            img[:, :, 0].ravel(),
            img[:, :, 1].ravel(),
            img[:, :, 2].ravel(),
        ])
        matrix = flat.reshape(h, w * 3).tolist()
        meta = {
            "mode": "color",           # 圖片模式：彩色 BGR 拉平
            "orig_h": h,               # 裁剪後高度
            "orig_w": w,               # 裁剪後寬度
            "matrix_cols": w * 3,      # DE 矩陣欄數（W×3）
        }
        # matrix: DE 用二維矩陣 | meta: 還原用的版面資訊
        return matrix, meta

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    matrix, rows, cols = _ensure_even_matrix(img.tolist())
    # matrix: DE 用二維矩陣 | layout: 灰階版面資訊
    return matrix, {"mode": "gray", "orig_h": rows, "orig_w": cols}  # mode=gray, orig_h/w=裁剪後尺寸


def de_matrix_to_image(matrix, layout: dict) -> np.ndarray:
    """將 DE matrix 還原成灰階或 BGR 圖。"""
    if layout["mode"] == "gray":
        return np.array(matrix, dtype=np.uint8)

    h, w = layout["orig_h"], layout["orig_w"]
    flat = np.array(matrix, dtype=np.uint8).ravel()
    n = h * w
    b = flat[0:n].reshape(h, w)
    g = flat[n : 2 * n].reshape(h, w)
    r = flat[2 * n : 3 * n].reshape(h, w)
    return cv2.merge([b, g, r])


# =====================================================
# 容量分析
# =====================================================
def benchmark_pair_safety_checks(pairs, *, reps: int = 5) -> dict:
    """比較 strict 解析邊界 vs 雙 bit 試算掃描耗時（後台統計用）。"""
    import time

    n = len(pairs)
    if n == 0:
        return {"pair_total": 0}

    t_bounds = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        c = 0
        for a, b in pairs:
            if is_strict_safe(a, b):
                c += 1
        t_bounds += time.perf_counter() - t0

    t_bitwise = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        c = 0
        for a, b in pairs:
            if _pair_expandable_strict_bitwise(a, b):
                c += 1
        t_bitwise += time.perf_counter() - t0

    ms_bounds = t_bounds / reps * 1000
    ms_bitwise = t_bitwise / reps * 1000
    speedup = ms_bitwise / ms_bounds if ms_bounds > 0 else 0.0
    return {
        "pair_total": n,
        "reps": reps,
        "strict_bounds_ms": round(ms_bounds, 3),
        "strict_bitwise_ms": round(ms_bitwise, 3),
        "speedup_x": round(speedup, 2),
        "strict_bounds_count": sum(1 for a, b in pairs if is_strict_safe(a, b)),
        "strict_bitwise_count": sum(
            1 for a, b in pairs if _pair_expandable_strict_bitwise(a, b)
        ),
    }


def analyze_payload_embed_tiers(
    pairs,
    loc_map: list[int],
    header_reserved: int,
) -> dict:
    """承載區弱配對分級：strict=0/1皆可；weak 僅一種 bit 可藏（loc_map=1 但未達 strict）。"""
    strict = 0
    weak_bit0_only = 0
    weak_bit1_only = 0
    for idx in range(header_reserved, len(pairs)):
        if loc_map[idx] != 1:
            continue
        a, b = pairs[idx]
        ok0 = _pair_embed_bit_in_range(a, b, 0)
        ok1 = _pair_embed_bit_in_range(a, b, 1)
        if ok0 and ok1:
            strict += 1
        elif ok0:
            weak_bit0_only += 1
        elif ok1:
            weak_bit1_only += 1
    optimistic = strict + weak_bit0_only + weak_bit1_only
    pessimistic = strict
    opt_bits = max(0, optimistic - 32)
    pes_bits = max(0, pessimistic - 32)
    return {
        "payload_strict_pairs": strict,
        "payload_weak_bit0_only": weak_bit0_only,
        "payload_weak_bit1_only": weak_bit1_only,
        "payload_optimistic_pairs": optimistic,
        "payload_pessimistic_pairs": pessimistic,
        "payload_optimistic_bits": opt_bits,
        "payload_pessimistic_bits": pes_bits,
        "max_data_bytes_optimistic": opt_bits // 8,
        "max_data_bytes_pessimistic": pes_bits // 8,
    }


def analyze_de_compatibility(matrix) -> dict:
    """快速評估圖片的 DE 適用性（藏 1 最嚴格試算）。"""
    matrix, _, _ = _ensure_even_matrix(matrix)
    _, pairs = _flatten_pairs(matrix)
    total = len(pairs)
    safe = sum(1 for a, b in pairs if _pair_expandable_strict(a, b))
    ratio = safe / total if total > 0 else 0.0
    return {
        "compat_safe": safe,
        "compat_total": total,
        "compat_ratio": round(ratio, 4),
        "compat_ratio_pct": round(ratio * 100, 1),
        "compat_grade": _de_compatibility_grade(ratio),
    }


def analyze_index_bits_profile(
    pairs,
    loc_map: list[int],
    overflow_pairs: list[int],
    *,
    filename: str = "secret.txt",
    payload_bytes: int = 0,
    weak_skip_indices: list[int] | None = None,
    header_reserved_bits: int | None = None,
) -> dict:
    """
    紀錄 INDEX_BITS1 / INDEX_BITS2 拆分是否划算的分析用數字。
    供 JSONL 累積與事後評估（不改封包格式）。
    """
    pair_total = len(pairs)
    n = len(overflow_pairs)
    ib_abs = compute_index_bits(pair_total)
    name_norm = normalize_filename(filename)
    payload_needed = payload_bytes * 8 + 32 if payload_bytes > 0 else 0

    if header_reserved_bits is None:
        if payload_needed > 0:
            header_reserved_bits, weak_skip_indices = resolve_packet_header_and_weak_skips(
                pairs, loc_map, n, pair_total, name_norm, payload_needed
            )
        else:
            header_reserved_bits = packet_header_bits(n, pair_total, name_norm, weak_skip_count=0)
            weak_skip_indices = weak_skip_indices or []

    weak_skip_indices = list(weak_skip_indices or [])
    w_count = len(weak_skip_indices)
    header_pairs = header_reserved_bits  # 標頭區 1 bit/組，第 0 組起連續

    ib_weak_rel_cap = 1
    if pair_total > header_pairs:
        ib_weak_rel_cap = max(1, (pair_total - header_pairs).bit_length())

    rel_values: list[int] = []
    if weak_skip_indices:
        for pair_1based in weak_skip_indices:
            rel = int(pair_1based) - header_pairs
            if rel > 0:
                rel_values.append(rel)
    ib_weak_rel_actual = (
        max(1, max(rel_values).bit_length()) if rel_values else ib_weak_rel_cap
    )

    tiers = analyze_payload_embed_tiers(pairs, loc_map, header_reserved_bits)
    weak0_payload = tiers.get("payload_weak_bit0_only", 0)
    weak1_payload = tiers.get("payload_weak_bit1_only", 0)

    iovf_bits = n * ib_abs
    iweak_bits_now = w_count * ib_abs
    iweak_bits_rel = w_count * ib_weak_rel_actual
    save_per_entry = ib_abs - ib_weak_rel_actual
    save_relative_total = w_count * max(0, save_per_entry)
    extra_index_bits2_field = 8
    net_if_split_field = save_relative_total - extra_index_bits2_field

    return {
        "N": pair_total,
        "n_overflow": n,
        "payload_bytes": payload_bytes,
        "payload_needed_bits": payload_needed,
        "header_reserved_bits": header_reserved_bits,
        "header_reserved_pairs": header_pairs,
        "index_bits_current": ib_abs,
        "index_bits_ovf_absolute": ib_abs,
        "index_bits_weak_absolute": ib_abs,
        "index_bits_weak_relative_cap": ib_weak_rel_cap,
        "index_bits_weak_relative_actual": ib_weak_rel_actual,
        "weak_skip_count": w_count,
        "weak_skip_pair_min": min(weak_skip_indices) if weak_skip_indices else None,
        "weak_skip_pair_max": max(weak_skip_indices) if weak_skip_indices else None,
        "weak_rel_min": min(rel_values) if rel_values else None,
        "weak_rel_max": max(rel_values) if rel_values else None,
        "payload_weak0_slots_total": weak0_payload,
        "payload_weak1_slots_total": weak1_payload,
        "iovf_bits": iovf_bits,
        "iweak_bits_current": iweak_bits_now,
        "iweak_bits_if_relative": iweak_bits_rel,
        "save_bits_if_relative_encoding": save_relative_total,
        "cost_extra_index_bits2_field": extra_index_bits2_field,
        "net_save_if_split_field": net_if_split_field,
        "worth_split_index_bits": net_if_split_field > 0,
        "index_bits_same_today": True,
        "note": (
            "INDEX_BITS1=INDEX_BITS2=index_bits_current；"
            "若 I_weak 改相對 header 編碼且另加 8-bit INDEX_BITS2 欄位，"
            "net_save_if_split_field>0 才值得改格式"
        ),
    }


def append_index_bits_stats_record(
    record: dict,
    path: str | Path | None = None,
) -> Path:
    """追加一筆圖片 INDEX_BITS 分析到 JSONL（供事後統計）。"""
    import json
    from datetime import datetime, timezone

    out = Path(path) if path else Path(__file__).resolve().parent / "data" / "image_index_bits_log.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    line = record | {"recorded_at": datetime.now(timezone.utc).isoformat()}
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return out


def analyze_capacity(img_or_matrix, filename: str = "secret.txt") -> dict:
    timing: dict[str, int] = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    if isinstance(img_or_matrix, np.ndarray):
        matrix, layout = image_to_de_matrix(img_or_matrix)
        color_mode = layout["mode"]
    else:
        matrix = img_or_matrix
        layout = {"mode": "gray"}
        color_mode = "gray"

    matrix, rows, cols = _ensure_even_matrix(matrix)
    _, pairs = _flatten_pairs(matrix)
    timing["matrix_ms"] = _ms_since(t0)

    t0 = time.perf_counter()
    loc_map, overflow_pairs = build_loc_map(pairs)
    timing["de_loc_map_ms"] = _ms_since(t0)

    t0 = time.perf_counter()
    pair_total = len(pairs)
    safe_pairs = sum(loc_map)
    n = len(overflow_pairs)
    continuous_safe = count_continuous_safe_pairs(pairs)
    index_bits = compute_index_bits(pair_total)
    name_norm = normalize_filename(filename)
    header_reserved = packet_header_bits(n, pair_total, name_norm)
    header_bits = bitstream_fixed_overhead(n, pair_total, name_norm)
    payload_safe = count_payload_embed_slots(pairs, loc_map, header_reserved)
    header_ok = continuous_safe >= header_reserved
    tiers = analyze_payload_embed_tiers(pairs, loc_map, header_reserved)
    cap_profile = compute_capacity_profile(
        pair_total, overflow_pairs, loc_map, pairs, continuous_safe, header_reserved
    )
    overflow_embed_map = build_overflow_embed_map(
        overflow_pairs, index_bits, header_reserved, pair_total, pairs, filename=name_norm
    )
    index_bits_profile = analyze_index_bits_profile(
        pairs, loc_map, overflow_pairs, filename=name_norm, payload_bytes=0
    )
    walk_cache = build_strict_walk_cache(pairs, loc_map)
    guaranteed = resolve_strict_payload_capacity(
        pairs,
        loc_map,
        overflow_pairs,
        pair_total,
        name_norm,
        payload_bytes=0,
        strict_walk_cache=walk_cache,
    )
    timing["capacity_rest_ms"] = _ms_since(t0)
    timing["total_ms"] = _ms_since(t_all)

    return {
        "mode": color_mode,
        "width": cols,
        "height": rows,
        "display_width": layout.get("orig_w", cols),
        "display_height": layout.get("orig_h", rows),
        "pair_total": pair_total,
        "safe_pairs": safe_pairs,
        "overflow_pairs": overflow_pairs,
        "overflow_count": n,
        "overflow_first": overflow_pairs[0] if overflow_pairs else None,
        "overflow_last": overflow_pairs[-1] if overflow_pairs else None,
        "overflow_first_embed_bit": overflow_embed_map[0]["embed_bit"] if overflow_embed_map else None,
        "overflow_last_embed_bit": overflow_embed_map[-1]["embed_bit"] if overflow_embed_map else None,
        "overflow_embed_map": overflow_embed_map,
        "index_bits": index_bits,
        "secret_filename": name_norm,
        "continuous_safe": continuous_safe,
        "header_reserved_bits": header_reserved,
        "header_bits": header_bits,
        "payload_safe_pairs": payload_safe,
        "capacity_bits": safe_pairs,
        "payload_bits": cap_profile["max_data_bits"],
        "max_data_bits": cap_profile["max_data_bits"],
        "max_text_bytes": cap_profile["max_text_bytes"],
        "header_region_ok": header_ok,
        "capacity_ok": cap_profile["capacity_ok"],
        "limiting_factor": cap_profile["limiting_factor"],
        "capacity": cap_profile,
        "index_bits_profile": index_bits_profile,
        "timing_ms": timing,
        **guaranteed,
        **tiers,
    }


def analysis_from_embed_meta(img: np.ndarray, meta: dict, secret_text: str = "") -> dict:
    """嵌入完成後組 UI 用分析結果，避免再掃一次 loc_map。"""
    matrix, layout = image_to_de_matrix(img)
    _, pairs = _flatten_pairs(matrix)
    loc_map, overflow_pairs = build_loc_map(pairs)
    ib = meta["index_bits"]
    n = meta["overflow_count"]
    hdr = meta["header_reserved_bits"]
    pair_total = meta["pair_total"]
    overflow_pairs = meta["overflow_pairs"]
    fname = meta.get("filename", "secret.txt")
    overflow_embed_map = build_overflow_embed_map(
        overflow_pairs, ib, hdr, pair_total, pairs, filename=fname
    )
    secret_bits = meta.get("secret_bits", len(secret_to_bits(secret_text)) if secret_text else 0)
    payload_needed = secret_bits + 32
    weak_skip_count = meta.get("weak_skip_count", 0)
    weak_skip_indices = meta.get("weak_skip_indices", [])
    header_bits = bitstream_fixed_overhead(
        n, pair_total, fname, weak_skip_count=weak_skip_count
    )
    payload_safe = meta["payload_safe_pairs"]
    header_ok = meta.get("continuous_safe", 0) >= hdr
    tiers = analyze_payload_embed_tiers(pairs, loc_map, hdr)
    cap_profile = compute_capacity_profile(
        pair_total, overflow_pairs, loc_map, pairs, meta.get("continuous_safe", 0), hdr
    )
    required = hdr + secret_bits + 32
    payload_bytes = secret_bits // 8 if secret_bits else 0
    guaranteed = resolve_strict_payload_capacity(
        pairs, loc_map, overflow_pairs, pair_total, fname, payload_bytes=0
    )
    if payload_bytes > 0:
        payload_cap = resolve_strict_payload_capacity(
            pairs, loc_map, overflow_pairs, pair_total, fname, payload_bytes=payload_bytes
        )
        fits = (
            cap_profile["capacity_ok"]
            and secret_bits <= guaranteed["max_data_bits_guaranteed"]
            and payload_cap["formula_fits"]
        )
    else:
        payload_cap = None
        fits = cap_profile["capacity_ok"]
    index_bits_profile = analyze_index_bits_profile(
        pairs,
        loc_map,
        overflow_pairs,
        filename=fname,
        payload_bytes=payload_bytes,
        weak_skip_indices=weak_skip_indices,
        header_reserved_bits=hdr,
    )
    return {
        "mode": layout["mode"],
        "width": layout.get("orig_w", img.shape[1]),
        "height": layout.get("orig_h", img.shape[0]),
        "display_width": layout.get("orig_w", img.shape[1]),
        "display_height": layout.get("orig_h", img.shape[0]),
        "pair_total": pair_total,
        "safe_pairs": meta["safe_pairs"],
        "overflow_pairs": overflow_pairs,
        "overflow_count": n,
        "overflow_first": overflow_pairs[0] if overflow_pairs else None,
        "overflow_last": overflow_pairs[-1] if overflow_pairs else None,
        "overflow_first_embed_bit": overflow_embed_map[0]["embed_bit"] if overflow_embed_map else None,
        "overflow_last_embed_bit": overflow_embed_map[-1]["embed_bit"] if overflow_embed_map else None,
        "overflow_embed_map": overflow_embed_map,
        "index_bits": ib,
        "secret_filename": fname,
        "continuous_safe": meta.get("continuous_safe"),
        "header_reserved_bits": hdr,
        "header_bits": header_bits,
        "payload_safe_pairs": payload_safe,
        "capacity_bits": meta["safe_pairs"],
        "payload_bits": cap_profile["max_data_bits"],
        "max_data_bits": cap_profile["max_data_bits"],
        "max_text_bytes": cap_profile["max_text_bytes"],
        "header_region_ok": header_ok,
        "capacity_ok": cap_profile["capacity_ok"],
        "limiting_factor": cap_profile["limiting_factor"],
        "capacity": cap_profile,
        "secret_bits": secret_bits,
        "required_bits": required,
        "weak_skip_count": weak_skip_count,
        "weak_skip_indices": weak_skip_indices[:50],
        "index_bits_profile": index_bits_profile,
        "fits": fits,
        **guaranteed,
        **tiers,
    }


def analyze_secret_fit(
    img_or_matrix,
    secret: str | bytes = "",
    filename: str = "secret.txt",
) -> dict:
    t_all = time.perf_counter()
    cap = analyze_capacity(img_or_matrix, filename=filename)
    timing = dict(cap.get("timing_ms") or {})
    if isinstance(secret, bytes):
        nbytes = len(secret)
        data_bytes = secret
    elif secret:
        data_bytes = secret.encode("utf-8")
        nbytes = len(data_bytes)
    else:
        nbytes = 0
        data_bytes = b""

    if isinstance(img_or_matrix, np.ndarray):
        matrix, _ = image_to_de_matrix(img_or_matrix)
    else:
        matrix = img_or_matrix
    matrix, _, _ = _ensure_even_matrix(matrix)
    _, pairs = _flatten_pairs(matrix)
    t_loc = time.perf_counter()
    loc_map, overflow_pairs = build_loc_map(pairs)
    timing["de_loc_map_repeat_ms"] = _ms_since(t_loc)
    pair_total = len(pairs)
    n = len(overflow_pairs)
    name_norm = normalize_filename(filename)

    walk_cache = build_strict_walk_cache(pairs, loc_map)
    secret_bits_count = nbytes * 8
    payload_needed = secret_bits_count + 32
    required = cap["header_reserved_bits"] + secret_bits_count + 32

    embed_feasible = True
    embed_error = None
    embed_fail_pair = None
    weak_skips: list[int] = []
    header_reserved = cap["header_reserved_bits"]
    payload_cap = None
    t_extra = time.perf_counter()
    if nbytes > 0:
        header_reserved, weak_skips = resolve_packet_header_and_weak_skips(
            pairs,
            loc_map,
            n,
            pair_total,
            name_norm,
            payload_needed,
            strict_walk_cache=walk_cache,
            need_weak_indices=True,
        )
        bitstream = build_bitstream(
            overflow_pairs, data_bytes, pair_total, filename, weak_skips
        )
        required = header_reserved + payload_needed
        t_verify = time.perf_counter()
        embed_feasible, embed_error, embed_fail_pair = verify_embed_bitstream(
            pairs,
            loc_map,
            bitstream,
            header_reserved,
        )
        timing["verify_embed_ms"] = _ms_since(t_verify)
        strict_slots = count_payload_embed_slots(pairs, loc_map, header_reserved)
        payload_cap = {
            "payload_strict_slots": strict_slots,
            "formula_fits": payload_needed <= strict_slots,
        }
        fits = (
            cap["capacity_ok"]
            and secret_bits_count <= cap["max_data_bits_guaranteed"]
            and payload_cap["formula_fits"]
        )
    else:
        fits = cap["capacity_ok"]

    timing["fit_extra_ms"] = _ms_since(t_extra)
    timing["analyze_total_ms"] = _ms_since(t_all)

    index_bits_profile = analyze_index_bits_profile(
        pairs,
        loc_map,
        overflow_pairs,
        filename=filename,
        payload_bytes=nbytes,
        weak_skip_indices=weak_skips,
        header_reserved_bits=header_reserved,
    )

    return {
        **cap,
        "secret_bits": secret_bits_count,
        "required_bits": required,
        "header_reserved_bits": header_reserved,
        "weak_skip_count": len(weak_skips),
        "weak_skip_indices": weak_skips[:50],
        "index_bits_profile": index_bits_profile,
        "payload_strict_slots": payload_cap["payload_strict_slots"] if payload_cap else None,
        "formula_fits": payload_cap["formula_fits"] if payload_cap else None,
        "fits": fits,
        "embed_feasible": embed_feasible,
        "embed_error": embed_error,
        "embed_fail_pair": embed_fail_pair,
        "timing_ms": timing,
    }


def _pair_position(pair_idx: int, cols: int) -> dict:
    i0, i1 = pair_idx * 2, pair_idx * 2 + 1
    r0, c0 = divmod(i0, cols)
    r1, c1 = divmod(i1, cols)
    return {"row_a": r0, "col_a": c0, "row_b": r1, "col_b": c1}  # 配對兩像素在矩陣中的座標


def analyze_overflow(img: np.ndarray) -> dict:
    orig_h, orig_w = img.shape[:2]
    color = is_color_image(img)
    matrix, layout = image_to_de_matrix(img)
    matrix, rows, cols = _ensure_even_matrix(matrix)
    _, pairs = _flatten_pairs(matrix)
    loc_map, overflow_pairs = build_loc_map(pairs)

    details = []
    for pair_num in overflow_pairs[:100]:
        pair_idx = pair_num - 1
        a, b = pairs[pair_idx]
        L = (a + b) // 2
        H = int(a) - int(b)
        H_test = H * 2
        a_test = L + ((H_test + 1) // 2)
        b_test = L - (H_test // 2)
        pos = _pair_position(pair_idx, cols)
        if color:
            pos = _color_pair_position(pair_idx, cols, layout["orig_h"], layout["orig_w"])
        details.append({
            "pair": pair_num,           # 第 N 組（1-based）
            **pos,
            "a": int(a),                # 像素 A 值
            "b": int(b),                # 像素 B 值
            "L": int(L),                # 平均值 L
            "H": int(H),                # 差值 H
            "a_test": int(a_test),      # 試算後 A'
            "b_test": int(b_test),      # 試算後 B'
            "reason": f"A'={a_test} 或 B'={b_test} 超出 [0,255]",  # 溢位原因
        })

    pair_total = len(pairs)
    overflow_count = len(overflow_pairs)
    safe_pairs = sum(loc_map)
    header_example = bitstream_fixed_overhead(overflow_count, pair_total)

    if overflow_count == 0:
        verdict = "通過：所有像素對均可安全嵌入（無溢位）"
    else:
        verdict = f"有溢位：{overflow_count} 組（第 {overflow_pairs[:10]} 組…）"

    return {
        "mode": layout.get("mode", "gray"),              # gray 或 color
        "original_width": int(orig_w),                   # 原始寬度
        "original_height": int(orig_h),                  # 原始高度
        "width": cols,                                   # DE 矩陣寬
        "height": rows,                                  # DE 矩陣高
        "display_width": layout.get("orig_w", cols),     # 顯示用寬
        "display_height": layout.get("orig_h", rows),    # 顯示用高
        "trimmed": orig_h != layout.get("orig_h", rows) or orig_w != layout.get("orig_w", cols),  # 是否裁剪奇數邊
        "pair_total": pair_total,                        # 像素對總數
        "safe_pairs": safe_pairs,                        # 安全配對數
        "overflow_count": overflow_count,                # 溢位組數
        "overflow_pairs": overflow_pairs,                # 溢位組別清單
        "overflow_rate": round(overflow_count / pair_total * 100, 2) if pair_total else 0.0,  # 溢位率 %
        "safe_rate": round(safe_pairs / pair_total * 100, 2) if pair_total else 0.0,            # 安全率 %
        "has_overflow": overflow_count > 0,              # 是否有溢位
        "all_safe": overflow_count == 0,                 # 是否全部安全
        "header_bits_compressed": header_example,        # 壓縮封包標頭 bits
        "max_payload_bits": max(0, safe_pairs - header_example),  # 最大可藏秘密 bits
        "verdict": verdict,                              # 文字結論
        "overflow_details": details,                     # 溢位細節（最多 100 筆）
        "overflow_details_limit": 100,                   # 細節顯示上限
        "overflow_details_truncated": overflow_count > 100,  # 是否截斷細節
    }


def _flat_index_to_hwc(flat_i: int, h: int, w: int) -> tuple[int, int, str]:
    n = h * w
    if flat_i < n:
        ch = "B"
        idx = flat_i
    elif flat_i < 2 * n:
        ch = "G"
        idx = flat_i - n
    else:
        ch = "R"
        idx = flat_i - 2 * n
    row, col = divmod(idx, w)
    return row, col, ch  # 彩色拉平後像素對應的列、行、通道(B/G/R)


def _color_pair_position(pair_idx: int, matrix_cols: int, h: int, w: int) -> dict:
    i0, i1 = pair_idx * 2, pair_idx * 2 + 1
    r0, c0, ch0 = _flat_index_to_hwc(i0, h, w)
    r1, c1, ch1 = _flat_index_to_hwc(i1, h, w)
    return {
        "row_a": r0, "col_a": c0, "ch_a": ch0,       # 像素 A 的列/行/通道(BGR)
        "row_b": r1, "col_b": c1, "ch_b": ch1,       # 像素 B 的列/行/通道(BGR)
        "matrix_cols": matrix_cols,                   # DE 矩陣欄數
    }


def build_overflow_overlay(img: np.ndarray) -> np.ndarray:
    matrix, layout = image_to_de_matrix(img)
    matrix, rows, cols = _ensure_even_matrix(matrix)
    _, pairs = _flatten_pairs(matrix)
    _, overflow_pairs = build_loc_map(pairs)

    if layout["mode"] == "color":
        h, w = layout["orig_h"], layout["orig_w"]
        src = img[:, :, :3].copy()
        if src.shape[0] > h:
            src = src[:h]
        if src.shape[1] > w:
            src = src[:, :w]
        overlay = src.astype(np.uint8)
        for pair_num in overflow_pairs:
            pair_idx = pair_num - 1
            for j in range(2):
                flat_i = pair_idx * 2 + j
                r, c, ch = _flat_index_to_hwc(flat_i, h, w)
                ch_i = {"B": 0, "G": 1, "R": 2}[ch]
                overlay[r, c, ch_i] = (0, 0, 255)
        return overlay

    base = np.array(matrix, dtype=np.uint8)
    overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    for pair_num in overflow_pairs:
        pair_idx = pair_num - 1
        for j in range(2):
            flat_i = pair_idx * 2 + j
            r, c = divmod(flat_i, cols)
            overlay[r, c] = (0, 0, 255)
    return overlay


# =====================================================
# 圖片 I/O
# =====================================================
def load_gray_image(path) -> np.ndarray:
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"找不到或無法讀取圖片：{path}")
    return img


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """從 bytes 解碼圖片（不寫入磁碟）。"""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("無法讀取圖片（格式不支援或檔案損毀）")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    return img


def load_image(path) -> np.ndarray:
    """讀取圖片（彩色保留 BGR，灰階為 2D）。"""
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"找不到或無法讀取圖片：{path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    return img


def save_gray_image(img: np.ndarray, path) -> None:
    cv2.imwrite(str(path), img.astype(np.uint8))


def save_image(img: np.ndarray, path) -> None:
    cv2.imwrite(str(path), img.astype(np.uint8))


def ndarray_to_data_url(img: np.ndarray) -> str:
    buf = io.BytesIO()
    from PIL import Image

    arr = img.astype(np.uint8)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(buf, format="PNG")
    else:
        Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# =====================================================
# 高階 API
# =====================================================
def _stego_detect_result(
    layout: dict,
    overflow_indices: list[int],
    secret_bytes: bytes,
    magic: int = PACKET_MAGIC,
) -> dict:
    preview = secret_bytes.decode("utf-8", errors="replace")
    if len(preview) > 60:
        preview = preview[:60] + "…"
    mode_label = "彩色 BGR" if layout["mode"] == "color" else "灰階"
    algo = get_stego_algorithm(magic) or {"id": "DE", "label": "DE", "version": "?"}
    return {
        "is_stego": True,
        "mode": layout["mode"],
        "mode_label": mode_label,
        "magic": magic,
        "algorithm": algo["id"],
        "algorithm_label": algo["label"],
        "packet_version": algo.get("version", ""),
        "overflow_count": len(overflow_indices),
        "data_preview": preview,
        "message": (
            f"此圖已是 {algo['label']} 隱寫圖（{mode_label}，magic {magic:#06x}，"
            f"溢位 n={len(overflow_indices)}），"
            "不能再次嵌入。請改用原始封面，或到②解碼專區提取 DATA。"
        ),
    }


def detect_de_stego(img: np.ndarray) -> dict:
    """上傳快速檢測：前 16-bit magic 0xDE01 + 尾端 32-bit CRC32；僅 magic 則保守阻擋。"""
    try:
        matrix, layout = image_to_de_matrix(img)
        parsed = probe_de_packet(matrix)
        if parsed is not None:
            overflow_indices, _, secret_bytes, _filename = parsed
            return _stego_detect_result(layout, overflow_indices, secret_bytes)
        if peek_de_magic(matrix):
            mode_label = "彩色 BGR" if layout["mode"] == "color" else "灰階"
            algo = STEGO_MAGIC[PACKET_MAGIC]
            return {
                "is_stego": True,
                "mode": layout["mode"],
                "mode_label": mode_label,
                "magic": PACKET_MAGIC,
                "algorithm": algo["id"],
                "algorithm_label": algo["label"],
                "overflow_count": None,
                "data_preview": "",
                "message": (
                    f"此圖疑似已是 {algo['label']} 隱寫圖（{mode_label}，magic {PACKET_MAGIC:#06x}），"
                    "不能再次嵌入。請改用原始封面，或到②解碼專區提取 DATA。"
                ),
            }
        return {"is_stego": False, "message": ""}
    except Exception:
        return {"is_stego": False, "message": ""}


def de_embed_image(
    img: np.ndarray,
    secret: str | bytes,
    filename: str = "secret.txt",
    raise_on_overflow: bool = True,
):
    stego_check = detect_de_stego(img)
    if stego_check["is_stego"]:
        raise ValueError(stego_check["message"])
    matrix, layout = image_to_de_matrix(img)
    data_bytes = secret if isinstance(secret, bytes) else secret.encode("utf-8")
    new_matrix, loc_map, logs, meta = embed_DE_core(
        matrix, data_bytes, filename=filename, raise_on_overflow=raise_on_overflow
    )
    stego = de_matrix_to_image(new_matrix, layout)
    cover_ref = de_matrix_to_image(matrix, layout)
    meta["shape"] = list(img.shape)          # 原始輸入圖尺寸
    meta["used_shape"] = list(stego.shape)    # 輸出隱寫圖尺寸
    meta["mode"] = layout["mode"]             # gray 或 color
    if layout["mode"] == "color":
        logs.insert(0, f"彩色 BGR 拉平：{layout['orig_h']}×{layout['orig_w']} → 3 通道串接，像素對 ×3")
    # stego: 隱寫圖 | meta: 統計 | logs: 日誌 | cover_ref: 裁剪後封面（供比對還原）
    return stego, meta, logs, cover_ref


def de_extract_image(stego: np.ndarray):
    import base64

    matrix, layout = image_to_de_matrix(stego)
    secret_bytes, loc_map, overflow_indices, bits_embedded, filename = extract_DE_core(
        matrix
    )
    payload_info = classify_extracted_payload(secret_bytes, filename)
    payload_info["data_base64"] = base64.b64encode(secret_bytes).decode("ascii")
    mode_label = "彩色 BGR" if layout["mode"] == "color" else "灰階"
    preview = payload_info["text"] if payload_info["viewable_text"] else (
        f"{payload_info['filename']}（{payload_info['size']} bytes）"
    )
    logs = [
        f"模式：{mode_label}",
        f"溢位點 n = {len(overflow_indices)}（封包索引，CRC 驗證通過）",
        f"檔案名稱：{payload_info['filename']}",
        f"提取的 DATA：{preview[:200]}{'…' if len(preview) > 200 else ''}",
        "CRC32 校驗：通過",
    ]
    restored_matrix = restore_DE_core(
        matrix, overflow_indices, bits_embedded, filename
    )
    restored = de_matrix_to_image(restored_matrix, layout)
    return restored, payload_info, overflow_indices, logs


def embed_DE_file(input_path, output_img_path, secret_text):
    img = load_image(input_path)
    print(f"讀取圖片：{input_path}，尺寸 = {img.shape}")
    stego, meta, logs, _ = de_embed_image(img, secret_text)
    for line in logs:
        print(line)
    save_image(stego, output_img_path)
    print(f"嵌入後圖片已儲存：{output_img_path}")
    return meta


def extract_DE_file(input_img_path):
    img = load_image(input_img_path)
    print(f"讀取圖片：{input_img_path}，尺寸 = {img.shape}")
    _, payload_info, overflow_indices, logs = de_extract_image(img)
    for line in logs:
        print(line)
    return payload_info.get("text") or payload_info["filename"]


if __name__ == "__main__":
    import os

    test_img_path = "test_input.png"
    output_img_path = "test_output.png"

    if not os.path.exists(test_img_path):
        print("產生測試圖片 test_input.png（512x512 灰階漸層）")
        test_img = np.arange(0, 512 * 512, dtype=np.uint8).reshape(512, 512) // 2
        cv2.imwrite(test_img_path, test_img)

    embed_DE_file(test_img_path, output_img_path, "Hello World! 你好")
    got = extract_DE_file(output_img_path)
    print("roundtrip:", got)
