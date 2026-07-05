"""DE 隱寫術 Flask 前後端（記憶體處理，不寫入磁碟）"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from DE import (
    analysis_from_embed_meta,
    analyze_capacity,
    analyze_secret_fit,
    append_index_bits_stats_record,
    de_embed_image,
    de_extract_image,
    detect_de_stego,
    is_color_image,
    load_image_from_bytes,
    ndarray_to_data_url,
    normalize_filename,
)

BASE_DIR = Path(__file__).resolve().parent

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
OVERFLOW_MAP_UI_LIMIT = 16
OVERFLOW_PAIRS_UI_LIMIT = 30
LOG_MAX_LINES = 10_000

app = Flask(__name__)
app.secret_key = "data-hiding-dev-key"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def allowed_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXT


def load_uploaded_image(file_storage):
    """從上傳串流讀圖，不儲存至磁碟。"""
    if not file_storage or not file_storage.filename:
        raise ValueError("請選擇或拖入圖片檔案")
    orig_name = secure_filename(Path(file_storage.filename).name) or "upload.png"
    if not allowed_image(orig_name):
        raise ValueError(f"不支援的格式，請用：{', '.join(sorted(ALLOWED_IMAGE_EXT))}")
    data = file_storage.read()
    if not data:
        raise ValueError("上傳的圖片為空")
    return load_image_from_bytes(data), orig_name


def read_secret_payload() -> tuple[bytes, str, str]:
    """從表單讀取 DATA：優先 secret_file，否則 secret 文字。回傳 (bytes, 顯示標籤, 完整檔名)。"""
    secret_file = request.files.get("secret_file")
    if secret_file and secret_file.filename:
        data = secret_file.read()
        if not data:
            raise ValueError("上傳的 DATA 檔案為空")
        name = secure_filename(Path(secret_file.filename).name) or "file"
        return data, f"[檔案] {name}", normalize_filename(name)
    text = (request.form.get("secret") or "").strip()
    if text:
        return text.encode("utf-8"), text, "secret.txt"
    return b"", "", "secret.txt"


def secret_display_text(secret_bytes: bytes, label: str) -> str:
    """嵌入/解碼結果顯示用：文字原樣，二進位顯示摘要。"""
    if label.startswith("[檔案]"):
        try:
            return secret_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return f"{label}（{len(secret_bytes)} bytes，二進位）"
    return label


def slim_analysis(analysis: dict) -> dict:
    """API 回傳用：截斷溢位清單，避免 JSON / 前端 DOM 過大。"""
    out = {k: v for k, v in analysis.items() if k != "overflow_pairs"}
    pairs = analysis.get("overflow_pairs") or []
    out["overflow_pairs_preview"] = pairs[:OVERFLOW_PAIRS_UI_LIMIT]
    full_map = analysis.get("overflow_embed_map") or []
    out["overflow_embed_map"] = full_map[:OVERFLOW_MAP_UI_LIMIT]
    if len(full_map) > OVERFLOW_MAP_UI_LIMIT:
        out["overflow_embed_map_total"] = len(full_map)
    return out


def format_overflow(overflow_pairs: list[int], limit: int = 30) -> str:
    if not overflow_pairs:
        return ""
    shown = overflow_pairs[:limit]
    suffix = f" …共 {len(overflow_pairs)} 組" if len(overflow_pairs) > limit else ""
    return f"第 {shown}{suffix} 組"


def cap_log_lines(lines: list[str], max_lines: int = LOG_MAX_LINES) -> str:
    """log 最多 max_lines 行，超出則截斷並附註。"""
    if not lines:
        return ""
    if len(lines) <= max_lines:
        return "\n".join(lines)
    omitted = len(lines) - max_lines
    return (
        "\n".join(lines[:max_lines])
        + f"\n…（已省略 {omitted} 行，最多顯示 {max_lines} 行，共 {len(lines)} 行）"
    )


def cap_multiline_text(text: str, max_lines: int = LOG_MAX_LINES) -> str:
    """多行文字（解碼結果等）最多 max_lines 行。"""
    if not text:
        return text
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    omitted = len(lines) - max_lines
    return (
        "\n".join(lines[:max_lines])
        + f"\n…（已省略 {omitted} 行，最多顯示 {max_lines} 行，共 {len(lines)} 行）"
    )


def image_stats_payload(analysis: dict) -> dict:
    return {
        "width": analysis.get("display_width", analysis.get("width")),
        "height": analysis.get("display_height", analysis.get("height")),
        "pair_total": analysis["pair_total"],
        "overflow_count": analysis["overflow_count"],
        "overflow_first": analysis.get("overflow_first"),
        "overflow_last": analysis.get("overflow_last"),
        "overflow_first_embed_bit": analysis.get("overflow_first_embed_bit"),
        "overflow_last_embed_bit": analysis.get("overflow_last_embed_bit"),
        "safe_pairs": analysis["safe_pairs"],
        "mode": analysis.get("mode", "gray"),
    }


def log_image_stats(label: str, name: str, stats: dict) -> None:
    logger.info(
        "%s %r 尺寸=%s×%s 像素對=%d 溢位 n=%d 安全=%d",
        label,
        name,
        stats["width"],
        stats["height"],
        stats["pair_total"],
        stats["overflow_count"],
        stats["safe_pairs"],
    )


def log_de_judge_timing(
    event: str,
    image_name: str,
    analysis: dict,
    *,
    server_ms: int | None = None,
) -> None:
    """印出 / 記錄 DE 差值溢位掃描（build_loc_map）與分析各階段耗時。"""
    t = analysis.get("timing_ms") or {}
    verify = t.get("verify_embed_ms")
    verify_s = f"{verify}ms" if verify is not None else "—"
    logger.info(
        "%s %r pairs=%s | DE差值判斷 loc_map=%dms"
        " (repeat=%s) | 矩陣=%dms 容量=%dms fit_extra=%dms verify=%s"
        " | analyze_total=%dms server=%sms | safe=%s overflow=%s",
        event,
        image_name,
        f"{analysis.get('pair_total', 0):,}",
        t.get("de_loc_map_ms", 0),
        t.get("de_loc_map_repeat_ms", "—"),
        t.get("matrix_ms", 0),
        t.get("capacity_rest_ms", 0),
        t.get("fit_extra_ms", 0),
        verify_s,
        t.get("analyze_total_ms", t.get("total_ms", 0)),
        server_ms if server_ms is not None else "—",
        analysis.get("safe_pairs"),
        analysis.get("overflow_count"),
    )


def record_index_bits_stats(
    event: str,
    image_name: str,
    profile: dict | None,
    *,
    secret_bytes: int = 0,
    embed_ok: bool | None = None,
    timing: dict | None = None,
) -> None:
    """追加 INDEX_BITS 分析數字到 data/image_index_bits_log.jsonl。"""
    if not profile and not timing:
        return
    try:
        record: dict = {
            "event": event,
            "image_name": image_name,
            "secret_bytes": secret_bytes,
            "embed_ok": embed_ok,
        }
        if profile:
            record["profile"] = profile
        if timing:
            record["timing"] = timing
        path = append_index_bits_stats_record(record)
        logger.info(
            "INDEX_BITS 紀錄 %s %r → %s timing=%s",
            event,
            image_name,
            path.name,
            timing,
        )
    except OSError as exc:
        logger.warning("INDEX_BITS 紀錄寫入失敗: %s", exc)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    t0 = time.perf_counter()
    try:
        cover_file = request.files.get("cover_file")
        img, name = load_uploaded_image(cover_file)
        stego_check = detect_de_stego(img)
        if stego_check["is_stego"]:
            cap = analyze_capacity(img)
            stats = image_stats_payload(cap)
            log_image_stats("分析", name, stats)
            elapsed = int((time.perf_counter() - t0) * 1000)
            log_de_judge_timing("analyze(stego)", name, cap, server_ms=elapsed)
            record_index_bits_stats(
                "analyze_stego",
                name,
                cap.get("index_bits_profile"),
                timing=cap.get("timing_ms"),
            )
            return jsonify({
                "ok": True,
                "already_stego": True,
                "stego_check": stego_check,
                "image_stats": stats,
                "cover_preview": ndarray_to_data_url(img),
                "message": stego_check["message"],
                "server_ms": elapsed,
                "timing": cap.get("timing_ms"),
            })

        secret_bytes, secret_label, secret_filename = read_secret_payload()
        result = slim_analysis(
            analyze_secret_fit(img, secret_bytes or "", filename=secret_filename)
        )
        stats = image_stats_payload(result)
        log_image_stats("分析", name, stats)
        record_index_bits_stats(
            "analyze",
            name,
            result.get("index_bits_profile"),
            secret_bytes=len(secret_bytes),
            timing=result.get("timing_ms"),
        )
        overflow_msg = format_overflow(result.get("overflow_pairs_preview", []))
        elapsed = int((time.perf_counter() - t0) * 1000)
        log_de_judge_timing("analyze", name, result, server_ms=elapsed)
        return jsonify({
            "ok": True,
            "already_stego": False,
            "stego_check": stego_check,
            "analysis": result | {"image_name": name},
            "image_stats": stats,
            "cover_preview": ndarray_to_data_url(img),
            "overflow_msg": overflow_msg,
            "server_ms": elapsed,
            "timing": result.get("timing_ms"),
            "message": (
                f"像素對 {result['pair_total']} 組，封包需 {result['required_bits']} bits"
                + (f"；溢位組別：{overflow_msg}" if overflow_msg else "")
            ),
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/embed", methods=["POST"])
def api_embed():
    t0 = time.perf_counter()
    try:
        logger.info("========== /api/embed 收到請求 ==========")

        cover_file = request.files.get("cover_file")
        secret_bytes, secret_label, secret_filename = read_secret_payload()
        logger.info(
            "cover_file=%r secret_len=%d secret_src=%r",
            cover_file.filename if cover_file else None,
            len(secret_bytes),
            secret_label[:80] if secret_label else "",
        )

        if not secret_bytes:
            return jsonify({"ok": False, "error": "請輸入 DATA 或上傳檔案"}), 400

        img, name = load_uploaded_image(cover_file)
        stego_check = detect_de_stego(img)
        if stego_check["is_stego"]:
            cap = slim_analysis(analyze_capacity(img))
            stats = image_stats_payload(cap)
            log_image_stats("拒絕嵌入", name, stats)
            logger.warning("拒絕二次嵌入：%s", stego_check["message"])
            return jsonify({
                "ok": False,
                "already_stego": True,
                "stego_check": stego_check,
                "image_stats": stats,
                "cover_preview": ndarray_to_data_url(img),
                "error": stego_check["message"],
            }), 400

        logger.info(
            "圖片已讀取: name=%r shape=%s mode=%s",
            name,
            img.shape,
            "color" if is_color_image(img) else "gray",
        )

        try:
            t_embed = time.perf_counter()
            stego_img, meta, embed_logs, cover_ref = de_embed_image(
                img, secret_bytes, filename=secret_filename
            )
            embed_ms = int((time.perf_counter() - t_embed) * 1000)
        except ValueError as exc:
            analysis = slim_analysis(
                analyze_secret_fit(img, secret_bytes, filename=secret_filename)
            )
            stats = image_stats_payload(analysis)
            overflow_msg = format_overflow(analysis.get("overflow_pairs_preview", []))
            log_de_judge_timing("embed_fail", name, analysis)
            return jsonify({
                "ok": False,
                "error": str(exc),
                "analysis": analysis | {"image_name": name},
                "overflow_msg": overflow_msg,
                "image_stats": stats,
            }), 400

        analysis = slim_analysis(analysis_from_embed_meta(img, meta, secret_label))
        stats = image_stats_payload(analysis)
        log_image_stats("嵌入", name, stats)
        verify_error = None
        decoded = ""
        decoded_bytes = b""
        decode_logs: list[str] = []
        restored_img = stego_img
        verify_ms = 0
        try:
            t_verify = time.perf_counter()
            restored_img, payload_info, overflow_indices, decode_logs = de_extract_image(
                stego_img
            )
            verify_ms = int((time.perf_counter() - t_verify) * 1000)
            import base64

            decoded_bytes = base64.b64decode(payload_info["data_base64"])
            decoded = cap_multiline_text(secret_display_text(decoded_bytes, secret_label))
        except Exception as verify_exc:
            verify_error = str(verify_exc)
            logger.warning("嵌入後自動驗證失敗: %s", verify_exc)

        success = (
            verify_error is None
            and decoded_bytes == secret_bytes
            and bool((restored_img == cover_ref).all())
        )
        server_ms = int((time.perf_counter() - t0) * 1000)
        timing = {
            "embed_ms": embed_ms,
            "verify_ms": verify_ms,
            "server_ms": server_ms,
        }
        record_index_bits_stats(
            "embed",
            name,
            analysis.get("index_bits_profile"),
            secret_bytes=len(secret_bytes),
            embed_ok=success,
            timing=timing,
        )

        logger.info(
            "嵌入結果: bits=%d overflow_n=%d success=%s",
            meta["bits_embedded"],
            meta["overflow_count"],
            success,
        )
        log_image_stats("嵌入完成", name, stats)
        logger.info("========== /api/embed 完成 ==========")

        stego_preview = ndarray_to_data_url(stego_img)
        download_name = f"stego_DE_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        overflow_msg = format_overflow(meta["overflow_pairs"])
        logger.info(
            "api/embed 完成 %s embed_ms=%d verify_ms=%d server_ms=%d",
            name,
            embed_ms,
            verify_ms,
            server_ms,
        )

        return jsonify({
            "ok": True,
            "analysis": analysis | {"image_name": name},
            "image_stats": stats,
            "server_ms": server_ms,
            "timing": timing,
            "embed": {
                "method": meta.get("method", "DE_V4"),
                "filename": meta.get("filename", secret_filename),
                "mode": meta.get("mode", "gray"),
                "secret": secret_display_text(secret_bytes, secret_label),
                "secret_source": secret_label,
                "decoded": decoded,
                "success": success,
                "verify_error": verify_error,
                "bits": meta["bits_embedded"],
                "weak_skip_count": meta.get("weak_skip_count", 0),
                "overflow_count": meta["overflow_count"],
                "overflow_pairs": meta["overflow_pairs"][:OVERFLOW_PAIRS_UI_LIMIT],
                "overflow_msg": overflow_msg,
                "image_name": name,
                "log": cap_log_lines(embed_logs),
                "decode_log": cap_multiline_text(verify_error or cap_log_lines(decode_logs)),
                "stego_download_name": download_name,
                "timing": timing,
            },
            "cover_preview": ndarray_to_data_url(img),
            "stego_preview": stego_preview,
            "restored_preview": ndarray_to_data_url(restored_img),
            "message": (
                "嵌入完成"
                + (f"，自動驗證{'成功' if success else '失敗'}" if not verify_error else "，但自動驗證失敗")
                + (f"：{verify_error}" if verify_error else "")
            ),
        })
    except Exception as exc:
        logger.exception("/api/embed 失敗: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/decode", methods=["POST"])
def api_decode():
    t0 = time.perf_counter()
    try:
        stego_file = request.files.get("stego_file")
        if not stego_file or not stego_file.filename:
            return jsonify({"ok": False, "error": "請上傳隱寫圖"}), 400

        stego, name = load_uploaded_image(stego_file)
        t_analyze = time.perf_counter()
        cap = analyze_capacity(stego)
        analyze_ms = int((time.perf_counter() - t_analyze) * 1000)
        stats = image_stats_payload(cap)
        log_image_stats("解碼", name, stats)
        t_extract = time.perf_counter()
        result_cover, payload_info, overflow_indices, logs = de_extract_image(stego)
        extract_ms = int((time.perf_counter() - t_extract) * 1000)
        server_ms = int((time.perf_counter() - t0) * 1000)
        timing = {
            **(cap.get("timing_ms") or {}),
            "analyze_ms": analyze_ms,
            "extract_ms": extract_ms,
            "server_ms": server_ms,
        }
        record_index_bits_stats(
            "decode",
            name,
            cap.get("index_bits_profile"),
            secret_bytes=payload_info.get("size", 0),
            timing=timing,
        )
        log_de_judge_timing("decode", name, cap, server_ms=server_ms)
        mode = "color" if is_color_image(stego) else "gray"
        secret_display = (
            cap_multiline_text(payload_info["text"])
            if payload_info.get("viewable_text")
            else f"{payload_info['filename']}（{payload_info['size']} bytes）"
        )
        text_capped = (
            cap_multiline_text(payload_info.get("text", ""))
            if payload_info.get("viewable_text")
            else payload_info.get("text", "")
        )
        logger.info(
            "api/decode 完成 %s extract_ms=%d analyze_ms=%d server_ms=%d",
            name,
            extract_ms,
            analyze_ms,
            server_ms,
        )
        return jsonify({
            "ok": True,
            "extract": {
                "method": "DE_V4",
                "mode": mode,
                "secret": secret_display,
                "extension": payload_info["extension"],
                "filename": payload_info["filename"],
                "mime": payload_info["mime"],
                "size": payload_info["size"],
                "viewable_text": payload_info["viewable_text"],
                "viewable_image": payload_info["viewable_image"],
                "text": text_capped,
                "data_base64": payload_info["data_base64"],
                "overflow_count": len(overflow_indices),
                "overflow_pairs": overflow_indices[:30],
                "log": cap_log_lines(logs),
                "width": stego.shape[1],
                "height": stego.shape[0],
                "timing": timing,
            },
            "image_stats": stats,
            "stego_preview": ndarray_to_data_url(stego),
            "cover_preview": ndarray_to_data_url(result_cover),
            "server_ms": server_ms,
            "timing": timing,
            "message": "解碼成功！",
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
