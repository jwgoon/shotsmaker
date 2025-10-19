#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shotsmaker main.py (single-file, stable)
- inputs/ 내 이미지들을 이어붙여 세로형(1080x1920) 쇼츠 영상 생성
- Ken Burns(느린 줌) 효과
- 자막: script.txt(줄 단위)를 자동 읽어 문장별 자막 구성 (없으면 --caption 한 줄 사용)
- 오디오: --audio 파일 사용, 없으면 script/caption으로 gTTS 자동 생성(옵션)
- QuickTime/웹 호환: yuv420p + +faststart
- MoviePy 1.0.3/Pillow 9.5.0 호환 (TextClip 인자명: txt)

예시:
    python main.py --in inputs --out outputs --duration 5
    python main.py --in inputs --out outputs --duration 5 --caption "비행기 창문 아래 구멍의 비밀"
    python main.py --in inputs --out outputs --duration 5 --script inputs/script.txt
    python main.py --in inputs --out outputs --duration 5 --script inputs/script.txt --tts_lang ko
"""

from __future__ import annotations

import os
import sys
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional

# MoviePy
from moviepy.editor import (
    ImageClip,
    ColorClip,
    CompositeVideoClip,
    TextClip,
    concatenate_videoclips,
    AudioFileClip,
)

# -----------------------------
# macOS 편의: 외부 도구 경로 힌트
# -----------------------------
if sys.platform == "darwin":
    os.environ.setdefault("IMAGEMAGICK_BINARY", "/opt/homebrew/bin/magick")
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "/opt/homebrew/bin/ffmpeg")

# -----------------------------
# 기본 설정
# -----------------------------
RESOLUTION: Tuple[int, int] = (1080, 1920)  # 세로 캔버스
FPS: int = 30
BG_COLOR: Tuple[int, int, int] = (0, 0, 0)
KB_ZOOM_RANGE: Tuple[float, float] = (1.06, 1.16)  # 6~16% 줌 범위

# macOS / Windows 한글 폰트 후보
MAC_FONT_CANDIDATES: List[str] = [
    "/Library/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
]
WIN_FONT_CANDIDATES: List[str] = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
]


def pick_font_path() -> Optional[str]:
    """사용 가능한 한글 폰트를 찾아 경로 반환. 없으면 None."""
    candidates = MAC_FONT_CANDIDATES if sys.platform == "darwin" else WIN_FONT_CANDIDATES
    for p in candidates:
        if Path(p).exists():
            return p
    return None


# -----------------------------
# IO 유틸
# -----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def scan_images(in_dir: Path) -> List[Path]:
    """입력 폴더(하위 포함)에서 이미지 파일 재귀 검색."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}
    files = [p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def load_script_lines(in_dir: Path, script_arg: Optional[str]) -> List[str]:
    """
    대본 파일을 줄 단위로 읽어 리스트 반환.
    우선순위: --script 경로 > {in_dir}/script.txt
    빈 줄 제거, 양끝 공백 제거.
    """
    target: Optional[Path] = None
    if script_arg:
        p = Path(script_arg)
        if p.exists():
            target = p
    else:
        p = in_dir / "script.txt"
        if p.exists():
            target = p

    if not target:
        return []

    lines: List[str] = []
    with open(target, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


# -----------------------------
# 영상 효과
# -----------------------------
def ken_burns_for_image(
    img_path: Path,
    segment_duration: float,
    resolution: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    zoom_range: Tuple[float, float] = KB_ZOOM_RANGE,
) -> CompositeVideoClip:
    """이미지 1장을 segment_duration 길이의 줌/팬 클립으로 변환."""
    base = ImageClip(str(img_path)).set_duration(max(0.01, float(segment_duration)))
    w0, h0 = base.size
    W, H = resolution

    # 캔버스 채우기 위한 최소 스케일
    sx, sy = W / w0, H / h0
    fill_scale = max(sx, sy)

    zmin, zmax = zoom_range
    start_scale = random.uniform(zmin, (zmin + zmax) / 2.0)
    end_scale = random.uniform((zmin + zmax) / 2.0, zmax)

    def scale_at(t: float) -> float:
        p = 0.0 if segment_duration <= 0 else min(max(t / segment_duration, 0.0), 1.0)
        return fill_scale * (start_scale + (end_scale - start_scale) * p)

    zoomed = base.resize(lambda t: scale_at(t))

    # 배경 + 중앙 정렬
    bg = ColorClip(resolution, color=bg_color).set_duration(segment_duration)
    clip = CompositeVideoClip([bg, zoomed.set_position("center")], size=resolution)
    return clip.set_duration(segment_duration)


# -----------------------------
# 자막
# -----------------------------
def make_caption_clip(
    text: str,
    duration: float,
    resolution: Tuple[int, int],
    font_path: Optional[str],
    fontsize: int = 56,
    color: Tuple[int, int, int] = (255, 255, 255),
    stroke_width: int = 2,
    stroke_color: Tuple[int, int, int] = (0, 0, 0),
    margin_bottom: int = 120,
) -> TextClip:
    """한 줄 자막 클립 (moviepy 1.0.3 → TextClip 인자명은 txt)."""
    color_str = f"rgb({color[0]},{color[1]},{color[2]})"
    stroke_str = f"rgb({stroke_color[0]},{stroke_color[1]},{stroke_color[2]})"

    base_kwargs = dict(
        txt=text,                 # 핵심: 'text'가 아니라 'txt'
        fontsize=fontsize,
        color=color_str,
        stroke_color=stroke_str,
        stroke_width=stroke_width,
    )
    if font_path:
        base_kwargs["font"] = font_path

    # caption 방식(자동 줄바꿈) → 실패 시 label 폴백
    try:
        tc = TextClip(
            **base_kwargs,
            method="caption",
            size=(resolution[0] - 100, None),
        ).set_duration(max(0.01, float(duration)))
    except Exception:
        safe = text
        if len(safe) > 40:
            parts = [safe[i:i + 40] for i in range(0, len(safe), 40)]
            safe = "\n".join(parts)
        fb = dict(base_kwargs)
        fb["txt"] = safe
        tc = TextClip(**fb, method="label").set_duration(max(0.01, float(duration)))

    return tc.set_position(("center", resolution[1] - margin_bottom))


def make_caption_sequence(
    lines: List[str],
    total_duration: float,
    resolution: Tuple[int, int],
    font_path: Optional[str],
) -> List[TextClip]:
    """
    대본 줄 별로 자막 클립을 만들고 total_duration 내에서 순차 배치.
    기본 시간: 글자수*0.06초(최소 1.2초) → 전체 길이에 맞춰 비율 스케일.
    """
    if not lines:
        return []

    raw_durs = [max(1.2, len(line) * 0.06) for line in lines]
    sum_raw = sum(raw_durs)
    factor = total_duration / sum_raw if sum_raw > 0 else 1.0
    durs = [max(0.8, rd * factor) for rd in raw_durs]

    # 누적 시작 시각
    starts, acc = [], 0.0
    for d in durs:
        starts.append(acc)
        acc += d

    # 마지막 조정
    if acc > 0 and abs(acc - total_duration) > 0.25:
        scale = total_duration / acc
        durs = [d * scale for d in durs]
        starts, acc = [], 0.0
        for d in durs:
            starts.append(acc)
            acc += d

    color_str = "rgb(255,255,255)"
    stroke_str = "rgb(0,0,0)"
    seq: List[TextClip] = []

    for line, st, du in zip(lines, starts, durs):
        kwargs = dict(
            txt=line,
            fontsize=56,
            color=color_str,
            stroke_color=stroke_str,
            stroke_width=2,
            method="caption",
            size=(resolution[0] - 100, None),
        )
        if font_path:
            kwargs["font"] = font_path

        try:
            tc = TextClip(**kwargs).set_duration(du)
        except Exception:
            safe = line
            if len(safe) > 40:
                parts = [safe[i:i + 40] for i in range(0, len(safe), 40)]
                safe = "\n".join(parts)
            kw = dict(kwargs)
            kw["txt"] = safe
            kw.pop("size", None)
            kw["method"] = "label"
            tc = TextClip(**kw).set_duration(du)

        tc = tc.set_position(("center", RESOLUTION[1] - 120)).set_start(st)
        seq.append(tc)

    return seq


# -----------------------------
# 최종 합성 / 출력
# -----------------------------
def build_final_video(
    images: List[Path],
    per_image_sec: float,
    caption_text: Optional[str],
    audio_path: Optional[Path],
    script_lines: Optional[List[str]] = None,
) -> CompositeVideoClip:
    """이미지 → 클립 → 연결, 자막/오디오 적용."""
    if not images:
        raise SystemExit("❌ inputs 폴더에 사용할 이미지가 없습니다.")

    clips = [
        ken_burns_for_image(
            img_path=img,
            segment_duration=per_image_sec,
            resolution=RESOLUTION,
            bg_color=BG_COLOR,
            zoom_range=KB_ZOOM_RANGE,
        )
        for img in images
    ]
    video = concatenate_videoclips(clips, method="compose")

    # 자막: script_lines 우선, 없으면 caption_text(한 줄)
    if script_lines and len(script_lines) > 0:
        font_path = pick_font_path()
        seq = make_caption_sequence(
            lines=script_lines,
            total_duration=video.duration,
            resolution=RESOLUTION,
            font_path=font_path,
        )
        if seq:
            video = CompositeVideoClip([video, *seq], size=RESOLUTION)
    elif caption_text:
        font_path = pick_font_path()
        cap = make_caption_clip(
            text=caption_text,
            duration=video.duration,
            resolution=RESOLUTION,
            font_path=font_path,
            fontsize=56,
            color=(255, 255, 255),
            stroke_width=2,
            stroke_color=(0, 0, 0),
            margin_bottom=120,
        )
        video = CompositeVideoClip([video, cap], size=RESOLUTION)

    # 오디오
    if audio_path and audio_path.exists():
        try:
            ac = AudioFileClip(str(audio_path))
            video = video.set_audio(ac)
        except Exception as e:
            print(f"⚠️ 오디오를 불러오지 못했습니다: {e}")

    return video.set_fps(FPS)


def write_video_safe(clip: CompositeVideoClip, out_path: Path) -> None:
    """재생 호환을 위한 안전한 인코딩 옵션 고정."""
    ensure_dir(out_path.parent)
    if not clip.duration or clip.duration <= 0:
        raise SystemExit("❌ 최종 클립 길이가 0초입니다. --duration(장당 길이)을 확인하세요.")

    print(f"\n➡️  최종 길이: {clip.duration:.2f}s, FPS: {FPS}, 출력: {out_path}\n")

    clip.write_videofile(
        str(out_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        bitrate="4000k",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        threads=max(1, (os.cpu_count() or 8) - 1),
        verbose=True,
        logger="bar",  # 진행바 표시 ("print"/None 도 가능)
    )


# -----------------------------
# CLI
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="inputs", help="입력 이미지 폴더 경로")
    ap.add_argument("--out", dest="out_dir", default="outputs", help="출력 폴더 경로")
    ap.add_argument("--tts-lang", "--tts_lang", dest="tts_lang", default="ko", help="TTS 언어 코드 (예: ko, en)")
    ap.add_argument("--duration", type=float, default=3.0, help="이미지 1장당 지속 시간(초)")
    ap.add_argument("--caption", type=str, default=None, help="(옵션) 전체 구간 한 줄 자막")
    ap.add_argument("--audio", type=str, default=None, help="(옵션) 오디오 파일 경로(mp3/wav)")
    ap.add_argument("--script", type=str, default=None, help="(옵션) 대본 파일 경로(미지정시 inputs/script.txt 시도)")
    return ap


def main() -> None:
    from gtts import gTTS
    import tempfile

    # 1) 인자
    ap = build_arg_parser()
    args = ap.parse_args()

    # 2) 경로
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_path = out_dir / "output.mp4"

    if not in_dir.exists():
        raise SystemExit(f"❌ 입력 폴더가 존재하지 않습니다: {in_dir}")

    # 3) 이미지 스캔
    images = scan_images(in_dir)
    if not images:
        raise SystemExit(f"❌ 이미지가 없습니다: {in_dir} (예: {in_dir}/image.png 또는 하위 폴더)")

    # 4) 스크립트/옵션
    per_image_sec = max(0.1, float(args.duration))
    caption_text = args.caption
    script_lines = load_script_lines(in_dir, args.script)
    audio_path = Path(args.audio) if args.audio else None

    # 5) 자동 TTS: 오디오 없고 스크립트 or 캡션 있으면 생성
    if audio_path is None and (script_lines or caption_text):
        try:
            tmpdir = Path(tempfile.gettempdir())
            tts_path = tmpdir / "shotsmaker_tts.mp3"
            tts_text = " ".join(script_lines) if script_lines else caption_text
            tts = gTTS(text=tts_text, lang=(args.tts_lang or "ko"))
            tts.save(str(tts_path))
            audio_path = tts_path
            print(f"🔊 TTS 생성 완료 → {tts_path}")
        except Exception as e:
            print(f"⚠️ TTS 생성 실패: {e} (무음으로 진행)")

    # 6) 합성 & 저장
    final_clip = build_final_video(
        images=images,
        per_image_sec=per_image_sec,
        caption_text=caption_text,
        audio_path=audio_path,
        script_lines=script_lines,
    )
    write_video_safe(final_clip, out_path)
    print("\n✅ 완료! 재생이 안 되면 VLC로도 테스트해 보세요 (QuickTime 문제일 수 있음).\n")


if __name__ == "__main__":
    main()
