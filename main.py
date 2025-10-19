import os, re, argparse, tempfile, math, random
from moviepy.video.fx import all as vfx # 추가 코드
from pathlib import Path

# === Config ===
USE_OFFLINE_TTS = False  # 오프라인 TTS로 바꾸려면 True로 바꾸고 pyttsx3 설치
FONT_SIZE = 54
CAPTION_MARGIN = 60
CAPTION_COLOR = 'white'
CAPTION_STROKE_COLOR = 'black'
CAPTION_STROKE_WIDTH = 2
RESOLUTION = (1080, 1920)  # 9:16
FPS = 30
BGM_DB = -18  # 배경음 볼륨(dB) 대략적 감쇠

# === Imports (lazy) ===
from moviepy.editor import (
    AudioFileClip, ImageClip, TextClip, CompositeAudioClip,
    CompositeVideoClip, concatenate_videoclips, ColorClip, afx
)
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np

def read_text(p: Path) -> str:
    return p.read_text(encoding='utf-8').strip()

def split_sentences_ko(text: str):
    # 마침표/물음표/느낌표 기준 단순 분리
    parts = re.split(r'(?<=[\.!\?…]|[다요죠고임음읍니까]|\))\s+', text)
    parts = [s.strip() for s in parts if s.strip()]
    return parts

def tts_gtts(text: str, lang: str, outfile: Path):
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang)
    tts.save(str(outfile))
    return outfile

def tts_offline_pyttsx3(text: str, outfile: Path):
    # 오프라인: 기기 TTS 음색/속도 품질은 OS마다 달라짐
    import pyttsx3
    engine = pyttsx3.init()
    # 속도/톤 약간 조정 가능
    # engine.setProperty('rate', 180)  # 말속도
    # engine.setProperty('volume', 0.9)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    engine.save_to_file(text, tmp_name)
    engine.runAndWait()
    # WAV -> MP3 변환은 생략 가능(moviepy가 wav도 읽음)
    os.replace(tmp_name, outfile)
    return outfile

def softwrap(text: str, width=18):
    # 아주 단순 줄바꿈(자막 너무 길지 않게)
    words = re.split(r'(\s+)', text)
    lines, line = [], ""
    for w in words:
        if len(line + w) > width and line:
            lines.append(line.strip())
            line = w.strip()
        else:
            line += w
    if line.strip():
        lines.append(line.strip())
    return "\n".join(lines)

def make_caption_clip(sentence, duration, fontsize=FONT_SIZE):
    # 자막 TextClip 생성
    txt = softwrap(sentence, width=20)
    tc = TextClip(
        txt,
        fontsize=fontsize,
        color=CAPTION_COLOR,
        font="Arial-Unicode-MS",  # 시스템에 따라 변경 필요(한글 지원 폰트)
        stroke_color=CAPTION_STROKE_COLOR,
        stroke_width=CAPTION_STROKE_WIDTH,
        method='caption',
        size=(RESOLUTION[0]-100, None)
    )
    # 화면 하단에 배치
    def pos(t):
        return ('center', RESOLUTION[1] - CAPTION_MARGIN - tc.h/2)
    return tc.set_position(pos).set_duration(duration)

'''
def ken_burns_for_image(img_path: Path, segment_duration: float):
    # 이미지에 간단한 패닝/줌 적용
    clip = ImageClip(str(img_path)).resize(height=RESOLUTION[1])
    # 세로 맞추고 가로 비율에 따라 좌우 크롭
    if clip.w < RESOLUTION[0]:
        # 세로 기준으로 늘렸더니 가로가 부족하면 가로 채우기
        clip = clip.resize(width=RESOLUTION[0])
    clip = clip.set_duration(segment_duration)

    # 시작/끝 스케일, x이동 랜덤
    start_scale = random.uniform(1.02, 1.08)
    end_scale   = random.uniform(1.10, 1.18)
    start_x     = random.uniform(-40, 40)
    end_x       = random.uniform(-80, 80)

    def make_frame(t):
        prog = t / max(segment_duration, 0.0001)
        scale = start_scale + (end_scale - start_scale) * prog
        x = start_x + (end_x - start_x) * prog
        frame = clip.get_frame(t)
        frame_clip = ImageClip(frame).resize(scale)
        # 중앙 정렬 + x 오프셋
        x_center = (RESOLUTION[0] - frame_clip.w) / 2 + x
        y_center = (RESOLUTION[1] - frame_clip.h) / 2
        bg = ColorClip(RESOLUTION, color=(0,0,0)).set_duration(clip.duration)
        return CompositeVideoClip(
            [bg, frame_clip.set_position((x_center, y_center))]
        ).get_frame(0)

    return clip.fl(make_frame, apply_to=[])
'''

def ken_burns_for_image(img_path, segment_duration: float):
    # 기본 이미지 클립
    base = ImageClip(str(img_path)).set_duration(segment_duration)

    # 화면(1080x1920) 꽉 채우기 위한 최소 스케일
    w0, h0 = base.size
    sx = RESOLUTION[0] / w0
    sy = RESOLUTION[1] / h0
    fill = max(sx, sy)

    # 랜덤 줌 범위 (부드러운 Ken Burns)
    start_scale = random.uniform(1.02, 1.08)
    end_scale   = random.uniform(1.10, 1.18)

    # 시간 t에 따른 스케일
    def scale_at(t):
        p = 0 if segment_duration <= 0 else (t / segment_duration)
        return fill * (start_scale + (end_scale - start_scale) * p)

    # 시간에 따라 크기 변화
    zoomed = base.resize(lambda t: scale_at(t))

    # 검정 배경 위에 중앙 정렬로 합성 (크롭 대신)
    bg = ColorClip(RESOLUTION, color=(0, 0, 0)).set_duration(segment_duration)
    comp = CompositeVideoClip(
        [bg, zoomed.set_position('center')],
        size=RESOLUTION
    ).set_duration(segment_duration)

    return comp

def mix_bgm(narration: AudioFileClip, bgm_path: Path = None):
    if not bgm_path or not bgm_path.exists():
        return narration
    bgm = AudioFileClip(str(bgm_path)).volumex(1.0)
    # 길이 맞추고 볼륨 감쇠
    bgm = afx.audio_loop(bgm, duration=narration.duration).volumex(10 ** (BGM_DB / 20.0))
    return CompositeAudioClip([bgm, narration]).set_duration(narration.duration)

def estimate_read_speed(text: str):
    # 한국어 대략 6~9글자/초 정도 읽는다고 가정, 평균치로 대충 길이 산정
    chars = len(re.sub(r'\s+', '', text))
    sec = max(3, chars / 7.5)
    return sec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_dir', required=True, help='inputs 폴더 경로')
    ap.add_argument('--out', dest='out_dir', required=True, help='outputs 폴더 경로')
    ap.add_argument('--tts_lang', default='ko', help='gTTS 언어코드 (ko, en 등)')
    ap.add_argument('--duration', type=int, default=45, help='목표 영상 길이(초)')
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    title = read_text(in_dir/'title.txt') if (in_dir/'title.txt').exists() else "Untitled Shorts"
    script = read_text(in_dir/'script.txt')
    sentences = split_sentences_ko(script)

    # 문장별 길이 대략 분배 (총합이 목표 duration 근처가 되도록)
    ests = [estimate_read_speed(s) for s in sentences]
    scale = args.duration / max(sum(ests), 1e-6)
    seg_durs = [max(2.0, d * scale) for d in ests]  # 최소 2초

    # TTS 생성
    tts_file = out_dir / 'narration.mp3'
    if USE_OFFLINE_TTS:
        tts_offline_pyttsx3(script, tts_file.with_suffix('.wav'))
        n_clip = AudioFileClip(str(tts_file.with_suffix('.wav')))
    else:
        tts_gtts(script, args.tts_lang, tts_file)
        n_clip = AudioFileClip(str(tts_file))

    # 이미지 로드(없으면 검정 배경)
    img_dir = in_dir/'images'
    images = sorted([p for p in img_dir.glob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png']])
    if not images:
        images = None

    # 문장별 비디오 조각 만들기
    clips = []
    img_idx = 0
    for sent, dur in zip(sentences, seg_durs):
        if images:
            img_path = images[img_idx % len(images)]
            v = ken_burns_for_image(img_path, dur)
            img_idx += 1
        else:
            v = ColorClip(RESOLUTION, color=(0,0,0), duration=dur)

        cap = make_caption_clip(sent, dur)
        clips.append(CompositeVideoClip([v, cap]).set_duration(dur))

    video = concatenate_videoclips(clips, method='compose')
    # 오디오: 내레이션 길이에 맞춰 잘라내기(문장 타이밍 정교화는 심플 버전)
    # 심플하게 전체 스크립트 TTS를 통짜로 쓰되, 영상 길이에 맞춤
    total_vdur = video.duration
    n_final = n_clip
    if n_clip.duration > total_vdur:
        n_final = n_clip.subclip(0, total_vdur)
    elif n_clip.duration < total_vdur:
        # 끝부분 약간의 무음 패드
        pad = total_vdur - n_clip.duration
        silence = AudioArrayClip(np.zeros((int(pad*44100), 2)), fps=44100)
        n_final = concatenate_videoclips([], method="compose")  # dummy
        n_final = CompositeAudioClip([AudioFileClip(str(tts_file)) , silence.set_start(n_clip.duration)]).set_duration(total_vdur)

    # BGM 믹스
    bgm_path = in_dir/'bgm.mp3'
    final_audio = mix_bgm(n_final, bgm_path if bgm_path.exists() else None)

    final = video.set_audio(final_audio).set_fps(FPS)
    safe_title = re.sub(r'[^0-9a-zA-Z가-힣_\-]+', '_', title)[:60]
    outfile = out_dir / f"{safe_title}_shorts.mp4"

    final.write_videofile(
        str(outfile),
        fps=FPS,
        codec='libx264',
        audio_codec='aac',
        bitrate='5000k',
        threads=4,
        preset='medium'
    )
    print(f"✅ Saved: {outfile}")

if __name__ == "__main__":
    main()
