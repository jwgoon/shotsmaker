# 🎬 shotsmaker

**자동으로 쇼츠(Shorts) 영상을 생성하는 Python 기반 프로젝트**

---

## 🧠 개요
`shotsmaker`는 이미지, 텍스트, TTS(음성 합성)를 이용해  
자동으로 짧은 영상 콘텐츠(YouTube Shorts, Reels 등)를 만들어주는 프로젝트입니다.

---

## ✨ 주요 기능
- 🖼️ `inputs/` 폴더의 이미지를 자동으로 영상화 (Ken Burns 줌 효과)
- 🔊 텍스트를 기반으로 한 TTS 음성 생성 (한국어 지원)
- 🧩 이미지 + 오디오 + 자막을 자동 합성하여 영상 출력
- ⚙️ CLI(Command Line Interface) 기반으로 간편한 실행

---

## 🚀 사용 방법
```bash
# 실행 예시
python main.py --in inputs --out outputs --tts_lang ko --duration 35