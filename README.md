# 🤟 수어 제스처 기반 생성형 AI 인터페이스
### 실시간 수어 및 제스처 인식을 통해 길찾기, 인프라 검색, 장소 정보 검색, 음성 변환, SOS 구조 요청의 기능을 통합하여 언어 및 청각장애가 있는 운전자의 편의성과 안전한 주행을 지원하는 스마트 보조 시스템이다. 

<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/jmin-code/2025ESWContest_mobility_6051/raw/main/Readme_img/IMG_4256.jpeg" width="250px"/><br>
      <sub>수어 및 제스처 입력</sub>
    </td>
    <td align="center">
      <img src="https://github.com/jmin-code/2025ESWContest_mobility_6051/raw/main/Readme_img/display.png" width="250px"/><br>
      <sub>디스플레이 출력</sub>
    </td>
    <td align="center">
      <img src="https://github.com/jmin-code/2025ESWContest_mobility_6051/raw/main/Readme_img/hud.png" width="250px"/><br>
      <sub>HUD 표시</sub>
    </td>
  </tr>
</table>

---

## 📖 작품 개요
최근 출시되는 차량들은 AI 음성 인식 서비스를 탑재하여 운전자가 목소리만으로 내비게이션을 조작하거나 공조 시스템을 제어하는 등 높은 편의성을 제공한다.

하지만, 이러한 음성 기반 인터페이스는 **언어나 청각 장애가있는 운전자**에게는 사실상 사용이 불가능한 서비스이다. 
이는 기술 발전의 혜택에서 특정 사용자층이 소외되는 '정보 격차' 문제를 야기하며, 나아가 **기술적 차별**로 이어질 수도 있다.

본 시스템은 언어/청각 장애인 운전자를 포함한 **모든 운전자에게 동등한 편의성과 안전을 제공**하는 것을 목표로 하는 차량용 스마트 보조 시스템이다.

핵심 기능인 수어 인식 시스템은 운전자의 제스처를 인식하여 시스템과 상호작용할 수 있게 한다. 또한, 양방향 소통 (수어-음성, 음성-텍스트 변환)을 지원하여 의사소통의 장벽을 허물 수 있다.

이에 더해, GPS 기반의 스마트 내비게이션으로 주행 편의성을 극대화하고, 위급 상황 시 원터치로 작동하는 긴급 SOS 시스템을 탑재하여 운전자의 안전을 최우선으로 확보한다. 본 시스템은 기술의 혜택을 모두가 누릴 수 있도록 돕는 **포용적인 솔루션**을 지향하는 것이다.

---

## 🔌 H/W 구성 (Hardware)
![HW 구성](https://github.com/jmin-code/2025ESWContest_mobility_6051/blob/main/Readme_img/Screenshot%20from%202025-10-28%2001-29-23.png)
본 시스템을 구축하기 위해 사용된 주요 하드웨어 구성 요소는 다음과 같다.

+ 메인 프로세서: Raspberry Pi 5 (시스템의 전반적인 연산(딥러닝 모델 추론, UI 실행 등)을 담당하는 메인 보드)

+ 카메라: PiCam(적외선 카메라) & USB 웹카메라

+ 디스플레이(Main): 15인치 터치 스크린

+ 디스플레이(HUD): 7인치 스크린

+ GPS 모듈: Raspberry Pi Pico 활용

+ 스피커: 블루투스로 통신

---

## 💻 S/W 구성 (Software)
![SW 구성](https://github.com/jmin-code/2025ESWContest_mobility_6051/blob/main/Readme_img/Screenshot%20from%202025-10-28%2001-29-37.png)
본 시스템은 Raspberry Pi OS 환경에서 Python 3.11을 기반으로 구동된다. 주요 소프트웨어 스택은 다음과 같다.

+ GUI 프레임워크: PyQt5

  + 시스템의 전체 사용자 인터페이스(UI)를 구축하고 화면 간 상호작용을 관리.

+ AI / 머신러닝: MediaPipe, PyTorch, OpenCV

  + MediaPipe를 사용하여 카메라 영상에서 실시간으로 손의 랜드마크를 추출.

  + 추출된 데이터를 딥러닝 모델로 전달하여 수어 동작을 인식하고 텍스트로 변환.

+ 지도 / API: Kakao Maps API, QWebEngineView

  + QWebEngineView를 통해 HTML/JavaScript로 작성된 카카오맵을 UI에 통합하여 내비게이션 기능을 구현.

+ 음성 처리: gTTS (Google Text-to-Speech), pyaudio

  + 번역된 텍스트를 음성으로 변환(TTS)하는데 사용.

+ 하드웨어 통신: pyserial

  + GPS 모듈과의 시리얼 통신을 통해 실시간 위치 데이터를 수신.

---

## 🌊 시스템 동작 흐름
![Flowchart](https://github.com/jmin-code/2025ESWContest_mobility_6051/blob/main/Readme_img/Screenshot%20from%202025-10-28%2001-30-00.png)
본 시스템은 크게 수어/제스처 인식 경로와 음성 인식 경로로 나뉘어 동작한다.

#### 수어/제스처 인식 (운전자 → 시스템)
1. 입력: 운전자가 수어 또는 제스처를 입력.

2. 인식: 카메라가 운전자의 동작을 촬영.

  + MediaPipe를 사용해 손의 **랜드마크(Hand Landmark)**를 실시간으로 추출.

  + 추출된 랜드마크 데이터를 전처리한 후, CTC 기법을 기법으로 인식된 수어를 분석하여 입력.

3. 출력 및 기능 수행:

  + [명령 수행] 인식된 제스처가 특정 명령(예: "길안내 시작", "긴급 호출")일 경우, 시스템은 해당 기능을 수행.

    + 내비게이션: 카카오맵 API와 연동하여 디스플레이에 경로 안내를 시작.
   
    + 인프라 탐색: 현재 위치 주변의 편의점, 주유소, 학교와 같은 인프라를 탐색.
   
    + 장소 정보 검색: 원하는 장소의 정보를 검색할 수 있음.(전화번호, 위치, 등)
   
    + 음성변환: 수어를 통해 입력한 단어 또는 문장을 음성으로 변환해 스피커를 통해 출력.

    + 긴급 SOS: 사전 등록된 보호자 또는 119에 현재위치의 주소를 알리며 구조를 요청.

---
## 🖥️시스템 UI

| 구분 | 설명 |
|:--|:--|
| 🧭 **길찾기 모드 (Navigation Mode)** | 사용자의 수어 명령을 인식하여 카카오맵 API를 통해 최적 경로를 탐색하고, 실시간으로 경로 안내를 제공합니다. |
| 📍 **인프라 검색 모드 (Infrastructure Search Mode)** | 주유소, 편의점, 병원 등 주변 인프라 시설을 수어 입력으로 검색하고 위치 정보를 지도에 표시합니다. |
| 🗺️ **장소 정보 검색 모드 (Place Info Mode)** | 특정 장소의 상세 정보를 조회하고, 위치 기반 결과를 시각적으로 제공합니다. |
| 🗣️ **음성 출력 모드 (Voice Output Mode)** | 인식된 수어 문장을 TTS(Text-To-Speech)로 변환하여 차량 내 스피커를 통해 음성으로 출력합니다. |
| 🚨 **SOS 모드 (Emergency Mode)** | 긴급 제스처 인식 시, 현재 위치 기반으로 구조 요청 신호를 전송하고 지도 화면에 위치를 표시합니다. |

<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/jmin-code/2025ESWContest_mobility_6051/raw/main/Readme_img/%EA%B8%B8%EC%B0%BE%EA%B8%B0.png" width="300px"/><br>
      <sub>네비게이션</sub>
    </td>
    <td align="center">
      <img src="https://github.com/jmin-code/2025ESWContest_mobility_6051/raw/main/Readme_img/%EC%A3%BC%EB%B3%80%EC%9D%B8%ED%94%84%EB%9D%BC%EA%B2%80%EC%83%89.png" width="300px"/><br>
      <sub>주변 인프라 검색</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/jmin-code/2025ESWContest_mobility_6051/raw/main/Readme_img/%EC%9E%A5%EC%86%8C%EC%A0%95%EB%B3%B4%EA%B2%80%EC%83%89.png" width="300px"/><br>
      <sub>장소 정보 검색</sub>
    </td>
    <td align="center">
      <img src="https://github.com/jmin-code/2025ESWContest_mobility_6051/raw/main/Readme_img/%EC%95%88%EC%A0%84%EB%AA%A8%EB%93%9C.png" width="300px"/><br>
      <sub>안전 모드</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/jmin-code/2025ESWContest_mobility_6051/raw/main/Readme_img/Voice.png" width="300px"/><br>
      <sub>음성 인식</sub>
    </td>
    <td align="center">
      <img src="https://github.com/jmin-code/2025ESWContest_mobility_6051/raw/main/Readme_img/SOS.png" width="300px"/><br>
      <sub>SOS 호출</sub>
    </td>
  </tr>
</table>



---
## 📁 프로젝트 구조

```bash
src/
├── core/                  # 핵심 로직 (수어 인식, GPS, TTS 등)
│   ├── bus.py
│   ├── ctc_model.py
│   ├── gps_reader.py
│   ├── hangul_composer.py
│   ├── picam.py
│   ├── sign_engine.py
│   ├── state.py
│   ├── tts.py
│   └── webcam.py
│
├── ui/                    # PyQt5 기반 UI 화면 및 리소스
│   ├── assets/        
│   ├── chat.py
│   ├── description.py
│   ├── location_provider.py
│   ├── navigation.py
│   ├── recognition.py
│   ├── search.py
│   ├── sos.py
│   ├── sos_receiver.py
│   ├── voice.py
│   ├── warning.py
│   └── welcome.py
│
├── CTC/                   # CTC 모델 학습 관련 파일
│   ├── gesture_ctc_model.pth
│   ├── gesture_labels.json
│   └── ...
│
├── SignLanguage_Detect/   # 수어 데이터 수집 및 테스트 코드
│   ├── collect_Hangul.py
│   └── ...
│
├── Kakao_API/             # 카카오맵 API 관련 HTML/JS 파일
│   └── index.html
│
├── config.py              # 전역 설정 파일
├── gps.py                 # GPS 테스트용 스크립트
└── main.py                # 프로그램 메인 실행 파일
```
---

## 개발환경
![개발환경](https://github.com/jmin-code/2025ESWContest_mobility_6051/blob/main/Readme_img/Screenshot%20from%202025-10-28%2001-29-52.png)
