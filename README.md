# analysis_agent Streamlit 앱

이 저장소에는 LangChain 기반 Pandas 에이전트 예제가 포함되어 있습니다. `app.py`는 Streamlit 앱으로 로컬에서 실행하거나 Streamlit Community Cloud에 배포할 수 있습니다.

## 로컬 실행 (PowerShell)

1. 가상환경 생성 및 활성화

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. 의존성 설치

```powershell
pip install -r requirements.txt
```

3. 환경변수 설정 (.env 파일 또는 PowerShell 세션)

- 로컬 개발: 프로젝트 루트에 `.env` 파일을 만들고 아래 내용을 추가하세요 (절대 커밋 금지):

```
OPENAI_API_KEY=sk-...
```

4. 앱 실행

```powershell
streamlit run app.py
```

## Streamlit Community Cloud 배포

1. 변경사항을 GitHub 저장소에 커밋하고 푸시합니다. 프로젝트 루트에 `app.py`, `requirements.txt`, `README.md`가 있어야 합니다.
2. https://streamlit.io/cloud 에 로그인하거나 계정을 생성합니다.
3. "New app" 버튼을 누르고 GitHub 저장소를 연결한 뒤 브랜치와 `app.py`를 선택합니다.
4. 배포 시 "Secrets" 또는 "Advanced settings"에서 `OPENAI_API_KEY`를 추가하세요 (키 이름: `OPENAI_API_KEY`, 값: 실제 키).
5. 배포가 완료되면 앱 URL에서 접속합니다.

주의사항:
- `.env` 또는 시크릿 키는 절대 깃에 커밋하지 마세요.
- 대용량 데이터 파일(`data/`)은 저장소에 올리지 말고, 필요하면 외부 스토리지(예: S3)나 DB를 사용하세요.
- 공개 앱에서 `allow_dangerous_code` 옵션은 끄세요.

## 추가 도움
원하시면 `Dockerfile`, CI(예: GitHub Actions) 배포 워크플로, 또는 Streamlit Cloud 환경설정(.streamlit/config.toml) 예시를 추가로 만들어 드립니다.

## Matplotlib / GUI 백엔드 관련(주의)

- 일부 환경에서는 `matplotlib`가 GUI 백엔드(PyQt 등)를 요구하며, 이로 인해 `FigureCanvasAgg is non-interactive` 또는 유사한 경고가 발생할 수 있습니다.
- 해결 방법:
	1. 간단: `requirements.txt`에 `PyQt6`를 추가했습니다. 로컬 환경에서 GUI가 필요한 경우 `pip install -r requirements.txt`로 설치하세요.
	2. 대안(권장): headless 환경에서는 matplotlib 백엔드를 `Agg`로 사용하고 Streamlit의 `st.pyplot()`으로 렌더링하세요. (`app.py`는 seaborn 스타일을 기본으로 사용하도록 설정되어 있으며, 필요한 경우 백엔드 관련 설정을 조정할 수 있습니다.)
	3. Python 버전 호환성: 최신 PyQt와 matplotlib의 호환성은 Python 버전에 영향을 받을 수 있습니다. 최신 조합 사용을 권장합니다(예: Python 3.12+).
