import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Any

# LangChain 관련 임포트
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# Visualization: prefer seaborn styling; keep matplotlib.pyplot available
try:
    import matplotlib
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None
try:
    import seaborn as sns
    sns.set_style("white")
except Exception:
    # seaborn may not be available; continue without style
    pass

# 환경설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()


DATA_DIR = Path.cwd() / "data"

def list_server_files(directory: Path):
    if not directory.exists():
        return []
    return [str(p) for p in directory.iterdir() if p.suffix.lower() in (".csv", ".xls", ".xlsx")]


@st.cache_data
def read_path_or_buffer(path_or_buffer) -> pd.DataFrame:
    # path_or_buffer can be a Path/str or an uploaded file-like object
    if hasattr(path_or_buffer, "read"):
        # uploaded file
        try:
            # try csv first
            return pd.read_csv(path_or_buffer)
        except Exception:
            path_or_buffer.seek(0)
            return pd.read_excel(path_or_buffer)
    else:
        p = Path(path_or_buffer)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        else:
            return pd.read_excel(p)


st.title("CSV/Excel 데이터 분석 에이전트 made by moon")

# Sidebar: file selection / upload and model options
st.sidebar.header("데이터 입력")
server_files = list_server_files(DATA_DIR)

# 1. 여러 파일 업로드 및 서버 파일 선택 지원
uploaded_files = st.sidebar.file_uploader(
    "CSV 또는 Excel 업로드 (여러 개 가능)", type=["csv", "xlsx", "xls"], accept_multiple_files=True
)
chosen_server_files = []
if server_files:
    chosen_server_files = st.sidebar.multiselect("서버 파일 선택 (여러 개 가능)", server_files)

all_files = []
if uploaded_files:
    all_files.extend(uploaded_files)
if chosen_server_files:
    all_files.extend(chosen_server_files)

if not all_files:
    st.sidebar.info("왼쪽에서 파일을 업로드하거나 서버에 있는 파일을 선택하세요. 현재 data/ 폴더 파일 목록:")
    for f in server_files:
        st.sidebar.write(f)
    st.stop()

# 2. 각 파일별로 시트 선택 및 DataFrame 생성
dataframes = []
for idx, file in enumerate(all_files):
    file_label = getattr(file, "name", str(file))
    is_excel = str(file_label).lower().endswith((".xlsx", ".xls"))
    
    st.sidebar.markdown(f"**{file_label}**")
    
    if is_excel:
        # 엑셀 파일: 시트 목록 읽기
        try:
            if hasattr(file, "read"):
                # UploadedFile: BytesIO
                file.seek(0)
                xls = pd.ExcelFile(file)
            else:
                xls = pd.ExcelFile(file)
            sheet_names = xls.sheet_names
            
            # 여러 시트 선택 가능
            selected_sheets = st.sidebar.multiselect(
                f"시트 선택", sheet_names, default=[sheet_names[0]], key=f"sheets_{idx}_{file_label}"
            )
            
            for sheet in selected_sheets:
                try:
                    df_temp = pd.read_excel(xls, sheet_name=sheet)
                    df_temp['_source_file'] = file_label
                    df_temp['_source_sheet'] = sheet
                    dataframes.append((f"{file_label}_{sheet}", df_temp))
                except Exception as e:
                    st.error(f"{file_label} 시트 {sheet} 읽기 실패: {e}")
        except Exception as e:
            st.error(f"{file_label} 시트 목록 읽기 실패: {e}")
            continue
    else:
        # CSV 파일
        try:
            if hasattr(file, "read"):
                file.seek(0)
                df_temp = pd.read_csv(file)
            else:
                df_temp = pd.read_csv(file)
            df_temp['_source_file'] = file_label
            dataframes.append((file_label, df_temp))
        except Exception as e:
            st.error(f"{file_label} 읽기 실패: {e}")

# 3. 여러 DataFrame 합치기 옵션
if not dataframes:
    st.error("선택된 파일에서 데이터를 읽을 수 없습니다.")
    st.stop()

if len(dataframes) > 1:
    st.sidebar.markdown("---")
    merge_option = st.sidebar.radio(
        "여러 데이터프레임 처리 방법", ["각각 개별 분석", "모두 합치기 (concat)"], key="merge_option"
    )
    
    if merge_option == "모두 합치기 (concat)":
        try:
            df = pd.concat([d[1] for d in dataframes], ignore_index=True)
            st.info(f"{len(dataframes)}개 파일/시트가 합쳐졌습니다. (총 {len(df):,}행)")
        except Exception as e:
            st.error(f"합치기 실패: {e}")
            st.stop()
    else:
        # 개별 분석을 위해 첫 번째 선택
        selected_df = st.sidebar.selectbox(
            "분석할 데이터 선택", 
            [f"{i+1}. {name} ({len(df):,}행)" for i, (name, df) in enumerate(dataframes)],
            key="selected_df"
        )
        selected_idx = int(selected_df.split('.')[0]) - 1
        df = dataframes[selected_idx][1]
        st.info(f"선택된 데이터: {dataframes[selected_idx][0]} ({len(df):,}행)")
else:
    df = dataframes[0][1]
    st.info(f"로드된 데이터: {dataframes[0][0]} ({len(df):,}행)")

st.subheader("데이터 미리보기")
st.write(df.head())

# Sidebar: model / safety options
st.sidebar.header("모델 설정")
model_name = st.sidebar.text_input("Model", value="gpt-4o-mini")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0)
allow_dangerous = st.sidebar.checkbox("Allow dangerous code (trusted only)", value=True)
execute_generated_code = False
if allow_dangerous:
    execute_generated_code = st.sidebar.checkbox("생성된 코드 자동 실행 (위험)", value=False)


@st.cache_resource
def build_llm(model_name: str, temperature: float):
    return ChatOpenAI(model=model_name, temperature=temperature)

def df_fingerprint(df: pd.DataFrame, allow_dangerous: bool) -> str:
    # simple fingerprint to detect DataFrame changes
    cols = ",".join(map(str, df.columns))
    return f"{df.shape}-{len(cols)}-{allow_dangerous}"

def get_or_create_agent(llm, df: pd.DataFrame, allow_dangerous: bool):
    fp = df_fingerprint(df, allow_dangerous)
    if "agent_fp" not in st.session_state or st.session_state["agent_fp"] != fp:
        # create new agent and store in session_state
        st.session_state["agent"] = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            allow_dangerous_code=allow_dangerous,
            verbose=False,
        )
        st.session_state["agent_fp"] = fp
    return st.session_state["agent"]


llm = build_llm(model_name, temperature)
agent = get_or_create_agent(llm, df, allow_dangerous)

if "running" not in st.session_state:
    st.session_state["running"] = False

query = st.text_area("질문을 입력하세요", "ex) 수치형 컬럼들 간 상관계수 상위 5개 쌍을 알려줘")
if st.button("질문하기"):
    if st.session_state["running"]:
        st.warning("이미 실행 중입니다. 완료 후 다시 시도하세요.")
    else:
        st.session_state["running"] = True
        try:
            # 콜백 핸들러를 사용해 에이전트의 툴 출력을 캡처
            class StreamlitCallbackHandler(BaseCallbackHandler):
                def __init__(self):
                    self.records = []

                # 유연하게 다양한 시그니처를 처리
                def on_tool_start(self, serialized: Any, input_str: str, **kwargs) -> None:
                    name = None
                    try:
                        name = serialized.get("name") if isinstance(serialized, dict) else getattr(serialized, "name", None)
                    except Exception:
                        name = str(serialized)
                    self.records.append({"event": "tool_start", "tool": name, "input": input_str})

                def on_tool_end(self, output: Any, **kwargs) -> None:
                    # output은 보통 문자열(툴의 출력)임
                    self.records.append({"event": "tool_end", "output": output})

                def on_chain_end(self, outputs: Any, **kwargs) -> None:
                    self.records.append({"event": "chain_end", "outputs": outputs})

            handler = StreamlitCallbackHandler()
            with st.spinner("에이전트가 답변 중..."):
                # callbacks 파라미터로 핸들러 전달
                result = agent.invoke(query, callbacks=[handler])

                st.subheader("답변")
                st.write(result)

                # 캡처된 레코드 출력 (코드로 보이는 출력은 코드 블록으로 렌더링)
                if handler.records:
                    st.subheader("에이전트 툴 출력 / 실행 로그")
                    # import types for detection
                    try:
                        from matplotlib.figure import Figure as MatplotlibFigure
                    except Exception:
                        MatplotlibFigure = None
                    try:
                        import plotly.graph_objects as go
                    except Exception:
                        go = None
                    try:
                        import altair as alt
                    except Exception:
                        alt = None

                    for rec in handler.records:
                        if rec.get("event") == "tool_end":
                            out = rec.get("output")
                            if out is None:
                                continue

                            # 1) Matplotlib figure
                            if MatplotlibFigure is not None and isinstance(out, MatplotlibFigure):
                                st.pyplot(out)
                                continue

                            # 2) Plotly figure
                            if go is not None and isinstance(out, go.Figure):
                                st.plotly_chart(out)
                                continue

                            # 3) Altair chart
                            if alt is not None and isinstance(out, alt.Chart):
                                st.altair_chart(out)
                                continue

                            # 4) If output is bytes or image path, show as image
                            if isinstance(out, (bytes, bytearray)):
                                st.image(out)
                                continue
                            if isinstance(out, str) and (out.endswith('.png') or out.endswith('.jpg') or out.endswith('.jpeg')):
                                img_path = out
                                try:
                                    st.image(img_path)
                                    continue
                                except Exception:
                                    pass

                            # 5) Fallback to text/code rendering
                            try:
                                text = out if isinstance(out, str) else str(out)
                            except Exception:
                                text = repr(out)

                            is_code = any(k in text for k in ["def ", "import ", "print(", "```python", "pd."])
                            if is_code:
                                st.code(text, language="python")

                                # Optionally execute generated code and render matplotlib figures via st.pyplot
                                if allow_dangerous and execute_generated_code:
                                    # remove plt.show() calls which are not needed for Streamlit
                                    code_exec = text.replace("plt.show()", "")
                                    code_exec = code_exec.replace("plt.show\n", "")
                                    code_exec = code_exec.replace("show()", "")
                                    try:
                                        exec_locals = {"pd": pd, "df": df}
                                        # provide plt and sns if available
                                        try:
                                            import seaborn as sns
                                            exec_locals["sns"] = sns
                                        except Exception:
                                            pass
                                        if 'plt' in globals() and plt is not None:
                                            exec_locals["plt"] = plt

                                        # Execute the code in a restricted local namespace
                                        exec(code_exec, {}, exec_locals)

                                        # Try to get a figure object
                                        fig = None
                                        if "fig" in exec_locals:
                                            fig = exec_locals.get("fig")
                                        else:
                                            try:
                                                if plt is not None:
                                                    fig = plt.gcf()
                                            except Exception:
                                                fig = None

                                        if fig is not None:
                                            try:
                                                st.pyplot(fig)
                                            except Exception as e:
                                                st.error(f"도표 렌더링 실패: {e}")
                                        else:
                                            st.info("코드 실행은 완료되었지만, matplotlib Figure 객체를 찾을 수 없습니다.")
                                    except Exception as e:
                                        st.error(f"생성된 코드 실행 중 에러: {e}")
                            else:
                                st.text(text)

        except Exception as e:
            st.error(f"에러 발생: {e}")
        finally:
            st.session_state["running"] = False