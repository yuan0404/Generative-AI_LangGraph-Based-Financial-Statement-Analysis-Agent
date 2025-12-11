from typing import TypedDict, Optional, Callable
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
import yfinance as yf
import os
import re


class State(TypedDict, total=False):
    user_input: str
    memory: str
    time: str

    parse_result: Optional[str]
    extract_result: Optional[str]
    fetch_result: Optional[str]
    clean_result: Optional[str]
    final_output: Optional[str]

    extract_retry_count: list[int]
    clean_retry_count: list[int]
    answer_retry_count: list[int]


def init_state(user_input: str, memory: str, time: str) -> State:
    return {
        "user_input": user_input,
        "memory": memory,
        "time": time,

        "parse_result": None,
        "extract_result": None,
        "fetch_result": None,
        "clean_result": None,
        "final_output": None,

        "extract_retry_count": [0],
        "clean_retry_count": [0],
        "answer_retry_count": [0],
    }


def init_chat() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")

    chat = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key
    )
    return chat


# Parse
def parse(state: State) -> State:
    user_input = state["user_input"]
    memory = state["memory"]
    time = state["time"]

    chat = init_chat()
    prompt = f"""
    你是一位專業的內容潤飾助手，當使用者輸入問題時，請依據以下兩項參數重新整理並提升其清晰度：
    1. 當前時間：{time}
    2. 過去的對話歷史：
    {memory}

    規則如下：
    1. 將問題內容重新整理，使其語意清楚、語法正確、邏輯完整。
    2. 如果問題中有模糊或不明確的地方，請主動釐清並補足，使其成為一個具體、完整且可理解的問題。
    3. 如果問題中有需要計算比率的地方，請加入計算的公式。
    4. 如果問題中提及某個時間範圍，請轉換成精準且標準化的時間格式（例如：「2024」、「2023Q4」）。
    5. 僅輸出最終潤飾後的版本，不需任何額外說明、步驟或段落。

    問題如下：
    {user_input}
    """
    state["parse_result"] = chat.invoke(prompt).content
    return state


def parse_check(state: State) -> bool:
    chat = init_chat()
    parse_result = state["parse_result"]
    prompt = f"""
    你是一個財務分析問題驗證器，任務是判斷給定的文字是否為清楚且定義明確的財務報表分析問題。

    規則如下：
    1. 與財務報表或財務指標相關。
    2. 明確指定一間或多間公司。
    3. 明確指定一個或多個時間區間。
    4. 問題內容具體、清楚且不含模糊之處。
    5. 最終僅輸出一個字：True 或 False，不得包含任何其他說明或段落。

    文字如下：
    {parse_result}
    """
    check = chat.invoke(prompt).content
    return check == "True"


def parse_error(state: State) -> State:
    state["final_output"] = "問題類型錯誤!"
    return state


def parse_router(state: State) -> int:
    if parse_check(state):
        return "success"
    return "failed"


# Extract
def extract(state: State) -> State:
    parse_result = state["parse_result"]
    chat = init_chat()
    prompt = f"""
    你是一位專業財務分析師，負責精準判斷使用者問題中明確提及的公司、時間、與財務報表資料需求。

    規則如下：
    一、公司判斷
    1. 僅在使用者問題中明確提到或根據問題語境必要且直接相關的公司名稱時，才輸出該公司的股票代號。
    2. 如問題未明確指向任何公司，或無法確定相關公司，請輸出「無相關公司」。
    3. 對於在台灣證券交易所上市的公司，股票代號需加上 .TW（例如：台積電 → 2330.TW）。
    4. 如果僅涉及一家公司，直接輸出該公司股票代號，例如：2330.TW。
    5. 如果涉及多家公司，僅輸出必要公司的股票代號，並以逗號（", "）分隔，例如：2454.TW, 2317.TW。
    6. 僅輸出股票代號，不解釋原因、不提供任何額外資訊或回應。
    7. 禁止推測或輸出與問題無關的股票代號。

    二、時間判斷
    1. 僅在使用者問題中明確提到或根據問題語境必要且直接相關的年份或季度時，才輸出該年份或季度。
    2. 如果問題未明確指向任何年份或季度，或無法確定相關時間範圍，請輸出「無相關年份或季度」。
    3. 如果僅需使用單一年份或季度，直接輸出，例如：2024。
    4. 如果涉及多個年份或季度，僅輸出必要的年份或季度，並以逗號（", "）分隔，例如：2024Q3, 2023Q1。
    5. 僅輸出年份或季度，不解釋原因、不提供任何額外資訊或回應。
    6. 輸出為單行，需準確反映當前財務報表的時間範圍，以年份或季度為單位。
    7. 禁止推測或輸出與問題無關的年份或季度。

    三、財務報表判斷
    1. 僅在使用者問題中明確提到或根據問題語境與計算比率間接相關的財務報表類型時，才輸出該財務報表類型。
    2. 如果問題中無法確定相關財務報表類型，請輸出「無相關報表」。
    3. 財務報表的類型僅包含：年度損益表, 季度損益表, 年度資產負債表, 季度資產負債表, 年度現金流量表, 季度現金流量表。
    4. 如果僅需要一個財務報表類型，直接輸出，例如：年度損益表。
    5. 如果需要多個財務報表類型，僅輸出必要財務報表類型，並以逗號（", "）分隔，例如：季度資產負債表, 年度現金流量表。
    6. 僅輸出財務報表類型，不解釋原因、不提供任何額外資訊或回應。
    7. 輸出為單行，不得包含換行符號或多餘內容。
    8. 禁止輸出與問題無關的財務報表類型。

    格式如下：
    公司：<股票代號或「無相關公司」>
    時間：<年份或季度或「無相關年份或季度」>
    報表：<財務報表類型或「無相關報表」>

    問題如下：
    {parse_result}
    """

    state["extract_result"] = chat.invoke(prompt).content
    return state


def extract_check(state: State) -> bool:
    chat = init_chat()
    extract_result = state["extract_result"]
    prompt = f"""
    你是一個財務分析欄位驗證器，任務是檢查輸入內容中的三個欄位（公司、時間、報表）是否皆為有效與完整的財務分析資訊。

    規則如下：
    1. 公司欄位：不得為「無相關公司」，且內容必須為有效的股票代號格式（例如：2330.TW, AAPL）。
    2. 時間欄位：不得為「無相關年份或季度」，且格式必須為正確的年份（如 2024）或季度格式（如 2023Q4）。
    3. 報表欄位：不得為「無相關報表」，且內容必須為以下其中之一或多個，以逗號分隔：年度損益表, 季度損益表, 年度資產負債表, 季度資產負債表, 年度現金流量表, 季度現金流量表。
    4. 三個欄位格式均需正確、無錯字、無多餘符號或無法辨識內容。
    5. 最終僅輸出一個字：True 或 False，不得包含任何其他說明或段落。

    文字如下：
    {extract_result}
    """
    check = chat.invoke(prompt).content
    return check == "True"


def extract_retry(state: State) -> bool:
    state["extract_retry_count"][0] += 1
    return state["extract_retry_count"][0] <= 4


def extract_error(state: State) -> State:
    state["final_output"] = "查無相關資料!"
    return state


def extract_router(state: State) -> int:
    if extract_check(state):
        return "success"
    if extract_retry(state):
        return "retry"
    return "failed"


# Fetch
def fetch(state: State) -> State:
    extract_result = state["extract_result"]
    lines = extract_result.split("\n")

    companies = []
    times = []
    reports = []

    for line in lines:
        if line.startswith("公司："):
            companies = [x.strip() for x in line.replace("公司：", "").split(",")]
        elif line.startswith("時間："):
            times = [x.strip() for x in line.replace("時間：", "").split(",")]
        elif line.startswith("報表："):
            reports = [x.strip() for x in line.replace("報表：", "").split(",")]

    result = ""

    for c in companies:
        for t in times:
            for r in reports:
                stock = yf.Ticker(c)

                is_quarter = "季度" in r and "Q" in t
                is_annual = "年度" in r and "Q" not in t
                if not is_quarter and not is_annual:
                    continue

                if "損益表" in r:
                    df = stock.quarterly_financials if is_quarter else stock.financials
                elif "資產負債表" in r:
                    df = stock.quarterly_balance_sheet if is_quarter else stock.balance_sheet
                elif "現金流量表" in r:
                    df = stock.quarterly_cashflow if is_quarter else stock.cashflow

                pattern = r"^(\d{4})(?:Q([1-4]))?$"
                match = re.match(pattern, t.strip().upper())
                year = int(match.group(1))
                quarter = int(match.group(2)) if match.group(2) else None

                if quarter == 1:
                    date = f"{year}-03-31"
                elif quarter == 2:
                    date = f"{year}-06-30"
                elif quarter == 3:
                    date = f"{year}-09-30"
                elif quarter == 4:
                    date = f"{year}-12-31"
                else:
                    date = f"{year}-12-31"

                df.columns = df.columns.astype(str)
                if date in df.columns:
                    df = df[[date]]
                    if not df.empty:
                        result += f"{c} 的 {t} {r}\n"
                        result += df.to_string()
                        result += "\n\n"
                    else:
                        result += f"找不到 {c} 的 {t} {r}\n\n"
                else:
                    result += f"找不到 {c} 的 {t} {r}\n\n"

    state["fetch_result"] = result
    return state


def fetch_check(state: State) -> bool:
    return state.get("fetch_result") is not None


def fetch_error(state: State) -> State:
    state["final_output"] = "查無相關資料!"
    return state


def fetch_router(state: State) -> int:
    if fetch_check(state):
        return "success"
    return "failed"


# Clean
def clean(state: State) -> State:
    parse_result = state["parse_result"]
    fetch_result = state["fetch_result"]
    chat = init_chat()
    prompt = f"""
    你是一位專業財務分析師，負責從提供的財務資料中，根據使用者問題篩選出真正有用的資料。

    規則如下：
    1. 保留表格標題，例如「2330.TW 的 2025Q3 季度損益表:」。
    2. 僅移除與問題完全無關的資料，只要有可能相關就保留。
    3. 每個行以換行符號分隔。
    4. 不要計算和額外的解釋，僅輸出篩選後的資料。
    5. 如果資料中沒有相關的行，輸出「無相關資料」。

    問題如下：
    {parse_result}

    資料如下：
    {fetch_result}
    """
    state["clean_result"] = chat.invoke(prompt).content
    return state


def clean_check(state: State) -> bool:
    chat = init_chat()
    parse_result = state["parse_result"]
    clean_result = state["clean_result"]
    prompt = f"""
    你是一個財務分析資料完整性驗證器，任務是檢查輸入的財務資料是否完整，是否包含回答使用者問題所需的所有欄位。

    規則如下：
    1. 資料不得為「無相關資料」。
    2. 檢查資料中是否包含所有回答問題必須有的資訊，且保留對應的表格標題。
    3. 最終僅輸出一個字：True 或 False，不得包含任何其他說明或段落。

    問題如下：
    {parse_result}

    資料如下：
    {clean_result}
    """
    check = chat.invoke(prompt).content
    return check == "True"


def clean_retry(state: State) -> bool:
    state["clean_retry_count"][0] += 1
    return state["clean_retry_count"][0] <= 4


def clean_error(state: State) -> State:
    state["final_output"] = "查無相關資料!"
    return state


def clean_router(state: State) -> int:
    if clean_check(state):
        return "success"
    if clean_retry(state):
        return "retry"
    return "failed"


# Answer
def answer(state: State) -> State:
    parse_result = state["parse_result"]
    clean_result = state["clean_result"]
    chat = init_chat()
    prompt = f"""
    你是一位專業財務分析師，負責根據提供的資料完整並詳細的回答使用者問題。

    規則如下：
    1. 回答必須使用提供的資料，不得杜撰內容。
    2. 使用 Markdown 格式作答。

    問題如下：
    {parse_result}

    資料如下：
    {clean_result}
    """
    state["final_output"] = chat.invoke(prompt).content
    return state


def answer_check(state: State) -> bool:
    parse_result = state["parse_result"]
    clean_result = state["clean_result"]
    final_output = state["final_output"]
    chat = init_chat()
    prompt = f"""
    你是一個財務分析回答正確性驗證器，任務是根據問題和資料，檢查回答的正確性和完整性。

    規則如下：
    1. 檢查回答是否完整反映問題所需的財務資訊。
    2. 核對回答是否與資料一致。
    3. 檢查是否為 Markdown 格式。
    4. 最終僅輸出一個字：True 或 False，不得包含任何其他說明或段落。

    問題如下：
    {parse_result}

    資料如下：
    {clean_result}

    回答如下：
    {final_output}
    """
    check = chat.invoke(prompt).content
    return check == "True"


def answer_retry(state: State) -> bool:
    state["answer_retry_count"][0] += 1
    return state["answer_retry_count"][0] <= 4


def answer_error(state: State) -> State:
    state["final_output"] = "查無相關資料!"
    return state


def answer_router(state: State) -> int:
    if answer_check(state):
        return "success"
    if answer_retry(state):
        return "retry"
    return "failed"


def init_graph() -> Callable[[State], State]:
    graph = StateGraph(State)

    graph.add_node("parse", parse)
    graph.add_node("extract", extract)
    graph.add_node("fetch", fetch)
    graph.add_node("clean", clean)
    graph.add_node("answer", answer)

    graph.add_node("parse_error", parse_error)
    graph.add_node("extract_error", extract_error)
    graph.add_node("fetch_error", fetch_error)
    graph.add_node("clean_error", clean_error)
    graph.add_node("answer_error", answer_error)

    graph.add_edge(START, "parse")

    graph.add_conditional_edges("parse", parse_router, {"success": "extract", "failed": "parse_error"})
    graph.add_conditional_edges("extract", extract_router, {"success": "fetch", "retry": "extract", "failed": "extract_error"})
    graph.add_conditional_edges("fetch", fetch_router, {"success": "clean", "failed": "fetch_error"})
    graph.add_conditional_edges("clean", clean_router, {"success": "answer", "retry": "clean", "failed": "clean_error"})
    graph.add_conditional_edges("answer", answer_router, {"success": END, "retry": "answer", "failed": "answer_error"})

    return graph.compile()
