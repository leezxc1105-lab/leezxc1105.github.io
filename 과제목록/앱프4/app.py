import gradio as gr
import sqlite3
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. DB에서 데이터를 가져오는 함수 (FastAPI 기능을 Gradio가 직접 수행하도록 합침)
def get_analysis():
    try:
        conn = sqlite3.connect('quotes.db')
        df = pd.read_sql_query("SELECT * FROM quotes", conn)
        conn.close()
        
        # 워드클라우드 생성
        text = " ".join(df['tags'].tolist()).replace(",", "")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        return df, fig
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]}), None

# 2. Gradio UI 구성
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Cloud 배포용 격언 분석 시스템")
    with gr.Tab("데이터 조회"):
        data_table = gr.Dataframe()
        btn = gr.Button("데이터 불러오기")
    with gr.Tab("시각화"):
        plot = gr.Plot()
        btn_plot = gr.Button("분석 시작")

    btn.click(get_analysis, outputs=[data_table, plot], show_progress=True)
    btn_plot.click(lambda: get_analysis()[1], outputs=plot)

# 3. 실행 (배포용이므로 share는 필요 없음)
demo.launch()
