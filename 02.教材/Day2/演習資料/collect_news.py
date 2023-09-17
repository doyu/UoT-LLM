import requests
from bs4 import BeautifulSoup
import time
import json
import re

urls = [
    "https://weblab.t.u-tokyo.ac.jp/category/lab-news/page/1/",
    "https://weblab.t.u-tokyo.ac.jp/category/lab-news/page/2/",
    "https://weblab.t.u-tokyo.ac.jp/category/lab-news/page/3/",
    "https://weblab.t.u-tokyo.ac.jp/category/lab-news/page/4/",
    "https://weblab.t.u-tokyo.ac.jp/category/lab-news/page/5/"
]

data_list = []
for url in urls:
    # リクエストを送信してHTMLを取得
    response = requests.get(url)
    html = response.text

    # BeautifulSoupを使用して情報を抽出
    soup = BeautifulSoup(html, 'html.parser')

    wf_cells = soup.find_all('div', class_='wf-cell iso-item')

    for wf_cell in wf_cells:
        url = wf_cell.find('a')['href']
        data_date = wf_cell['data-date']
        data_name = wf_cell['data-name']

        # URLにアクセスして記事の内容を取得
        article_response = requests.get(url)
        article_html = article_response.text
        article_soup = BeautifulSoup(article_html, 'html.parser')
        
        entry_content = article_soup.find('div', class_='entry-content')
        if entry_content:
            paragraphs = entry_content.find_all('p')
            article_text = '\n'.join([p.get_text().strip() for p in paragraphs])
        else:
            article_text = "No content available."
            continue
        text_splits = re.split(r'\n{2,}', article_text)
        text_id = 0
        for text_split in text_splits:
            text_split = text_split.replace('\n', '')
            if len(text_split) < 3 or len(text_split) > 300:
                continue
            article_info = {
                "url": url,
                "date": data_date,
                "title": data_name,
                "content": text_split,
                "text_id": text_id
            }
            data_list.append(article_info)
            text_id += 1
        time.sleep(1)
with open('data.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_list, json_file, ensure_ascii=False, indent=4)