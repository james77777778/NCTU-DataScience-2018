# Homework 1 - PTT beauty crawler

## Packages:
- BeautifulSoup
- pandas

## Usage:
1. 將表特板2017年一整年的文章爬下來
    ```bash
    $ Python 0750730.py crawl
    ```
    - Output:
        - all_articles.txt (所有文章)
        - all_popular.txt (所有爆文)
    - Format:
        ```bash
        日期,標題,URL
        ```
2. 數推文噓文和找出前10名最會推跟噓的人
    ```bash
    $ Python 0750730.py push start_date end_date
    ```
    - Input:
        - all_articles.txt (所有文章)
    - Output:
        - push[start_date-end_date].txt
    - Format:
        ```bash
        all like: 推文數量
        all boo: 噓文數量
        like #rank: userid 推文數
        boo #rank: userid 噓文數
        ```
3. 找爆文和圖片URL
    ```bash
    $ Python 0750730.py popular start_date end_date
    ```
    - Input:
        - all_popular.txt (所有爆文)
    - Output:
        - popular [start_date-end_date].txt
    - Format:
        ```bash
        number of popular articles: 文章數量
        圖片URL
        ...
        ```
4. 找內文中含有{keyword}的文章中的所有圖片
    ```bash
    $ Python 0750730.py keyword {keyword} start_date end_date
    ```
    - Input:
        - all_articles.txt (所有文章)
    - Output:
        - keyword({keyword})[start_date-end_date].txt
    - Format:
        ```bash
        圖片URL
        ...
        ```