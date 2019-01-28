# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:01:22 2018

@author: JamesChiou
"""

import sys
import re
import requests
import time
from multiprocessing import Pool, Manager
from bs4 import BeautifulSoup
import pandas as pd

timeSleep = 0.1


class PTTCrawler:
    def __init__(self):
        if(len(sys.argv) > 1):
            if sys.argv[1] == 'crawl':
                self.crawl()
            elif sys.argv[1] == 'push':
                self.push(int(sys.argv[2]), int(sys.argv[3]))
            elif sys.argv[1] == 'popular':
                self.popular(int(sys.argv[2]), int(sys.argv[3]))
            elif sys.argv[1] == 'keyword':
                self.keyword(str(sys.argv[2]),
                             int(sys.argv[3]), int(sys.argv[4]))
            else:
                print("InputError")
        else:
            print("InputError")

    # Parse the article
    def parseArticle(self, index_number):
        PTT_URL = "https://www.ptt.cc"
        PTT_URL_index = "https://www.ptt.cc/bbs/Beauty/index" + \
                        str(index_number)+".html"
        linksIgnore = ["https://www.ptt.cc/bbs/Beauty/M.1490936972.A.60D.html",
                       "https://www.ptt.cc/bbs/Beauty/M.1494776135.A.50A.html",
                       "https://www.ptt.cc/bbs/Beauty/M.1503194519.A.F4C.html",
                       "https://www.ptt.cc/bbs/Beauty/M.1504936945.A.313.html",
                       "https://www.ptt.cc/bbs/Beauty/M.1505973115.A.732.html",
                       "https://www.ptt.cc/bbs/Beauty/M.1507620395.A.27E.html",
                       "https://www.ptt.cc/bbs/Beauty/M.1510829546.A.D83.html",
                       "https://www.ptt.cc/bbs/Beauty/M.1512141143.A.D31.html"]

        request = requests.get(PTT_URL_index)
        content = request.text
        soup = BeautifulSoup(content, "html.parser")
        divs = soup.find_all("div", "r-ent")

        links = []
        titles = []
        pushs = []
        authors = []
        dates = []

        for div in divs:
            try:
                title = div.find('a').text
                href = div.find('a')['href']
                link = PTT_URL + href
                # Ignore [公告] & illegal link
                if ("[公告]" in title[:4]) or (link in linksIgnore):
                    continue

                titles.append(str(title).replace(",", " "))
                links.append(link)

                # Check if there is any push or boo
                push = div.find("div", "nrec").find("span")
                if push is None:
                    pushs.append("0")
                else:
                    pushs.append(str(push.text))

                author = div.find("div", "author")
                authors.append(str(author.text))

                date = div.find("div", "date")
                dates.append(date.text.replace("/", "").replace(" ", ""))

            except:
                pass

        return links, pushs, titles, authors, dates

    # Crawler
    def crawl(self):
        # start = 1992 end = 2340
        START = 1992
        END = 2340

        # Restart new file
        with open('all_articles.txt', 'w', encoding='utf-8') as file:
            file.write("")
        with open('all_popular.txt', 'w', encoding='utf-8') as file:
            file.write("")

        for i in range(START, END+1):
            print("Processing: index"+str(i))
            links, pushs, titles, authors, dates = self.parseArticle(i)

            # First page at 2017
            if i == 1992:
                index = dates.index("101")
                links = links[index:]
                pushs = pushs[index:]
                titles = titles[index:]
                authors = authors[index:]
                dates = dates[index:]

            # Last page at 2017
            if i == 2340:
                index = dates.index("101")
                links = links[:index]
                pushs = pushs[:index]
                titles = titles[:index]
                authors = authors[:index]
                dates = dates[:index]

            # Write the file (all_articles & all_popular)
            with open("all_articles.txt", "a", encoding="utf-8") as file:
                with open("all_popular.txt", "a", encoding="utf-8") as file2:
                    for j in range(0, len(links)):
                        file.write(dates[j]+","+titles[j]+","+links[j]+"\n")
                        if "爆" in pushs[j]:
                            file2.write(dates[j]+","+titles[j]+","+links[j] +
                                        "\n")

    # Multiprocess subjob for Push
    def multiprocess_push(self, i, links, pushs, boos):
        print("Processing: "+str(i+1)+"/"+str(len(links)))
        request = requests.get(links[i])
        content = request.text
        soup = BeautifulSoup(content, "html.parser")

        # Find the <div class=push>
        divs = soup.find_all("div", "push")

        # Count push & boo
        for div in divs:
            push_tag = str(div.find("span", {"class": "push-tag"}).text)
            push_id = str(div.find("span", {"class": "push-userid"}).text)

            if("推" in push_tag):
                if(pushs.get(push_id)):
                    pushs.update({push_id: pushs[push_id]+1})
                else:
                    pushs.update({push_id: int(1)})
            elif("噓" in push_tag):
                if(boos.get(push_id)):
                    boos.update({push_id: boos[push_id]+1})
                else:
                    boos.update({push_id: int(1)})
            else:
                continue

        # Protect from banning
        time.sleep(timeSleep)

    # Push
    def push(self, start_date, end_date):
        df = pd.read_csv("all_articles.txt", sep="\n", encoding="utf-8",
                         header=None)
        dates = []
        links = []

        # Find the links
        for i in range(0, len(df[0])):
            matches = re.findall(re.compile(r'[^,]+'), df[0][i])
            # end_date>= i >= start_date
            if (int(matches[0]) >= start_date) and \
               (int(matches[0]) <= end_date):
                dates.append(int(matches[0]))
                links.append(matches[-1])

        # Build Manager dict & list
        pushs = {}
        boos = {}
        m = Manager()
        pushs = m.dict(pushs)
        boos = m.dict(boos)
        links = m.list(links)

        print("Numbers of links: "+str(len(links)))

        # Multiprocess for push analysis
        with Pool(2) as p:
            for i in range(0, len(links)):
                p.apply_async(self.multiprocess_push,
                              args=(i, links, pushs, boos))
                # Protect from banning
                time.sleep(0.1)

            p.close()
            p.join()

        pushs = pd.DataFrame.from_dict(pushs, orient="index",
                                       columns=['value'])
        boos = pd.DataFrame.from_dict(boos, orient="index", columns=['value'])

        # Sort the pushs & boos
        pushs.sort_index(inplace=True)
        boos.sort_index(inplace=True)
        pushs.sort_values(by='value', ascending=False, inplace=True)
        boos.sort_values(by='value', ascending=False, inplace=True)
        pushs_number = pushs["value"].sum()
        boos_number = boos["value"].sum()

        # Write the file
        with open('push[%d-%d].txt' % (int(start_date), int(end_date)),
                  'w', encoding='utf-8') as file:
            file.write("all like: "+str(pushs_number)+"\n")
            file.write("all boo: "+str(boos_number)+"\n")
            for i in range(0, 10):
                file.write("like #%d: %s %d\n" % (int(i+1), pushs.index[i],
                           int(pushs.iloc[i]["value"])))
            for i in range(0, 10):
                file.write("boo #%d: %s %d\n" % (int(i+1), boos.index[i],
                           int(boos.iloc[i]["value"])))

    # popular
    def popular(self, start_date, end_date):
        df = pd.read_csv("all_popular.txt", sep="\n", encoding="utf-8",
                         header=None)
        links = []

        # Find the links
        for i in range(0, len(df[0])):
            matches = re.findall(re.compile(r'[^,]+'), df[0][i])
            # end_date>= i >= start_date
            if (int(matches[0]) >= start_date) and \
               (int(matches[0]) <= end_date):
                links.append(matches[-1])
        popular_number = len(links)
        print("Numbers of links: "+str(popular_number))
        # Write the file
        with open('popular[%d-%d].txt' % (int(start_date), int(end_date)), 'w',
                  encoding='utf-8') as file:
            file.write("number of popular articles: "+str(popular_number)+"\n")
            for i in range(0, popular_number):
                print("Processing: "+str(i+1)+"/"+str(len(links)))
                request = requests.get(links[i])
                content = request.text
                soup = BeautifulSoup(content, "html.parser")
                # Find all <a href> in the content
                divs = soup.find_all("a", href=True)
                for div in divs:
                    # Find image url in <a href>
                    pattern = '(http.+(jpg|png|jpeg|gif)$)'
                    matches = re.findall(re.compile(pattern), div.text)
                    if matches:
                        file.write(""+str(matches[0][0])+"\n")
                # Protect from banning
                time.sleep(timeSleep)

    # keyword
    def keyword(self, keywords, start_date, end_date):
        print(keywords)
        df = pd.read_csv("all_articles.txt", sep="\n", encoding="utf-8",
                         header=None)
        links = []
        # Find the links
        for i in range(0, len(df[0])):
            matches = re.findall(re.compile(r'[^,]+'), df[0][i])
            # end_date>= i >= start_date
            if (int(matches[0]) >= start_date) and \
               (int(matches[0]) <= end_date):
                links.append(matches[-1])

        # Write the file
        with open('keyword(%s)[%d-%d].txt' % (str(keywords), int(start_date),
                  int(end_date)), 'w', encoding='utf-8') as file:
            for i in range(0, len(links)):
                print("Processing: "+str(i+1)+"/"+str(len(links)), end=" ")
                # Set isKeywords = False
                isKeywords = False
                request = requests.get(links[i])
                content = request.text
                soup = BeautifulSoup(content, "html.parser")
                # Find all <div class=article-metaline> in the content
                divs = soup.find_all("div",
                                     {"id": "main-content",
                                      "class": "bbs-screen bbs-content"})
                for div in divs:
                    # Get content before "--"
                    if "※ 發信站" in div.text:
                        string = "--\n※ 發信站: 批踢踢實業坊(ptt.cc)"
                        articleContent = div.text.split(string, 1)[0]
                        # Check if there is keywords
                        if keywords in articleContent:
                            isKeywords = True

                if isKeywords:
                    print(isKeywords)
                    # Find all <a href> in the content
                    isKeywordsDivs = soup.find_all("a", href=True)
                    for div in isKeywordsDivs:
                        # Find image url in <a href>
                        pattern = '(http.+(jpg|png|jpeg|gif)$)'
                        matches = re.findall(re.compile(pattern), div.text)
                        if matches:
                            file.write(""+str(matches[0][0])+"\n")
                else:
                    print()
                # Protect from banning
                time.sleep(timeSleep)


if __name__ == '__main__':
    c = PTTCrawler()
