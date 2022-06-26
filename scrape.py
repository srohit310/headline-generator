import requests
import json
import time
from bs4 import BeautifulSoup
import pandas as pd
from os import system, name

class Inshorts:

    def __init__(self, dict):

        self.url_inshorts = "https://www.inshorts.com/en/read"
        self.ajax_url_inshorts = "https://inshorts.com/en/ajax/more_news"
        self.dict = dict

    def storedata(self, soup):

        for data in soup.findAll("div",{"class":"news-card z-depth-1"}):
            if data.find(itemprop="headline").getText() not in self.dict["headlines"]:
                self.dict["headlines"].append(data.find(itemprop="headline").getText())
                self.dict["text"].append(data.find(itemprop="articleBody").getText())

    
    def extract_data(self, headers):

        r=requests.get(self.url_inshorts,headers=headers)
        soup=BeautifulSoup(r.content,"lxml")
        self.storedata(soup)

        start_id=soup.findAll("script",{"type":"text/javascript"})[-1].getText().split()[3].strip(";").strip('"')

        for i in range(1000):
            payload={"news_offset":start_id,"categopry":""}

            try:
                r=requests.post(self.ajax_url_inshorts,payload,headers=headers)
                start_id=r.content.decode("utf-8")[16:26]
                soup=BeautifulSoup(r.text.replace('\\',""),"lxml")
                self.storedata(soup)
            except:
                pass

            system('cls')
            print(len(self.dict["headlines"]), 'articles fetched from inshorts')
        
        return self.dict

class Ndtv:

    def __init__(self, dict):

        self.dict = dict

    def storedata(self, soup):

        for data in soup.findAll("div",{"class":"news_Itm-cont"}):
            if data.h2.getText() not in self.dict["headlines"]:
                self.dict["headlines"].append(data.h2.getText())
                self.dict["text"].append(data.p.getText())

    
    def extract_data(self, headers):

        index = 1
        while True:
            url = 'http://www.ndtv.com/latest/page-'+str(index)
            res = requests.get(url, headers=headers)
            if res.status_code == 200:
                soup=BeautifulSoup(res.content,"lxml")
                self.storedata(soup)
                index += 1
            else:
                break
            system('cls')
            print(len(self.dict["headlines"]), 'articles fetched from NDTV')
        
        return self.dict

class Toi:

    def __init__(self, dict):

        self.dict = dict

    def storedata(self, jsontxt):

        itemsList =  jsontxt['sections'][0]['items']
        for item in itemsList:
            item['hl'] = item['hl'].encode("ascii", "ignore").decode("utf-8")
            item['des'] = item['des'].encode("ascii", "ignore").decode("utf-8")
            if item['hl'] not in self.dict["headlines"]:
                self.dict["headlines"].append(item['hl'])
                self.dict["text"].append(item['des'])

    
    def extract_data(self, headers):

        for i in range(1,5):
            url = 'https://toifeeds.indiatimes.com/treact/feeds/toi/web/list/section?path=/toi-plus/all-toi-stories&curpg='+str(i)
            res = requests.get(url, headers=headers)
            if res.status_code == 200:
                soup=BeautifulSoup(res.content,"lxml")
                jsontxt = json.loads(soup.p.text)
                self.storedata(jsontxt)
            else:
                break
            system('cls')
            print(len(self.dict["headlines"]), 'articles fetched from times of india')
        
        return self.dict

def extract_stored_data(dict,site):

    dict={"headlines":[],"text":[]}
    try:
        df = pd.read_csv("dataset/news_summary_"+site+".csv")
        dict["headlines"] = df['headlines'].to_list()
        dict["text"] = df['text'].to_list()
    except:
        dict={"headlines":[],"text":[]}

    return dict

def combine():
    
    sits_names = ['inshorts','ndtv','toi']
    for site in sits_names:
        df = pd.read_csv('dataset/news_summary_'+site+'.csv')
        df.to_csv("dataset/news_summary_combine.csv", mode = 'a', index=False)


def extract_news(id):

    dict = {}
    INSHORTS = "inshorts"
    NDTV = "ndtv"
    TOI = "toi"

    if(id == 1 or id == 4):
        dict = extract_stored_data(dict,INSHORTS)
        inshorts = Inshorts(dict)
        dict = inshorts.extract_data(headers)
        df = pd.DataFrame(dict)
        df.to_csv("dataset/news_summary_"+INSHORTS+".csv", index=False)

    if(id == 2 or id == 4):
        dict = extract_stored_data(dict,NDTV)
        ndtv = Ndtv(dict)
        dict = ndtv.extract_data(headers)
        df = pd.DataFrame(dict)
        df.to_csv("dataset/news_summary_"+NDTV+".csv", index=False)
    
    if(id == 3 or id == 4):
        dict = extract_stored_data(dict,TOI)
        toi = Toi(dict)
        dict = toi.extract_data(headers)
        df = pd.DataFrame(dict)
        df.to_csv("dataset/news_summary_"+TOI+".csv", index=False)

    if(id == 4):
        combine()

if __name__ == '__main__':
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'X-Requested-With': 'XMLHttpRequest',
    }

    extract_news(4)

