from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
l=[]
def main():
    j = 0
    df = pd.read_excel('first_300_des.xlsx', engine='openpyxl')
    cities = df['city'].tolist()
    for i in range(len(cities)):
        crawl(cities[i], df, j)
        j = j+1
    df.to_excel('first_300_des_res.xlsx')


def crawl(city, df, j):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument("start-maximized")
    PATH= "C:\Program Files (x86)\chromedriver.exe"
    driver= webdriver.Chrome(PATH)
    driver.maximize_window()
    driver.get("https://www.lonelyplanet.com/"+city)
    try:
        button = driver.find_element_by_xpath('//*[@title="Read More"]')
        driver.execute_script("arguments[0].click();", button)
    except:
        print()
    driver.implicitly_wait(5)
    html2 = driver.page_source
    soup = BeautifulSoup(html2, 'html.parser')
    driver.quit()
    body = soup.body
    try:
        text = ''
        #article_body bs4 element with everithing we need
        article_body = body.find("div", { "class" : "jsx-3835216876 introduction mx-auto"})
        text += article_body.find('p').text
        for i in article_body:
            if "</h3>" in str(i):
                title = str(i).split("</h3>")[0].split("text-lg")[1][2:].replace("amp;", "")
                content = str(i).split("</h3>")[1].split("<p>")[1].split("</p")[0].replace("amp;", "")
                text += title + '\n' + content
        df.loc[j, 'description'] = text
    except:
        print()



if __name__ == '__main__':
    main()