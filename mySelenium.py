from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
l=[]
def main():
    j = 0
    df = pd.read_excel('first_300_des.xlsx', engine='openpyxl')
    cities = df['city'].tolist()
    countries = df['country'].tolist()
    for i in range(len(cities)):
        crawl(cities[i],countries[i], df, j)
        j = j+1
    df.to_excel('first_300_des_res.xlsx')

def crawl(city,country, df, j):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument("start-maximized")
    PATH= "C:\Program Files (x86)\chromedriver.exe"
    driver= webdriver.Chrome(PATH)
    driver.maximize_window()
    if pd.isna(country):
        driver.get("https://www.lonelyplanet.com/" + city)
    else:
        driver.get("https://www.lonelyplanet.com/"+country+"/"+city)
    try:
        button = driver.find_element_by_xpath('//*[@title="Read More"]')
        driver.execute_script("arguments[0].click();", button)
    except:
        print()
    driver.implicitly_wait(2)
    html2 = driver.page_source
    soup = BeautifulSoup(html2, 'html.parser')
    driver.quit()
    body = soup.body
    try:
        text = ''
        #article_body bs4 element with everithing we need
        article_body = body.find("div", { "class" : "jsx-1359272777 introduction mx-auto"})
        text += article_body.find('p').text
        for i in article_body:
            title = i.find_all('h3')
            content = i.find_all('p')
            for k in range(len(title)):
                text += '\n' + title[k].text + '\n' + content[k].text
        df.loc[j, 'description'] = text
    except:
        print("")

if __name__ == '__main__':
    main()