from selenium import webdriver
from bs4 import BeautifulSoup


chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument("start-maximized")


PATH= "C:\Program Files (x86)\chromedriver.exe"
driver= webdriver.Chrome(PATH)
driver.maximize_window()
#driver.implicitly_wait(5)
driver.get("https://www.lonelyplanet.com/England/london")
driver.implicitly_wait(10)


html2 = driver.page_source
soup = BeautifulSoup(html2, 'html.parser')
driver.quit()
body = soup.body

#article_body bs4 element with everithing we need
article_body = body.find("div", { "class" : "jsx-3835216876 introduction mx-auto"})

print(article_body.find('p').text)
for i in article_body:
    if "</h3>" in str(i):
        title = str(i).split("</h3>")[0].split("text-lg")[1][2:].replace("amp;", "")
        content = str(i).split("</h3>")[1].split("<p>")[1].split("</p")[0].replace("amp;", "")
        print(title)
        print(content)

