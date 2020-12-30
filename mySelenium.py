# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# import time
# from selenium.webdriver.common.keys import Keys
#
# PATH= "C:\Program Files (x86)\chromedriver.exe"
# driver= webdriver.Chrome(PATH)
# driver.get("https://www.lonelyplanet.com/England/london")
#
# # search= driver.find_elements_by_name("search-recommendation-place-input")
# # search.send_Keys("London")
# # search.send_Keys(Keys.RETURN)
#
# try:
#     main= WebDriverWait(driver,50).until(
#         EC.presence_of_element_located((By.ID,"main"))
#     )
#     articles= main.find_elements_by_tag_name("article")
#     for article in articles:
#         header= article.find_elements_by_class_name("jsx-3835216876 text")
#         print(header)
# finally:
#      driver.quit()





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
#driver.implicitly_wait(15)
driver.quit()
body = soup.body
#article_body bs4 element with everithing we need
article_body = body.find("div", { "class" : "jsx-3835216876 introduction mx-auto"})
for i in article_body:
    print(i.text)


























