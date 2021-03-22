import requests
from bs4 import BeautifulSoup

indeed_result = requests.get('https://www.uplus.co.kr/css/chgi/chgi/RetrieveTvContentsMFamily.hpi#today')
# print(indeed_result.text)

indeed_soup = BeautifulSoup(indeed_result.text, "html.parser")
# print(indeed_soup)

schedules = indeed_soup.find("div", {"class":"scheduleBox"})
print(schedules)