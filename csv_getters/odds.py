
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup as soup
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from nba_api.stats.static import teams 
import time
from datetime import datetime, timedelta

months = dict(Jan=1,Feb=2,Mar=3,Apr=4,May=5,Jun=6,Jul=7,Aug=8,Sep=9,Oct=10,Nov=11,Dec=12)

def get_betting_odds(season):
    nba_teams = teams.get_teams()
    base_url = "https://www.oddsportal.com/basketball/usa/nba-{}/results/".format(season)
    option = Options()
    option.headless = True
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=option)
    driver.get(base_url)
    time.sleep(5)
    button = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div[6]/div[1]/div/div[1]/div[2]/div[1]/div[6]/div/a[14]')
    button.click()
    time.sleep(2)
    
    element = driver.find_elements_by_id("tournamentTable")[0]
    htmlContent = element.get_attribute('outerHTML')
    page_soup = soup(htmlContent, "html.parser")
    active_page = int(''.join(page_soup.find('span', {"class": "active-page"}).findAll(text=True)))
    print('Page', active_page)
    
    games = []
    
    while int(active_page) >= 1:
        dates = page_soup.findAll('tr', {"class": "center nob-border"})
        games_registered = 0
        
        for i in range(len(dates), 0, -1):
            if len(dates) == i:
                date_games = dates[i-1].find_next_siblings('tr', {"class": "deactivate"})
            else:
                date_games = dates[i-1].find_next_siblings('tr', {"class": "deactivate"})[:-1*games_registered]
            games_registered += len(date_games)
            date_info_splitted = dates[i-1].contents[0].text.split('-')
            if len(date_info_splitted) > 1:
                continue
            date_text = date_info_splitted[0].strip()
            # print(date_text)
            games_parsed = [dict(date=datetime(int(date_text.split(' ')[2]), 
                                                months[date_text.split(' ')[1]], 
                                                int(date_text.split(' ')[0]), 
                                                hour=int(g.contents[0].text.strip().split(':')[0]), 
                                                minute=int(g.contents[0].text.strip().split(':')[1]), 
                                                second=0) - timedelta(hours=3, minutes=0), # Substracting 3 hours to align with the nba api timezone
                                  # "{} - {}".format(date_text, g.contents[0].text.strip()),
                                  team_a_id=next(filter(lambda x: x['full_name'] == g.contents[1].text.split('-')[0].strip(), nba_teams))['id'],
                                  team_b_id=next(filter(lambda x: x['full_name'] == g.contents[1].text.split('-')[1].strip(), nba_teams))['id'],
                                  team_a=g.contents[1].text.split('-')[0].strip(),
                                  team_b=g.contents[1].text.split('-')[1].strip(),
                                  pts_team_a=g.contents[2].text.split(':')[0].strip(),
                                  pts_team_b=g.contents[2].text.split(':')[1].replace('OT', '').strip(),
                                  odds_team_a=g.contents[3].text, 
                                  odds_team_b=g.contents[4].text) 
                                  for g in date_games if len(g.contents[2].text.split(':')) > 1]
            games.extend(games_parsed)
            # print('{} Games appended'.format(len(games_parsed)))
        
        btn_next_page = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div[6]/div[1]/div/div[1]/div[2]/div[1]/div[6]/div/a[2]')
        btn_next_page.click()
        time.sleep(4)
        
        element = driver.find_elements_by_id("tournamentTable")[0]
        htmlContent = element.get_attribute('outerHTML')
        page_soup = soup(htmlContent, "html.parser")
        active_page -= 1
        # active_page = ''.join(page_soup.find('span', {"class": "active-page"}).findAll(text=True))
        print('Page', active_page)
    
    driver.quit()
    return games
    

if __name__ == "__main__":
    season = '2009-2010'
    games = get_betting_odds(season)