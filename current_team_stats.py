# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 18:49:21 2021

@author: caiog
"""

def get_k(vic_margin, elo_diff_winner):
    return 20*((vic_margin+3)**0.8)/(7.5 + 0.006*elo_diff_winner)

def get_e_team(team_elo, opp_team_elo):
    return 1/(1+10**((opp_team_elo - team_elo)/400))

def reset_season_elo(season_id, g, elo_dic):
    if season_id != '' and season_id != g.iloc[[0],:].iloc[0]['SEASON_ID']:
            for k, v in elo_dic.items():
                elo_dic[k] = v*0.75 + 0.25*1505
                
def update_elo(winner, elo_a, elo_b, elo_dic, team_a_id, team_b_id):
    if winner == 'A':
        vic_margin = team_a_pts - team_b_pts
        elo_diff_winner = elo_a - elo_b
        elo_dic[team_a_id] = get_k(vic_margin, elo_diff_winner)*(1 - get_e_team(elo_a, elo_b)) + elo_a
        elo_dic[team_b_id] = get_k(vic_margin, elo_diff_winner)*(0 - get_e_team(elo_b, elo_a)) + elo_b
    else:
        vic_margin = team_b_pts - team_a_pts
        elo_diff_winner = elo_b - elo_a
        elo_dic[team_a_id] = get_k(vic_margin, elo_diff_winner)*(0 - get_e_team(elo_a, elo_b)) + elo_a
        elo_dic[team_b_id] = get_k(vic_margin, elo_diff_winner)*(1 - get_e_team(elo_b, elo_a)) + elo_b
        
def current_streak (previous_games):
    if len(previous_games.index) > 0:
        previous_games['start_of_streak'] = previous_games.WL.ne(previous_games['WL'].shift())
        previous_games['streak_id'] = previous_games['start_of_streak'].cumsum()
        previous_games['streak_counter'] = previous_games.groupby('streak_id').cumcount() + 1
        if previous_games.iloc[-1,7] == 'W':
            return previous_games.iloc[-1,-1]
        else:
            return -1*previous_games.iloc[-1,-1]
    else:
        return 0


def get_wl_pct (previous_games):
    if len(previous_games.index) > 0:
        wl = previous_games['WL'].value_counts(normalize=True)
        if 'W' in wl and 'L' in wl:
            win_pct = wl['W']
            loss_pct = wl['L']
        elif 'W' not in wl and 'L' in wl:
            win_pct = 0
            loss_pct = wl['L']
        elif 'W' in wl and 'L' not in wl:
            win_pct = wl['W']
            loss_pct = 0
        return win_pct, loss_pct
    return 0, 0

import pandas as pd
import numpy as np
from functools import reduce
from nba_api.stats.endpoints import teamplayerdashboard, leaguestandings, teamplayerdashboard, leagueleaders, teamestimatedmetrics, teamgamelog, teamgamelogs, leaguegamelog
from nba_api.stats.static import teams 

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None  # default='warn'
    teams = teams.get_teams()
    
    seasons_teams = []
    seasons_players = []
    first_season = 2008
    last_season = 2021
    
    print("Getting NBA Seasons Information...")
    for i in range(first_season,last_season):
        season_i_teams = leaguegamelog.LeagueGameLog(season = str(i)).get_data_frames()[0]
        season_i_players = leaguegamelog.LeagueGameLog(season = str(i), player_or_team_abbreviation = 'P').get_data_frames()[0]
        seasons_teams.append(season_i_teams)
        seasons_players.append(season_i_players)
        print("{}/{}".format(i,last_season))
        
    
    dfs = []
    
    season_games = reduce(lambda  left,right: pd.merge(left,right, how='outer'), seasons_teams)
    season_games_plyrs = reduce(lambda  left,right: pd.merge(left,right, how='outer'), seasons_players)
    season_games.dropna(subset=['FG_PCT','FT_PCT','FG3_PCT'], inplace=True)
    
    season_games_plyrs['GAME_ID'] = pd.to_numeric(season_games_plyrs['GAME_ID'])
    season_games['GAME_ID'] = pd.to_numeric(season_games['GAME_ID'])
    
    print('size', len(season_games.index))
    
    elo_dic = dict()
    
    for team in teams:
        elo_dic[team['id']] = 1500
        
    season_id = ''    
    
    for i, g in season_games.groupby(season_games.index // 2):
        if g.iloc[[0],:].iloc[0]['WL'] == None:
            break
        print("{}/{}".format(i, len(season_games.index) // 2))
        
        team_a_id = g.iloc[[0],:].iloc[0]['TEAM_ID']
        team_b_id = g.iloc[1:2,:].iloc[0]['TEAM_ID']
        
        team_a_abbv = g.iloc[[0],:].iloc[0]['TEAM_ABBREVIATION']
        team_b_abbv = g.iloc[1:2,:].iloc[0]['TEAM_ABBREVIATION']
        
        reset_season_elo(season_id, g, elo_dic)
        
        season_id = g.iloc[[0],:].iloc[0]['SEASON_ID']
        
        winner = 'B'
        
        if g.iloc[[0],:].iloc[0]['WL'] == 'W':
            winner = 'A'
            
        team_a_pts = g.iloc[[0],:].iloc[0]['PTS']
        team_b_pts = g.iloc[1:2,:].iloc[0]['PTS']
        elo_a = elo_dic[team_a_id]
        elo_b = elo_dic[team_b_id]
        
        update_elo(winner, elo_a, elo_b, elo_dic, team_a_id, team_b_id)
            
    
    teams_dic = dict()
    
    for team in teams:
        team_id = team['id']
        team_abbv = team['abbreviation']
        teams_dic[team_abbv] = dict()
    
        previous_games = season_games.loc[(season_games['TEAM_ID'] == team_id) & (season_games['MIN'] > 0)]
        team_season_games = previous_games.loc[previous_games['SEASON_ID'] == season_id]
        
                
        # Season Win Percentage
        season_pct = get_wl_pct(team_season_games)[0]
        
        # Calculating Current Streak
        streak = current_streak(team_season_games)
            
        # Getting Home/Away information
        last_home_games = previous_games[previous_games['MATCHUP'].str.contains('@')].iloc[-10:,:]
        last_away_games = previous_games[~previous_games['MATCHUP'].str.contains('@')].iloc[-10:,:]
            
        if len(last_home_games.index) > 0 and len(last_away_games.index) > 0:
            # print('TEAM A: {}\nTEAM B: {}\nA: {}\nB: {}'.format(team_a_last_ha_games, team_b_previous_games, pd.DataFrame([team_a_last_ha_games['WL'].value_counts(normalize=True)]), pd.DataFrame([team_b_last_ha_games['WL'].value_counts(normalize=True)])))
            home_wl = last_home_games['WL'].value_counts(normalize=True)
            away_wl = last_away_games['WL'].value_counts(normalize=True)
            if 'W' in home_wl:
                home_pct = home_wl['W']
            else:
                home_pct = 0
            if 'W' in away_wl:
                away_pct = away_wl['W']
            else:
                away_pct = 0
        
        # Getting teams last 10 games
        previous_games = previous_games.iloc[-10:,:]
        pts_conceded = []
        for index, game in previous_games.iterrows():
            opp_game = season_games.loc[(season_games['GAME_ID'] == game['GAME_ID']) & (season_games['TEAM_ID'] != game['TEAM_ID'])].iloc[0]
            pts_conceded.append(opp_game['PTS'])
        if len(pts_conceded) > 0:
            pts_conceded = sum(pts_conceded) / len(pts_conceded)
            
            
        if len(previous_games.index) > 0 and len(last_home_games.index) > 0 and len(last_away_games.index) > 0:
            # TEAM A
            teams_dic[team_abbv]['PTS'] = previous_games['PTS'].mean()
            teams_dic[team_abbv]['PTS_CON'] = pts_conceded
            teams_dic[team_abbv]['FG_PCT'] = previous_games['FG_PCT'].mean()
            teams_dic[team_abbv]['FG3_PCT'] = previous_games['FG3_PCT'].mean()
            teams_dic[team_abbv]['FT_PCT'] = previous_games['FT_PCT'].mean()
            teams_dic[team_abbv]['REB'] = previous_games['REB'].mean()
            teams_dic[team_abbv]['TOV'] = previous_games['TOV'].mean()
            teams_dic[team_abbv]['BLK'] = previous_games['BLK'].mean()
            teams_dic[team_abbv]['STREAK'] = streak
            teams_dic[team_abbv]['SEASON_PCT'] = season_pct
            teams_dic[team_abbv]['HOME'] = home_pct
            teams_dic[team_abbv]['AWAY'] = away_pct
            teams_dic[team_abbv]['ELO'] = elo_dic[team_id]
            
    # 'PTS_A', 'PTS_CON_A', 'FG_PCT_A', 'FG3_PCT_A', 'FT_PCT_A', 'REB_A', 'TOV_A', 'BLK_A', 'H/A_A', 'ELO_A', 
    # 'PTS_B', 'PTS_CON_B', 'FG_PCT_B', 'FG3_PCT_B', 'FT_PCT_B', 'REB_B', 'TOV_A', 'BLK_B', 'H/A_B', 'ELO_B', 
    home = input("Digite o time mandante: ")
    away = input("Digite o time visitante: ")
    
    while home != '-1':
        print("print('{} x {}', classifier.predict(sc.transform([[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]])))".format(home, away, teams_dic[home]['PTS'], teams_dic[home]['PTS_CON'], teams_dic[home]['FG_PCT'], teams_dic[home]['FG3_PCT'], teams_dic[home]['FT_PCT'], 
                                                                                          teams_dic[home]['REB'], teams_dic[home]['TOV'], teams_dic[home]['BLK'], teams_dic[home]['SEASON_PCT'], teams_dic[home]['HOME'], teams_dic[home]['ELO'], teams_dic[home]['STREAK'], 
                                                                                          teams_dic[away]['PTS'], teams_dic[away]['PTS_CON'], teams_dic[away]['FG_PCT'], teams_dic[away]['FG3_PCT'], teams_dic[away]['FT_PCT'],
                                                                                          teams_dic[away]['REB'], teams_dic[away]['TOV'], teams_dic[away]['BLK'], teams_dic[away]['SEASON_PCT'], teams_dic[away]['AWAY'], teams_dic[away]['ELO'], teams_dic[away]['STREAK']))
        
        home = input("Digite o time mandante: ")
        if home != '-1':
            away = input("Digite o time visitante: ")
