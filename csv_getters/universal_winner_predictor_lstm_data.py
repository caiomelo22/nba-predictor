# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:03:16 2021

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
                
def update_elo(winner, elo_a, elo_b, elo_dic, team_a_id, team_b_id, team_a_pts, team_b_pts):
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
        
def team_points_conceded(previous_games, season_games):
    previous_games_pts_conceded = []
    for index, game in previous_games.iterrows():
        opp_game = season_games.loc[(season_games['GAME_ID'] == game['GAME_ID']) & (season_games['TEAM_ID'] != game['TEAM_ID'])].iloc[0]
        previous_games_pts_conceded.append(opp_game['PTS'])
    if len(previous_games_pts_conceded) > 0:
        return sum(previous_games_pts_conceded) / len(previous_games_pts_conceded)
    return 0

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
    
def get_player_mean_per(playerLastGames):
    perValues = []
    for index, game in playerLastGames.iterrows():
        perValues.append((game['FGM'] * 85.910 + game['STL'] * 53.897 + game['FG3M'] * 51.757 + game['FTM'] * 46.845 + game['BLK'] * 39.190 + game['OREB'] * 39.190 + game['AST'] * 34.677 + game['DREB'] * 14.707
                          - game['PF'] * 17.174 - (game['FTA'] - game['FTM']) * 20.091 - (game['FGA'] - game['FGM'])* 39.190 - game['TOV'] * 53.897 ) * (1 / game['MIN']))
    if len(perValues) > 0:
        return mean(perValues)
    return 0
    
def get_team_per_mean(teamId, gameId, gameDate, seasonId, seasonAllPlayers):
    gamePlayers = seasonAllPlayers.loc[(seasonAllPlayers['GAME_ID'] == gameId) & (seasonAllPlayers['TEAM_ID'] == teamId) & (seasonAllPlayers['MIN'] >= 22)]
    seasonPlayers = seasonAllPlayers.loc[(seasonAllPlayers['GAME_DATE'] < gameDate) & (seasonAllPlayers['TEAM_ID'] == teamId) & (seasonAllPlayers['SEASON_ID'] == seasonId) & (seasonAllPlayers['MIN'] > 0)]
    perValues = []
    for index, player in gamePlayers.iterrows():
        playerLastTenGames = seasonPlayers.loc[seasonPlayers['PLAYER_ID'] == player['PLAYER_ID']].iloc[-10:]
        perValues.append(get_player_mean_per(playerLastTenGames))
    if len(perValues) > 0:
        return mean(perValues)
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
    
def get_team_stats (previous_games, previous_games_pts_conceded, season_pct, ha_percentage, elo, streak, matchups_pct, per):
    return [previous_games['PTS'].mean(), previous_games_pts_conceded, previous_games['FG_PCT'].mean(), previous_games['FG3_PCT'].mean(), 
                        previous_games['FT_PCT'].mean(), previous_games['REB'].mean(), previous_games['TOV'].mean(),
                        previous_games['BLK'].mean(), season_pct, ha_percentage, elo, streak, matchups_pct, per]

import pandas as pd
import numpy as np
from functools import reduce
from nba_api.stats.endpoints import teamplayerdashboard, leaguestandings, teamplayerdashboard, leagueleaders, teamestimatedmetrics, teamgamelog, teamgamelogs, leaguegamelog
from nba_api.stats.static import teams 
from statistics import mean

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None  # default='warn'
    teams = teams.get_teams()
    
    teams_per = dict()
    
    for team in teams:
        team_id = team['id']
        teams_per[team_id] = 0
    
    seasons_teams = []
    seasons_players = []
    first_season = 2018
    last_season = 2019
    
    print("Getting NBA Seasons Information...")
    for i in range(first_season,last_season):
        season_i_teams = leaguegamelog.LeagueGameLog(season = str(i)).get_data_frames()[0]
        season_i_players = leaguegamelog.LeagueGameLog(season = str(i), player_or_team_abbreviation = 'P').get_data_frames()[0]
        seasons_teams.append(season_i_teams)
        seasons_players.append(season_i_players)
        print("{}/{}".format(i,last_season))
        
    
    print("Cleaning the data...")
    
    season_games = reduce(lambda  left,right: pd.merge(left,right, how='outer'), seasons_teams)
    season_games_plyrs = reduce(lambda  left,right: pd.merge(left,right, how='outer'), seasons_players)
    season_games.dropna(subset=['FG_PCT','FT_PCT','FG3_PCT'], inplace=True)
    
    season_games_plyrs['GAME_ID'] = pd.to_numeric(season_games_plyrs['GAME_ID'])
    season_games['GAME_ID'] = pd.to_numeric(season_games['GAME_ID'])
    season_games['GAME_DATE'] = pd.to_datetime(season_games['GAME_DATE'])
    season_games_plyrs['GAME_DATE'] = pd.to_datetime(season_games_plyrs['GAME_DATE'])
    
    print('size', len(season_games.index))
    
    print("Initializing ELOs...")
    
    elo_dic = dict()
    
    for team in teams:
        elo_dic[team['id']] = 1500
    
    matches_organized = []
    
    season_id = ''    
    
    print("Creating CSV file of all games...")
    for i, g in season_games.groupby(season_games.index // 2):
        print("{}/{}".format(i, len(season_games.index) // 2))
        if g.iloc[[0],:].iloc[0]['WL'] == None:
            break
        
        season_id = g.iloc[[0],:].iloc[0]['SEASON_ID']
        game_id = g.iloc[[0],:].iloc[0]['GAME_ID']
        game_date = g.iloc[[0],:].iloc[0]['GAME_DATE']
        
        team_a_id = g.iloc[[0],:].iloc[0]['TEAM_ID']
        team_b_id = g.iloc[1:2,:].iloc[0]['TEAM_ID']
        
        team_a_abbv = g.iloc[[0],:].iloc[0]['TEAM_ABBREVIATION']
        team_b_abbv = g.iloc[1:2,:].iloc[0]['TEAM_ABBREVIATION']
        
        reset_season_elo(season_id, g, elo_dic)
        
        winner = 'B'
        winner_bin = 0
        
        if g.iloc[[0],:].iloc[0]['WL'] == 'W':
            winner_bin = 1
            winner = 'A'
        
        team_a_previous_games = season_games.loc[(season_games['TEAM_ID'] == team_a_id) & (season_games['GAME_DATE'] < game_date)]
        team_b_previous_games = season_games.loc[(season_games['TEAM_ID'] == team_b_id) & (season_games['GAME_DATE'] < game_date)]
        team_a_season_games = team_a_previous_games.loc[team_a_previous_games['SEASON_ID'] == season_id]
        team_b_season_games = team_b_previous_games.loc[team_b_previous_games['SEASON_ID'] == season_id]
        
        # Getting teams last 10 games
        team_a_previous_10_games = team_a_season_games.iloc[-10:,:]
        team_b_previous_10_games = team_b_season_games.iloc[-10:,:]
        
        # Getting Home/Away information
        if '@' in g.iloc[[0],:].iloc[0]['MATCHUP']:
            team_a_last_ha_games = team_a_season_games[team_a_season_games['MATCHUP'].str.contains('@')].iloc[-10:,:]
            team_b_last_ha_games = team_b_season_games[~team_b_season_games['MATCHUP'].str.contains('@')].iloc[-10:,:]
        else:
            team_a_last_ha_games = team_a_season_games[~team_a_season_games['MATCHUP'].str.contains('@')].iloc[-10:,:]
            team_b_last_ha_games = team_b_season_games[team_b_season_games['MATCHUP'].str.contains('@')].iloc[-10:,:]
        
        if len(team_a_previous_games.index) > 0:
            if team_a_previous_games.iloc[-1]['GAME_ID'] == g.iloc[[0],:].iloc[0]['GAME_ID']:
                print('São iguais', i*2)
                break
            
        # Update ELO after stats computed
        team_a_pts = g.iloc[[0],:].iloc[0]['PTS']
        team_b_pts = g.iloc[1:2,:].iloc[0]['PTS']
        elo_a = elo_dic[team_a_id]
        elo_b = elo_dic[team_b_id]
        
        if not (len(team_a_previous_10_games.index) >= 5 and len(team_b_previous_10_games.index) >= 5 and len(team_a_last_ha_games.index) >= 2 and len(team_b_last_ha_games.index) >= 2):
            print("Sem jogos suficientes.")
            update_elo(winner, elo_a, elo_b, elo_dic, team_a_id, team_b_id, team_a_pts, team_b_pts)
            continue
        
        teams_per[team_a_id] = get_team_per_mean(team_a_id, game_id, game_date, season_id, season_games_plyrs)
        teams_per[team_b_id] = get_team_per_mean(team_b_id, game_id, game_date, season_id, season_games_plyrs)
        
        # Season Win Percentage
        team_a_season_pct = get_wl_pct(team_a_season_games)[0]
        team_b_season_pct = get_wl_pct(team_b_season_games)[0]
        
        # Calculating Current Streak
        team_a_streak = current_streak(team_a_season_games)
        team_b_streak = current_streak(team_b_season_games)
            
        team_a_ha_percentage = get_wl_pct(team_a_last_ha_games)[0]
        team_b_ha_percentage = get_wl_pct(team_b_last_ha_games)[0]
 
        
        matches_organized.append([team_a_abbv, team_a_id, team_a_pts, team_b_pts, g.iloc[[0],:].iloc[0]['FG_PCT'], g.iloc[[0],:].iloc[0]['FG3_PCT'], 
                        g.iloc[[0],:].iloc[0]['FT_PCT'], g.iloc[[0],:].iloc[0]['REB'], g.iloc[[0],:].iloc[0]['TOV'],
                        g.iloc[[0],:].iloc[0]['BLK'], team_a_season_pct, team_a_ha_percentage, elo_a, team_a_streak,
                         teams_per[team_a_id], winner_bin])
        
        matches_organized.append([team_b_abbv, team_b_id, team_b_pts, team_a_pts, g.iloc[1:2,:].iloc[0]['FG_PCT'], g.iloc[1:2,:].iloc[0]['FG3_PCT'], 
                        g.iloc[1:2,:].iloc[0]['FT_PCT'], g.iloc[1:2,:].iloc[0]['REB'], g.iloc[1:2,:].iloc[0]['TOV'],
                        g.iloc[1:2,:].iloc[0]['BLK'], team_b_season_pct, team_b_ha_percentage, elo_b, team_b_streak,
                         teams_per[team_b_id], abs(winner_bin-1)])
        
        update_elo(winner, elo_a, elo_b, elo_dic, team_a_id, team_b_id, team_a_pts, team_b_pts)
    
    final_df = pd.DataFrame(matches_organized, columns=['TEAM_ABBV', 'TEAM_ID',
                                                        'PTS_A', 'PTS_CON_A', 'FG_PCT_A', 'FG3_PCT_A', 'FT_PCT_A', 'REB_A', 'TOV_A', 'BLK_A', 'SEASON_A_PCT', 'H/A_A', 'ELO_A', 'STREAK_A', 'PER_A',
                                                        'WINNER'])
    final_df.to_csv('../data/seasons/winner/LSTM/{}-{}.csv'.format(first_season, last_season-1))