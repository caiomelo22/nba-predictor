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
    if season_id != '' and season_id != g['SEASON_ID']:
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
    
def get_team_stats (previous_games, previous_games_pts_conceded, season_games, elo, per, ha_previous_games, ha_previous_games_pts_conceded):
    if(ha_previous_games['PTS'].mean() == None):
        exit()
    return [previous_games['PTS'].mean(), previous_games_pts_conceded, previous_games['FT_PCT'].mean(), previous_games['FG_PCT'].mean(), previous_games['FG3_PCT'].mean(), 
            elo, per, ha_previous_games['PTS'].mean(), ha_previous_games_pts_conceded, season_games['PTS'].mean()]

import pandas as pd
import numpy as np
from sys import exit
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
    team_selected_abbv = "PHI"
    first_season = 2008
    last_season = 2019
    
    print("Getting NBA Seasons Information...")
    for i in range(first_season,last_season):
        season_i_teams = leaguegamelog.LeagueGameLog(season = str(i)).get_data_frames()[0]
        season_i_players = leaguegamelog.LeagueGameLog(season = str(i), player_or_team_abbreviation = 'P').get_data_frames()[0]
        seasons_teams.append(season_i_teams)
        seasons_players.append(season_i_players)
        print("{}/{}".format(i,last_season))
        
    
    dfs = []
    
    print("Cleaning the data...")
    
    season_games = reduce(lambda  left,right: pd.merge(left,right, how='outer'), seasons_teams)
    season_games_plyrs = reduce(lambda  left,right: pd.merge(left,right, how='outer'), seasons_players)
    season_games.dropna(subset=['FG_PCT','FT_PCT','FG3_PCT'], inplace=True)
    
    season_games_plyrs['GAME_ID'] = pd.to_numeric(season_games_plyrs['GAME_ID'])
    season_games['GAME_ID'] = pd.to_numeric(season_games['GAME_ID'])
    season_games['GAME_DATE'] = pd.to_datetime(season_games['GAME_DATE'])
    season_games_plyrs['GAME_DATE'] = pd.to_datetime(season_games_plyrs['GAME_DATE'])
    
    team_selected_season_games = season_games.loc[season_games['TEAM_ABBREVIATION'] == team_selected_abbv].reset_index(drop=True)
    team_selected_season_games_players = season_games_plyrs.loc[season_games_plyrs['TEAM_ABBREVIATION'] == team_selected_abbv].reset_index(drop=True)
    
    print('size', len(season_games.index))
    
    print("Initializing ELOs...")
    
    elo_dic = dict()
    
    for team in teams:
        elo_dic[team['id']] = 1500
    
    matches_organized = []
    
    season_id = ''    
    
    print("Creating CSV file of all games...")
    for i, g in team_selected_season_games.iterrows():
        print("{}/{}".format(i, len(team_selected_season_games.index)))
        if g['WL'] == None:
            break
        
        season_id = g['SEASON_ID']
        game_id = g['GAME_ID']
        game_date = g['GAME_DATE']
        
        opponent = season_games.loc[~(season_games['TEAM_ABBREVIATION'] == team_selected_abbv) & (season_games['GAME_ID'] == game_id)].iloc[0]
        
        team_selected_id = g['TEAM_ID']
        opponent_id = opponent['TEAM_ID']
        
        opponent_abbv = opponent['TEAM_ABBREVIATION']
        
        reset_season_elo(season_id, g, elo_dic)
        
        winner = 'B'
        
        if g['WL'] == 'W':
            winner = 'A'
        
        team_selected_previous_games = team_selected_season_games.loc[team_selected_season_games['GAME_DATE'] < game_date]
        opponent_previous_games = season_games.loc[(season_games['TEAM_ID'] == opponent_id) & (season_games['GAME_DATE'] < game_date)]
        team_selected_current_season_games = team_selected_previous_games.loc[team_selected_previous_games['SEASON_ID'] == season_id]
        opponent_season_games = opponent_previous_games.loc[opponent_previous_games['SEASON_ID'] == season_id]
        
        team_selected_pts = g['PTS']
        opponent_pts = opponent['PTS']
            
        elo_a = elo_dic[team_selected_id]
        elo_b = elo_dic[opponent_id]
        
        # Getting Home/Away information
        if '@' in g['MATCHUP']:
            team_selected_last_ha_games = team_selected_current_season_games[team_selected_current_season_games['MATCHUP'].str.contains('@')].iloc[-10:,:]
            opponent_last_ha_games = opponent_season_games[~opponent_season_games['MATCHUP'].str.contains('@')].iloc[-10:,:]
        else:
            team_selected_last_ha_games = team_selected_current_season_games[~team_selected_current_season_games['MATCHUP'].str.contains('@')].iloc[-10:,:]
            opponent_last_ha_games = opponent_season_games[opponent_season_games['MATCHUP'].str.contains('@')].iloc[-10:,:]
        
        # Getting teams last 10 games
        team_selected_previous_10_games = team_selected_current_season_games.iloc[-10:,:]
        opponent_previous_10_games = opponent_season_games.iloc[-10:,:]
        
        if not (len(team_selected_previous_10_games.index) >= 5 and len(opponent_previous_10_games.index) >= 5 and len(team_selected_last_ha_games.index) >= 2 and len(opponent_last_ha_games.index) >= 2):
            print("Sem jogos suficientes. Jogos A: {} // Jogos HA A: {} // Jogos B: {} // Jogos HA B: {}".format(len(team_selected_previous_10_games.index), len(opponent_previous_10_games.index), len(team_selected_last_ha_games.index), len(opponent_last_ha_games.index)))    
            
            # Update ELO after stats computed
            update_elo(winner, elo_a, elo_b, elo_dic, team_selected_id, opponent_id, team_selected_pts, opponent_pts)
            continue
        
        if len(team_selected_previous_games.index) > 0:
            if team_selected_previous_games.iloc[-1]['GAME_ID'] == g['GAME_ID']:
                print('São iguais', i*2)
                break
        
        # Getting player information
        teams_per[team_selected_id] = get_team_per_mean(team_selected_id, game_id, game_date, season_id, team_selected_season_games_players)
        teams_per[opponent_id] = get_team_per_mean(opponent_id, game_id, game_date, season_id, season_games_plyrs)
        
        # Points Conceded
        team_selected_previous_games_pts_conceded = team_points_conceded(team_selected_previous_10_games, season_games)
        opponent_previous_games_pts_conceded = team_points_conceded(opponent_previous_10_games, season_games)
        
        # HA Points Conceded
        team_selected_ha_previous_games_pts_conceded = team_points_conceded(team_selected_last_ha_games, season_games)
        opponent_ha_previous_games_pts_conceded = team_points_conceded(opponent_last_ha_games, season_games)
            
        # Defining list of stats for each team
        stats_team_selected = get_team_stats (team_selected_previous_10_games, team_selected_previous_games_pts_conceded, team_selected_current_season_games, elo_a, teams_per[team_selected_id], team_selected_last_ha_games, team_selected_ha_previous_games_pts_conceded)
        stats_opponent = get_team_stats (opponent_previous_10_games, opponent_previous_games_pts_conceded, opponent_season_games, elo_b, teams_per[opponent_id], opponent_last_ha_games, opponent_ha_previous_games_pts_conceded)
            
        matches_organized.append([season_id, team_selected_abbv, opponent_abbv] + stats_team_selected + stats_opponent + [team_selected_pts, opponent_pts])
        
        # Update ELO after stats computed
        update_elo(winner, elo_a, elo_b, elo_dic, team_selected_id, opponent_id, team_selected_pts, opponent_pts)
    
    final_df = pd.DataFrame(matches_organized, columns=['SEASON_ID', 'TEAM_SELECTED', 'OPPONENT',
                                                        'PTS_A', 'PTS_CON_A', 'FT_PCT_A', 'FG_PCT_A', 'FG3_PCT_A', 'ELO_A', 'PER_A', 'HA_PTS_A', 'HA_PTS_CON_A', 'SEASON_PTS_A',
                                                        'PTS_B', 'PTS_CON_B', 'FT_PCT_B', 'FG_PCT_B', 'FG3_PCT_B', 'ELO_B', 'PER_B', 'HA_PTS_B', 'HA_PTS_CON_B', 'SEASON_PTS_B',
                                                        'SCORE_A', 'SCORE_B'])
    final_df.to_csv('../data/teams/score/{}-{}-{}.csv'.format(team_selected_abbv, first_season, last_season-1))