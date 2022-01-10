# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:58:14 2021

@author: caiog
"""

from statistics import mean
from datetime import datetime, timedelta
import pandas as pd
import os.path

def get_k(vic_margin, elo_diff_winner):
    return 20*((vic_margin+3)**0.8)/(7.5 + 0.006*elo_diff_winner)

def get_e_team(team_elo, opp_team_elo):
    return 1/(1+10**((opp_team_elo - team_elo)/400))

def reset_season_elo(season_id, g, elo_dic):
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
    gamePlayers = seasonAllPlayers.loc[(seasonAllPlayers['GAME_ID'] == gameId) & (seasonAllPlayers['TEAM_ID'] == teamId)].nlargest(5, 'MIN')
    seasonPlayers = seasonAllPlayers.loc[(seasonAllPlayers['GAME_DATE'] < gameDate) & (seasonAllPlayers['TEAM_ID'] == teamId) & (seasonAllPlayers['SEASON_ID'] == seasonId) & (seasonAllPlayers['MIN'] > 0)]
    perValues = []
    for index, player in gamePlayers.iterrows():
        playerLastTenGames = seasonPlayers.loc[seasonPlayers['PLAYER_ID'] == player['PLAYER_ID']].iloc[-10:]
        perValues.append(get_player_mean_per(playerLastTenGames))
    if len(perValues) > 0:
        return mean(perValues)
    else:
        return 0
    
def load_bets_csv():
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, '../data/odds.csv')
    dataset = pd.read_csv(path)
    dataset['GAME_DATE'] = pd.to_datetime(dataset['GAME_DATE']).dt.normalize()
    return dataset

def get_teams_odds(team_a_id, team_b_id, game_date, season_odds):
    try:
        game = season_odds[(season_odds['TEAM_A_ID'] == team_a_id) & (season_odds['TEAM_B_ID'] == team_b_id) & (game_date <= season_odds['GAME_DATE']) & ((game_date + timedelta(days=2)) >= season_odds['GAME_DATE'])].iloc[0]
        # game = next(filter(lambda x: game_date.date() <= x['date'].date() and (game_date.date() + timedelta(days=2)) >= x['date'].date() and x['team_a_id'] == team_a_id and x['team_b_id'] == team_b_id, season_odds))
        return float(game['TEAM_A_ODDS']), float(game['TEAM_B_ODDS'])
    except IndexError:
        return None, None
    except ValueError:
        return None, None
    

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
    
def get_team_stats (previous_games, previous_games_pts_conceded, season_pct, ha_percentage, streak, matchups_pct, elo, per, odds):
    return [previous_games['PTS'].mean(), previous_games_pts_conceded, previous_games['FG_PCT'].mean(), previous_games['FG3_PCT'].mean(), 
                        previous_games['FT_PCT'].mean(), previous_games['REB'].mean(), previous_games['TOV'].mean(),
                        previous_games['BLK'].mean(), season_pct, ha_percentage, elo, streak, matchups_pct, per, odds]
    

def get_team_stats_regression (previous_games, previous_games_pts_conceded, season_games, elo, per, ha_previous_games, ha_previous_games_pts_conceded):
    if(ha_previous_games['PTS'].mean() == None):
        exit()
    return [previous_games['PTS'].mean(), previous_games_pts_conceded, previous_games['FT_PCT'].mean(), previous_games['FG_PCT'].mean(), previous_games['FG3_PCT'].mean(), 
            elo, per, ha_previous_games['PTS'].mean(), ha_previous_games_pts_conceded, season_games['PTS'].mean()]

if __name__ == "__main__":
    dataset = load_bets_csv()