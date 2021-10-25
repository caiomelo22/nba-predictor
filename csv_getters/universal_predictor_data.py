# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:03:16 2021

@author: caiog
"""

import pandas as pd
import numpy as np
from functools import reduce
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import teams 
import helper_functions as hf

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None  # default='warn'
    teams = teams.get_teams()
    
    teams_per = dict()
    
    for team in teams:
        team_id = team['id']
        teams_per[team_id] = 0
    
    seasons_teams = []
    seasons_players = []
    first_season = 2017
    last_season = 2018
    
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
    
    print('size', len(season_games.index))
    
    print("Initializing ELOs...")
    
    elo_dic = dict()
    
    for team in teams:
        elo_dic[team['id']] = 1500
    
    matches_organized = []
    matches_organized_lstm = []
    matches_organized_regression = []
    
    season_id = ''    
    print('Getting historical odds...')
    odds = hf.load_bets_csv()
    right_matchup_baseline = 0
    right_odds_baseline = 0
    
    print("Creating CSV file of all games...")
    for i, g in season_games.groupby(season_games.index // 2):
        print("{}/{}".format(i, len(season_games.index) // 2))
        if g.iloc[[0],:].iloc[0]['WL'] == None:
            break
        
        if season_id != '' and season_id != g.iloc[[0],:].iloc[0]['SEASON_ID']:
            hf.reset_season_elo(season_id, g, elo_dic)
        
        season_id = g.iloc[[0],:].iloc[0]['SEASON_ID']
        game_id = g.iloc[[0],:].iloc[0]['GAME_ID']
        game_date = g.iloc[[0],:].iloc[0]['GAME_DATE']
        
        team_a_id = g.iloc[[0],:].iloc[0]['TEAM_ID']
        team_b_id = g.iloc[1:2,:].iloc[0]['TEAM_ID']
        
        team_a_abbv = g.iloc[[0],:].iloc[0]['TEAM_ABBREVIATION']
        team_b_abbv = g.iloc[1:2,:].iloc[0]['TEAM_ABBREVIATION']
        
        winner = 'B'
        
        if g.iloc[[0],:].iloc[0]['WL'] == 'W':
            winner = 'A'
            
        if '@' in g.iloc[[0],:].iloc[0]['MATCHUP']:
            team_b_odds, team_a_odds = hf.get_teams_odds(team_b_id, team_a_id, game_date, odds)
        else:
            team_a_odds, team_b_odds = hf.get_teams_odds(team_a_id, team_b_id, game_date, odds)
        
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
        
        # Getting Previous A x B Matchups
        last_matchups = team_a_previous_games[team_a_previous_games['MATCHUP'].str.contains(team_a_abbv) & 
                                              team_a_previous_games['MATCHUP'].str.contains(team_b_abbv)].iloc[-10:,:]
        
        if len(team_a_previous_games.index) > 0:
            if team_a_previous_games.iloc[-1]['GAME_ID'] == g.iloc[[0],:].iloc[0]['GAME_ID']:
                print('São iguais', i*2)
                break
            
        # Update ELO after stats computed
        team_a_pts = g.iloc[[0],:].iloc[0]['PTS']
        team_b_pts = g.iloc[1:2,:].iloc[0]['PTS']
        elo_a = elo_dic[team_a_id]
        elo_b = elo_dic[team_b_id]
        
        if not (len(team_a_previous_10_games.index) >= 5 and len(team_b_previous_10_games.index) >= 5 and len(team_a_last_ha_games.index) >= 2 and len(team_b_last_ha_games.index) >= 2 and len(last_matchups.index) > 0 and team_a_odds != None and team_b_odds != None):
            print("Sem jogos suficientes. Jogos A: {} // Jogos HA A: {} // Jogos B: {} // Jogos HA B: {}".format(len(team_a_previous_10_games.index), len(team_b_previous_10_games.index), len(team_a_last_ha_games.index), len(team_b_last_ha_games.index)))
            hf.update_elo(winner, elo_a, elo_b, elo_dic, team_a_id, team_b_id, team_a_pts, team_b_pts)
            continue
        
        # Getting player information
        # team_a_per, teams_per[team_a_id] = get_team_per_mean(team_a_id, game_id, game_date, season_id, season_games_plyrs, teams_per[team_a_id])
        # team_b_per, teams_per[team_b_id] = get_team_per_mean(team_b_id, game_id, game_date, season_id, season_games_plyrs, teams_per[team_b_id])
        teams_per[team_a_id] = hf.get_team_per_mean(team_a_id, game_id, game_date, season_id, season_games_plyrs)
        teams_per[team_b_id] = hf.get_team_per_mean(team_b_id, game_id, game_date, season_id, season_games_plyrs)
        
        # Season Win Percentage
        team_a_season_pct = hf.get_wl_pct(team_a_season_games)[0]
        team_b_season_pct = hf.get_wl_pct(team_b_season_games)[0]
        
        # Calculating Current Streak
        team_a_streak = hf.current_streak(team_a_season_games)
        team_b_streak = hf.current_streak(team_b_season_games)
    
        # Updating the matchup baseline
        team_a_last_matchups_percentage, team_b_last_matchups_percentage = hf.get_wl_pct(last_matchups)
        if (team_a_last_matchups_percentage >= team_b_last_matchups_percentage and winner == 'A') or (team_b_last_matchups_percentage > team_a_last_matchups_percentage and winner == 'B'):
            right_matchup_baseline+=1
        
        # Updating the odds baseline
        if (team_a_odds <= team_b_odds and winner == 'A') or (team_b_odds < team_a_odds and winner == 'B'):
            right_odds_baseline+=1
            
        team_a_ha_percentage = hf.get_wl_pct(team_a_last_ha_games)[0]
        team_b_ha_percentage = hf.get_wl_pct(team_b_last_ha_games)[0]
        
        # Poins Conceded
        team_a_previous_games_pts_conceded = hf.team_points_conceded(team_a_previous_10_games, season_games)
        team_b_previous_games_pts_conceded = hf.team_points_conceded(team_b_previous_10_games, season_games)
        
        # HA Points Conceded
        team_a_ha_previous_games_pts_conceded = hf.team_points_conceded(team_a_last_ha_games, season_games)
        team_b_ha_previous_games_pts_conceded = hf.team_points_conceded(team_b_last_ha_games, season_games)
            
        # Defining list of stats for each team
        stats_team_a = hf.get_team_stats (team_a_previous_10_games, team_a_previous_games_pts_conceded, team_a_season_pct, team_a_ha_percentage, elo_a, team_a_streak, team_a_last_matchups_percentage, teams_per[team_a_id], team_a_odds)
        stats_team_b = hf.get_team_stats (team_b_previous_10_games, team_b_previous_games_pts_conceded, team_b_season_pct, team_b_ha_percentage, elo_b, team_b_streak, team_b_last_matchups_percentage, teams_per[team_b_id], team_b_odds)
            
        stats_team_a_regression = hf.get_team_stats_regression (team_a_previous_10_games, team_a_previous_games_pts_conceded, team_a_season_games, elo_a, teams_per[team_a_id], team_a_last_ha_games, team_a_ha_previous_games_pts_conceded)
        stats_team_b_regression = hf.get_team_stats_regression (team_b_previous_10_games, team_b_previous_games_pts_conceded, team_b_season_games, elo_b, teams_per[team_b_id], team_b_last_ha_games, team_b_ha_previous_games_pts_conceded)
            
        if '@' in g.iloc[[0],:].iloc[0]['MATCHUP']:
            matches_organized.append([season_id, game_date, team_b_abbv, team_a_abbv] + stats_team_b + stats_team_a + [1 if winner == 'B' else 0])
            matches_organized_regression.append([season_id, game_date, team_b_abbv, team_a_abbv] + stats_team_b_regression + stats_team_a_regression + [team_b_pts, team_a_pts])
        else:
            matches_organized.append([season_id, game_date, team_a_abbv, team_b_abbv] + stats_team_a + stats_team_b + [1 if winner == 'A' else 0])
            matches_organized_regression.append([season_id, game_date, team_a_abbv, team_b_abbv] + stats_team_a_regression + stats_team_b_regression + [team_a_pts, team_b_pts])
            
            
        matches_organized_lstm.append([team_a_abbv, team_a_id, game_date, team_a_pts, team_b_pts, g.iloc[[0],:].iloc[0]['FG_PCT'], g.iloc[[0],:].iloc[0]['FG3_PCT'], 
                        g.iloc[[0],:].iloc[0]['FT_PCT'], g.iloc[[0],:].iloc[0]['REB'], g.iloc[[0],:].iloc[0]['TOV'],
                        g.iloc[[0],:].iloc[0]['BLK'], team_a_season_pct, team_a_ha_percentage, elo_a, elo_b, team_a_streak,
                         teams_per[team_a_id], 1 if winner == 'A' else 0])
        
        matches_organized_lstm.append([team_b_abbv, team_b_id, game_date, team_b_pts, team_a_pts, g.iloc[1:2,:].iloc[0]['FG_PCT'], g.iloc[1:2,:].iloc[0]['FG3_PCT'], 
                        g.iloc[1:2,:].iloc[0]['FT_PCT'], g.iloc[1:2,:].iloc[0]['REB'], g.iloc[1:2,:].iloc[0]['TOV'],
                        g.iloc[1:2,:].iloc[0]['BLK'], team_b_season_pct, team_b_ha_percentage, elo_b, elo_a, team_b_streak,
                         teams_per[team_b_id],  1 if winner == 'B' else 0])
    
        
        hf.update_elo(winner, elo_a, elo_b, elo_dic, team_a_id, team_b_id, team_a_pts, team_b_pts)
    
    print("Baseline Last Matchups: {}/{} -> {}".format(right_matchup_baseline,len(matches_organized),100*right_matchup_baseline/len(matches_organized)))
    print("Baseline Odds: {}/{} -> {}".format(right_odds_baseline,len(matches_organized),100*right_odds_baseline/len(matches_organized)))
    final_df = pd.DataFrame(matches_organized, columns=['SEASON_ID', 'GAME_DATE', 'TEAM_A', 'TEAM_B',
                                                        'PTS_A', 'PTS_CON_A', 'FG_PCT_A', 'FG3_PCT_A', 'FT_PCT_A', 'REB_A', 'TOV_A', 'BLK_A', 'SEASON_A_PCT', 'H/A_A', 'ELO_A', 'STREAK_A', 'MATCHUP_A', 'PER_A', 'ODDS_A',
                                                        'PTS_B', 'PTS_CON_B', 'FG_PCT_B', 'FG3_PCT_B', 'FT_PCT_B', 'REB_B', 'TOV_B', 'BLK_B', 'SEASON_B_PCT', 'H/A_B', 'ELO_B', 'STREAK_B', 'MATCHUP_B', 'PER_B', 'ODDS_B',
                                                        'WINNER'])
    final_df_lstm = pd.DataFrame(matches_organized_lstm, columns=['TEAM_ABBV', 'TEAM_ID', 'DATE',
                                                        'PTS_A', 'PTS_CON_A', 'FG_PCT_A', 'FG3_PCT_A', 'FT_PCT_A', 'REB_A', 'TOV_A', 'BLK_A', 
                                                        'SEASON_A_PCT', 'H/A_A', 'ELO_A', 'ELO_OPP', 'STREAK_A', 'PER_A',
                                                        'WINNER'])
    final_df_regression = pd.DataFrame(matches_organized_regression, columns=['SEASON_ID', 'GAME_DATE', 'TEAM_A', 'TEAM_B',
                                                        'PTS_A', 'PTS_CON_A', 'FT_PCT_A', 'FG_PCT_A', 'FG3_PCT_A', 'ELO_A', 'PER_A', 'HA_PTS_A', 'HA_PTS_CON_A', 'SEASON_PTS_A',
                                                        'PTS_B', 'PTS_CON_B', 'FT_PCT_B', 'FG_PCT_B', 'FG3_PCT_B', 'ELO_B', 'PER_B', 'HA_PTS_B', 'HA_PTS_CON_B', 'SEASON_PTS_B',
                                                        'SCORE_A', 'SCORE_B'])
    final_df_regression.to_csv('../data/seasons/score/{}-{}.csv'.format(first_season, last_season-1))
    final_df.to_csv('../data/seasons/winner/{}-{}.csv'.format(first_season, last_season-1))
    final_df_lstm.to_csv('../data/seasons/winner/LSTM/{}-{}.csv'.format(first_season, last_season-1))