# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:03:16 2021

@author: caiog
"""

import pandas as pd
import numpy as np
from functools import reduce
from nba_api.stats.endpoints import teamplayerdashboard, leaguestandings, teamplayerdashboard, leagueleaders, teamestimatedmetrics, teamgamelog, teamgamelogs, leaguegamelog
from nba_api.stats.static import teams 
from statistics import mean
from odds import get_betting_odds
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
    team_selected_abbv = "BOS"
    first_season = 2018
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
    matches_organized_regression = []
    
    season_id = ''    
    current_season = first_season
    print('Getting odds for season {}-{}...'.format(current_season, current_season + 1))
    season_odds = get_betting_odds('{}-{}'.format(current_season, current_season + 1))
    right_matchup_baseline = 0
    right_odds_baseline = 0
    
    print("Creating CSV file of all games...")
    for i, g in team_selected_season_games.iterrows():
        print("{}/{}".format(i, len(team_selected_season_games.index)))
        if g['WL'] == None:
            break
        
        if season_id != '' and season_id != g['SEASON_ID']:
            current_season += 1
            hf.reset_season_elo(season_id, g, elo_dic)
            print('Getting odds for season {}-{}...'.format(current_season, current_season + 1))
            season_odds = get_betting_odds('{}-{}'.format(current_season, current_season + 1))
        
        season_id = g['SEASON_ID']
        game_id = g['GAME_ID']
        game_date = g['GAME_DATE']
        
        opponent = season_games.loc[~(season_games['TEAM_ABBREVIATION'] == team_selected_abbv) & (season_games['GAME_ID'] == game_id)].iloc[0]
        
        team_selected_id = g['TEAM_ID']
        opponent_id = opponent['TEAM_ID']
        
        team_selected_abbv = g['TEAM_ABBREVIATION']
        opponent_abbv = opponent['TEAM_ABBREVIATION']
        
        winner = 'B'
        
        if g['WL'] == 'W':
            winner = 'A'
            
        if '@' in g['MATCHUP']:
            opponent_odds, team_selected_odds = hf.get_teams_odds(opponent_id, team_selected_id, game_date, season_odds)
        else:
            team_selected_odds, opponent_odds = hf.get_teams_odds(team_selected_id, opponent_id, game_date, season_odds)
        
        team_selected_previous_games = team_selected_season_games.loc[team_selected_season_games['GAME_DATE'] < game_date]
        opponent_previous_games = season_games.loc[(season_games['TEAM_ID'] == opponent_id) & (season_games['GAME_DATE'] < game_date)]
        team_selected_current_season_games = team_selected_previous_games.loc[team_selected_previous_games['SEASON_ID'] == season_id]
        opponent_season_games = opponent_previous_games.loc[opponent_previous_games['SEASON_ID'] == season_id]
        
        # Getting teams last 10 games
        team_selected_previous_10_games = team_selected_current_season_games.iloc[-10:,:]
        opponent_previous_10_games = opponent_season_games.iloc[-10:,:]
        
        # Getting Home/Away information
        if '@' in g['MATCHUP']:
            team_selected_last_ha_games = team_selected_current_season_games[team_selected_current_season_games['MATCHUP'].str.contains('@')].iloc[-10:,:]
            opponent_last_ha_games = opponent_season_games[~opponent_season_games['MATCHUP'].str.contains('@')].iloc[-10:,:]
        else:
            team_selected_last_ha_games = team_selected_current_season_games[~team_selected_current_season_games['MATCHUP'].str.contains('@')].iloc[-10:,:]
            opponent_last_ha_games = opponent_season_games[opponent_season_games['MATCHUP'].str.contains('@')].iloc[-10:,:]
        
        # Getting Previous A x B Matchups
        last_matchups = team_selected_previous_games[team_selected_previous_games['MATCHUP'].str.contains(opponent_abbv)].iloc[-10:,:]
        
        if len(team_selected_previous_games.index) > 0:
            if team_selected_previous_games.iloc[-1]['GAME_ID'] == g['GAME_ID']:
                print('São iguais', i*2)
                break
            
        # Update ELO after stats computed
        team_selected_pts = g['PTS']
        opponent_pts = opponent['PTS']
        elo_a = elo_dic[team_selected_id]
        elo_b = elo_dic[opponent_id]
        
        if not (len(team_selected_previous_10_games.index) >= 5 and len(opponent_previous_10_games.index) >= 5 and len(team_selected_last_ha_games.index) >= 2 and len(opponent_last_ha_games.index) >= 2 and len(last_matchups.index) > 0):
            print("Sem jogos suficientes. Jogos A: {} // Jogos HA A: {} // Jogos B: {} // Jogos HA B: {}".format(len(team_selected_previous_10_games.index), len(opponent_previous_10_games.index), len(team_selected_last_ha_games.index), len(opponent_last_ha_games.index)))
            hf.update_elo(winner, elo_a, elo_b, elo_dic, team_selected_id, opponent_id, team_selected_pts, opponent_pts)
            continue
        
        # Getting player information
        # team_selected_per, teams_per[team_selected_id] = get_team_per_mean(team_selected_id, game_id, game_date, season_id, season_games_plyrs, teams_per[team_selected_id])
        # opponent_per, teams_per[opponent_id] = get_team_per_mean(opponent_id, game_id, game_date, season_id, season_games_plyrs, teams_per[opponent_id])
        teams_per[team_selected_id] = hf.get_team_per_mean(team_selected_id, game_id, game_date, season_id, team_selected_season_games_players)
        teams_per[opponent_id] = hf.get_team_per_mean(opponent_id, game_id, game_date, season_id, season_games_plyrs)
        
        # Season Win Percentage
        team_selected_season_pct = hf.get_wl_pct(team_selected_current_season_games)[0]
        opponent_season_pct = hf.get_wl_pct(opponent_season_games)[0]
        
        # Calculating Current Streak
        team_selected_streak = hf.current_streak(team_selected_current_season_games)
        opponent_streak = hf.current_streak(opponent_season_games)
    
        team_selected_last_matchups_percentage, opponent_last_matchups_percentage = hf.get_wl_pct(last_matchups)
        
        # Updating the matchup baseline
        team_a_last_matchups_percentage, team_b_last_matchups_percentage = hf.get_wl_pct(last_matchups)
        if (team_a_last_matchups_percentage >= team_b_last_matchups_percentage and winner == 'A') or (team_b_last_matchups_percentage > team_a_last_matchups_percentage and winner == 'B'):
            right_matchup_baseline+=1
        
        # Updating the odds baseline
        if (team_selected_odds <= opponent_odds and winner == 'A') or (opponent_odds < team_selected_odds and winner == 'B'):
            right_odds_baseline+=1
            
        team_selected_ha_percentage = hf.get_wl_pct(team_selected_last_ha_games)[0]
        opponent_ha_percentage = hf.get_wl_pct(opponent_last_ha_games)[0]
        
        # Poins Conceded
        team_selected_previous_games_pts_conceded = hf.team_points_conceded(team_selected_previous_10_games, season_games)
        opponent_previous_games_pts_conceded = hf.team_points_conceded(opponent_previous_10_games, season_games)
        
        # HA Points Conceded
        team_selected_ha_previous_games_pts_conceded = hf.team_points_conceded(team_selected_last_ha_games, season_games)
        opponent_ha_previous_games_pts_conceded = hf.team_points_conceded(opponent_last_ha_games, season_games)
            
        # Defining list of stats for each team
        stats_team_selected = hf.get_team_stats (team_selected_previous_10_games, team_selected_previous_games_pts_conceded, team_selected_season_pct, team_selected_ha_percentage, elo_a, team_selected_streak, team_selected_last_matchups_percentage, teams_per[team_selected_id], team_selected_odds)
        stats_opponent = hf.get_team_stats (opponent_previous_10_games, opponent_previous_games_pts_conceded, opponent_season_pct, opponent_ha_percentage, elo_b, opponent_streak, opponent_last_matchups_percentage, teams_per[opponent_id], opponent_odds)
            
        matches_organized.append([season_id, game_date, team_selected_abbv, opponent_abbv] + stats_team_selected + stats_opponent + [winner])
        
        stats_team_selected_regression = hf.get_team_stats_regression (team_selected_previous_10_games, team_selected_previous_games_pts_conceded, team_selected_current_season_games, elo_a, teams_per[team_selected_id], team_selected_last_ha_games, team_selected_ha_previous_games_pts_conceded)
        stats_opponent_regression = hf.get_team_stats_regression (opponent_previous_10_games, opponent_previous_games_pts_conceded, opponent_season_games, elo_b, teams_per[opponent_id], opponent_last_ha_games, opponent_ha_previous_games_pts_conceded)
            
        matches_organized_regression.append([season_id, game_date, team_selected_abbv, opponent_abbv] + stats_team_selected_regression + stats_opponent_regression + [team_selected_pts, opponent_pts])
        
        hf.update_elo(winner, elo_a, elo_b, elo_dic, team_selected_id, opponent_id, team_selected_pts, opponent_pts)
    
    print("Baseline Last Matchups: {}/{} -> {}".format(right_matchup_baseline,len(matches_organized),100*right_matchup_baseline/len(matches_organized)))
    print("Baseline Odds: {}/{} -> {}".format(right_odds_baseline,len(matches_organized),100*right_odds_baseline/len(matches_organized)))
    final_df = pd.DataFrame(matches_organized, columns=['SEASON_ID', 'GAME_DATE', 'TEAM_SELECTED', 'OPPONENT',
                                                        'PTS_A', 'PTS_CON_A', 'FG_PCT_A', 'FG3_PCT_A', 'FT_PCT_A', 'REB_A', 'TOV_A', 'BLK_A', 'SEASON_A_PCT', 'H/A_A', 'ELO_A', 'STREAK_A', 'MATCHUP_A', 'PER_A', 'ODDS_A',
                                                        'PTS_B', 'PTS_CON_B', 'FG_PCT_B', 'FG3_PCT_B', 'FT_PCT_B', 'REB_B', 'TOV_B', 'BLK_B', 'SEASON_B_PCT', 'H/A_B', 'ELO_B', 'STREAK_B', 'MATCHUP_B', 'PER_B', 'ODDS_B',
                                                        'WINNER'])
    final_df_regression = pd.DataFrame(matches_organized_regression, columns=['SEASON_ID', 'GAME_DATE', 'TEAM_SELECTED', 'OPPONENT',
                                                        'PTS_A', 'PTS_CON_A', 'FT_PCT_A', 'FG_PCT_A', 'FG3_PCT_A', 'ELO_A', 'PER_A', 'HA_PTS_A', 'HA_PTS_CON_A', 'SEASON_PTS_A',
                                                        'PTS_B', 'PTS_CON_B', 'FT_PCT_B', 'FG_PCT_B', 'FG3_PCT_B', 'ELO_B', 'PER_B', 'HA_PTS_B', 'HA_PTS_CON_B', 'SEASON_PTS_B',
                                                        'SCORE_A', 'SCORE_B'])
    final_df_regression.to_csv('../data/teams/score/{}-{}-{}.csv'.format(team_selected_abbv, first_season, last_season-1))
    final_df.to_csv('../data/teams/winner/{}-{}-{}.csv'.format(team_selected_abbv, first_season, last_season-1))