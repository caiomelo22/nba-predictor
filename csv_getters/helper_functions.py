from statistics import mean
from datetime import datetime, timedelta
import pandas as pd
import os.path

def get_k(vic_margin, elo_diff_winner):
    return 20*((vic_margin+3)**0.8)/(7.5 + 0.006*elo_diff_winner)

def get_e_team(team_elo, opp_team_elo):
    return 1/(1+10**((opp_team_elo - team_elo)/400))

def reset_season_elo(elo_dic):
    for k, v in elo_dic.items():
        elo_dic[k] = round(v*0.75 + 0.25*1505, 2)
                
def update_elo(winner, elo_a, elo_b, elo_dic, team_a_id, team_b_id, team_a_pts, team_b_pts):
    if winner == 'H':
        vic_margin = team_a_pts - team_b_pts
        elo_diff_winner = elo_a - elo_b
        elo_dic[team_a_id] = round(get_k(vic_margin, elo_diff_winner)*(1 - get_e_team(elo_a, elo_b)) + elo_a, 2)
        elo_dic[team_b_id] = round(get_k(vic_margin, elo_diff_winner)*(0 - get_e_team(elo_b, elo_a)) + elo_b, 2)
    else:
        vic_margin = team_b_pts - team_a_pts
        elo_diff_winner = elo_b - elo_a
        elo_dic[team_a_id] = round(get_k(vic_margin, elo_diff_winner)*(0 - get_e_team(elo_a, elo_b)) + elo_a, 2)
        elo_dic[team_b_id] = round(get_k(vic_margin, elo_diff_winner)*(1 - get_e_team(elo_b, elo_a)) + elo_b, 2)


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
        perValues.append((game['fgm'] * 85.910 + game['stl'] * 53.897 + game['fg3m'] * 51.757 + game['ftm'] * 46.845 + game['blk'] * 39.190 + game['oreb'] * 39.190 + game['ast'] * 34.677 + game['dreb'] * 14.707
                          - game['pf'] * 17.174 - (game['fta'] - game['ftm']) * 20.091 - (game['fga'] - game['fgm'])* 39.190 - game['tov'] * 53.897 ) * (1 / game['minutes']))
    if len(perValues) > 0:
        return mean(perValues)
    return 0
    
def get_team_per_mean(teamId, gameId, gameDate, seasonId, seasonAllPlayers):
    gamePlayers = seasonAllPlayers.loc[(seasonAllPlayers['game_id'] == gameId) & (seasonAllPlayers['team_id'] == teamId)].nlargest(5, 'minutes')
    seasonPlayers = seasonAllPlayers.loc[(seasonAllPlayers['date'] < gameDate) & (seasonAllPlayers['team_id'] == teamId) & (seasonAllPlayers['season'] == seasonId) & (seasonAllPlayers['minutes'] > 0)]
    perValues = []
    for index, player in gamePlayers.iterrows():
        playerLastTenGames = seasonPlayers.loc[seasonPlayers['player_id'] == player['player_id']].iloc[-10:]
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

def get_team_possessions(game, scenario):
    if scenario == 'H':
        return game['home_fga'] - game['home_oreb'] + game['home_tov'] + (0.4 * game['home_fta'])
    else:
        return game['away_fga'] - game['away_oreb'] + game['away_tov'] + (0.4 * game['away_fta'])

def get_team_offensive_rating_game(game, scenario):
    possessions = get_team_possessions(game, scenario)
    return ((game['home_pts'] / possessions) * 100) if scenario == 'H' else ((game['away_pts'] / possessions) * 100)

def get_team_defensive_rating_game(game, scenario):
    possessions = get_team_possessions(game, scenario)
    return ((game['away_pts'] / possessions) * 100) if scenario == 'H' else ((game['home_pts'] / possessions) * 100)
    
def get_team_stats (previous_games, season_pct, per, elo, matchup_pct, ha_pct, streak, pct_last_n_games, ha_pct_last_n_games, totals_overall_pct, totals_ha_pct):
    return [previous_games['team_pts'].mean(), previous_games['opp_pts'].mean(), previous_games['team_fg_pct'].mean(), previous_games['team_fg3_pct'].mean(), previous_games['team_ft_pct'].mean(), previous_games['team_reb'].mean(), previous_games['team_tov'].mean(), season_pct, per, elo, matchup_pct, ha_pct, streak, pct_last_n_games, ha_pct_last_n_games, previous_games['team_off_rtg'].mean(), previous_games['team_def_rtg'].mean(), totals_overall_pct, totals_ha_pct]

def get_team_previous_games(season_games, team_id, game_date, season):
    home_previous_games = season_games.loc[(season_games['home_id'] == team_id) & (season_games['date'] < game_date)]
    away_previous_games = season_games.loc[(season_games['away_id'] == team_id) & (season_games['date'] < game_date)]
    
    if len(home_previous_games.index) == 0 or len(away_previous_games.index) == 0:
        return None
    
    home_previous_games.rename(columns = {'home_id': 'team_id', 'home_name': 'team_name',
       'home_pts': 'team_pts', 'home_fgm': 'team_fgm', 'home_fga': 'team_fga', 'home_fg_pct': 'team_fg_pct', 'home_fg3m': 'team_fg3m',
       'home_fg3a': 'team_fg3a', 'home_fg3_pct': 'team_fg3_pct', 'home_ftm': 'team_ftm', 'home_fta': 'team_fta', 'home_ft_pct': 'team_ft_pct',
       'home_oreb': 'team_oreb', 'home_dreb': 'team_dreb', 'home_reb': 'team_reb', 'home_ast': 'team_ast', 'home_stl': 'team_stl',
       'home_blk': 'team_blk', 'home_tov': 'team_tov', 'home_pf': 'team_pf',
       'home_off_rtg': 'team_off_rtg', 'home_def_rtg': 'team_def_rtg',
                                          
       'away_id': 'opp_id', 'away_name': 'opp_name', 'away_pts': 'opp_pts',
       'away_fgm': 'opp_fgm', 'away_fga': 'opp_fga', 'away_fg_pct': 'opp_fg_pct', 'away_fg3m': 'opp_fg3m', 'away_fg3a': 'opp_fg3a',
       'away_fg3_pct': 'opp_fg3_pct', 'away_ftm': 'opp_ftm', 'away_fta': 'opp_fta', 'away_ft_pct': 'opp_ft_pct', 'away_oreb': 'opp_oreb',
       'away_dreb': 'opp_dreb', 'away_reb': 'opp_reb', 'away_ast': 'opp_ast', 'away_stl': 'opp_stl', 
       'away_blk': 'opp_blk', 'away_tov': 'opp_tov', 'away_pf': 'opp_pf', 
       'away_off_rtg': 'opp_off_rtg', 'away_def_rtg': 'opp_def_rtg',
                                          
       'home_odds': 'team_odds', 'away_odds': 'opp_odds'}, inplace=True)
    home_previous_games['scenario'] = 'H'
    home_previous_games['WL'] = home_previous_games.apply(lambda row: 'W' if row.winner == row.scenario else 'L', axis=1)
    
    away_previous_games.rename(columns = {'away_id': 'team_id', 'away_name': 'team_name',
       'away_pts': 'team_pts', 'away_fgm': 'team_fgm', 'away_fga': 'team_fga', 'away_fg_pct': 'team_fg_pct', 'away_fg3m': 'team_fg3m',
       'away_fg3a': 'team_fg3a', 'away_fg3_pct': 'team_fg3_pct', 'away_ftm': 'team_ftm', 'away_fta': 'team_fta', 'away_ft_pct': 'team_ft_pct',
       'away_oreb': 'team_oreb', 'away_dreb': 'team_dreb', 'away_reb': 'team_reb', 'away_ast': 'team_ast', 'away_stl': 'team_stl',
       'away_blk': 'team_blk', 'away_tov': 'team_tov', 'away_pf': 'team_pf', 
       'away_off_rtg': 'team_off_rtg', 'away_def_rtg': 'team_def_rtg',
                                          
       'home_id': 'opp_id', 'home_name': 'opp_name', 'home_pts': 'opp_pts',
       'home_fgm': 'opp_fgm', 'home_fga': 'opp_fga', 'home_fg_pct': 'opp_fg_pct', 'home_fg3m': 'opp_fg3m', 'home_fg3a': 'opp_fg3a',
       'home_fg3_pct': 'opp_fg3_pct', 'home_ftm': 'opp_ftm', 'home_fta': 'opp_fta', 'home_ft_pct': 'opp_ft_pct', 'home_oreb': 'opp_oreb',
       'home_dreb': 'opp_dreb', 'home_reb': 'opp_reb', 'home_ast': 'opp_ast', 'home_stl': 'opp_stl', 
       'home_blk': 'opp_blk', 'home_tov': 'opp_tov', 'home_pf': 'opp_pf',
       'home_off_rtg': 'opp_off_rtg', 'home_def_rtg': 'opp_def_rtg',
                                          
       'home_odds': 'opp_odds', 'away_odds': 'team_odds'}, inplace=True)
    away_previous_games['scenario'] = 'A'
    away_previous_games['WL'] = away_previous_games.apply(lambda row: 'W' if row.winner == row.scenario else 'L', axis=1)
    
    previous_games = pd.concat([home_previous_games, away_previous_games], axis=0, ignore_index=True)
    previous_games.sort_values('date', inplace=True)
    
    previous_season_games = previous_games.loc[previous_games['season'] == season]
    home_previous_season_games = home_previous_games.loc[home_previous_games['season'] == season]
    away_previous_season_games = away_previous_games.loc[away_previous_games['season'] == season]
    
    return home_previous_games, away_previous_games, previous_games, previous_season_games, home_previous_season_games, away_previous_season_games
