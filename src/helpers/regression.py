from datetime import datetime
import pandas as pd
from tqdm import tqdm

def build_avg_columns(season_games_plyrs: pd.DataFrame, stats_columns: list):
    for i in tqdm(range(len(stats_columns))):
        stat = stats_columns[i]
        season_games_plyrs[f'{stat}_avg'] = (
            season_games_plyrs.sort_values(by=['player_id', 'date'])
                    .groupby('player_id')[stat]
                    .transform(lambda x: x.shift(1).rolling(window=5).mean())
        )

def build_days_since_last_game_col(season_games_plyrs: pd.DataFrame):
    season_games_plyrs['days_since_last_game'] = (
        season_games_plyrs.sort_values(by=['player_id', 'date'])
                    .groupby('player_id')['date']
                    .transform(lambda x: x.diff().dt.days.fillna(7))
    )

def get_team_stats(games_df, stats_columns):
    # Combine home and away stats into a unified DataFrame for conceded stats
    team_stats = pd.concat([
        games_df[['date', 'home_id'] + [f'away_{stat}' for stat in stats_columns]].rename(
            columns={**{'home_id': 'team_id'},
                    **{f'away_{stat}': f'{stat}_conceded' for stat in stats_columns}}
        ),
        games_df[['date', 'away_id'] + [f'home_{stat}' for stat in stats_columns]].rename(
            columns={**{'away_id': 'team_id'},
                    **{f'home_{stat}': f'{stat}_conceded' for stat in stats_columns}}
        )
    ], ignore_index=True)

    # Sort by team_id and date
    team_stats = team_stats.sort_values(by=['team_id', 'date'])

    # Calculate rolling averages for stats conceded by each team
    for stat in stats_columns:
        conceded_stat = f'{stat}_conceded'
        rolling_stat = f'rolling_avg_{conceded_stat}'
        team_stats[rolling_stat] = (
            team_stats.groupby('team_id')[conceded_stat]
            .transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
        )

    # Rename columns dynamically for clarity
    renamed_columns = {
        f'rolling_avg_{stat}_conceded': f'opponent_avg_conceded_{stat}' for stat in stats_columns
    }
    team_stats.rename(columns=renamed_columns, inplace=True)

    return team_stats

def build_opponent_conceded_columns(season_games_plyrs: pd.DataFrame, games_df: pd.DataFrame, stats_columns: list):
    # Define the opponent_team_id column
    season_games_plyrs['opponent_team_id'] = season_games_plyrs.apply(
        lambda row: row['away_id'] if row['team_id'] == row['home_id'] else row['home_id'],
        axis=1
    )

    # Combine home and away stats into a unified DataFrame for conceded stats
    team_stats = get_team_stats(games_df, stats_columns)

    # Map opponent stats back to the players DataFrame
    # First, merge opponent stats into the players DataFrame using opponent_team_id
    rolling_columns = [f'opponent_avg_conceded_{stat}' for stat in stats_columns]
    season_games_plyrs = season_games_plyrs.merge(
        team_stats[['team_id', 'date'] + rolling_columns],
        left_on=['opponent_team_id', 'date'],  # Match on opponent and game date
        right_on=['team_id', 'date'],  # Match on team_id and date
        how='left'
    )

    return season_games_plyrs

def build_next_games_df(lineup, team, opp_team, season_games_plyrs, stats_columns, games_df):
    players_data = []
    
    # Get the rolling averages for the opponent's conceded stats
    team_stats = get_team_stats(games_df, stats_columns)
    
    # Find the row for the opponent's last game (including current stats)
    opponent_team_stats = team_stats[team_stats["team_id"] == opp_team["id"]].sort_values(by="date")

    opponent_conceded_stats = {
        f'opponent_avg_conceded_{stat}': opponent_team_stats[f'{stat}_conceded'].iloc[-10:].mean()
        for stat in stats_columns
    }
    
    for player in lineup:
        player = player.replace("'", "")
        try:
            player_object = season_games_plyrs.loc[
                (season_games_plyrs["team_id_x"] == team["id"])
                & (
                    (season_games_plyrs["player_name"].str.contains(player))
                    | (season_games_plyrs["player_name"] == player)
                    | (
                        season_games_plyrs["player_name"].str.startswith(player[0])
                        & season_games_plyrs["player_name"].str.endswith(
                            player.split(" ")[-1]
                        )
                    )
                )
            ].iloc[-1]
            
            last_ten_games = season_games_plyrs.loc[
                (season_games_plyrs["minutes"] > 0) &
                (season_games_plyrs["player_id"] == player_object["player_id"])
            ].iloc[-10:]

            last_date = last_ten_games.iloc[-1]['date']

            now = datetime.now()
            days_difference = (now - last_date).days

            # Add player data
            player_data = {
                "player_id": player_object["player_id"],
                "lineup_name": player,
                "player_name": player_object["player_name"],
                "team_name": team["name"],
                "opp_name": opp_team["name"],
                **{
                    f"{stat}_avg": last_ten_games[stat].mean() for stat in stats_columns
                },
                **opponent_conceded_stats,
                "days_since_last_game": days_difference
            }
            
            players_data.append(player_data)
        except Exception as e:
            print(
                "Error when trying to get the games for {} of the {}: {}".format(
                    player, team["name"], e
                )
            )
            continue

    players_df = pd.DataFrame(players_data)

    return players_df