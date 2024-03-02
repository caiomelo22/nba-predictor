import pandas as pd
from datetime import datetime, timedelta
from helpers.per import get_team_per_mean
from helpers.feature_engineer import (
    current_streak,
    get_wl_pct,
)


def get_team_stats(
    previous_games,
    season_pct,
    per,
    elo,
    matchup_pct,
    ha_pct,
    streak,
    pct_last_n_games,
    ha_pct_last_n_games,
    is_b2b
):
    return [
        previous_games["team_pts"].mean(),
        previous_games["opp_pts"].mean(),
        previous_games["team_fg_pct"].mean(),
        previous_games["team_fg3_pct"].mean(),
        previous_games["team_ft_pct"].mean(),
        previous_games["team_reb"].mean(),
        previous_games["team_tov"].mean(),
        season_pct,
        per,
        elo,
        matchup_pct,
        ha_pct,
        streak,
        pct_last_n_games,
        ha_pct_last_n_games,
        previous_games["team_off_rtg"].mean(),
        previous_games["team_def_rtg"].mean(),
        is_b2b
    ]


def get_team_previous_games(season_games, team_id, game_date, season):
    home_previous_games = season_games.loc[
        (season_games["home_id"] == team_id) & (season_games["date"] < game_date)
    ]
    away_previous_games = season_games.loc[
        (season_games["away_id"] == team_id) & (season_games["date"] < game_date)
    ]

    if len(home_previous_games) == 0 or len(away_previous_games) == 0:
        return None

    home_previous_games.rename(
        columns={
            "home_id": "team_id",
            "home_name": "team_name",
            "home_pts": "team_pts",
            "home_fgm": "team_fgm",
            "home_fga": "team_fga",
            "home_fg_pct": "team_fg_pct",
            "home_fg3m": "team_fg3m",
            "home_fg3a": "team_fg3a",
            "home_fg3_pct": "team_fg3_pct",
            "home_ftm": "team_ftm",
            "home_fta": "team_fta",
            "home_ft_pct": "team_ft_pct",
            "home_oreb": "team_oreb",
            "home_dreb": "team_dreb",
            "home_reb": "team_reb",
            "home_ast": "team_ast",
            "home_stl": "team_stl",
            "home_blk": "team_blk",
            "home_tov": "team_tov",
            "home_pf": "team_pf",
            "home_off_rtg": "team_off_rtg",
            "home_def_rtg": "team_def_rtg",
            "away_id": "opp_id",
            "away_name": "opp_name",
            "away_pts": "opp_pts",
            "away_fgm": "opp_fgm",
            "away_fga": "opp_fga",
            "away_fg_pct": "opp_fg_pct",
            "away_fg3m": "opp_fg3m",
            "away_fg3a": "opp_fg3a",
            "away_fg3_pct": "opp_fg3_pct",
            "away_ftm": "opp_ftm",
            "away_fta": "opp_fta",
            "away_ft_pct": "opp_ft_pct",
            "away_oreb": "opp_oreb",
            "away_dreb": "opp_dreb",
            "away_reb": "opp_reb",
            "away_ast": "opp_ast",
            "away_stl": "opp_stl",
            "away_blk": "opp_blk",
            "away_tov": "opp_tov",
            "away_pf": "opp_pf",
            "away_off_rtg": "opp_off_rtg",
            "away_def_rtg": "opp_def_rtg",
            "home_odds": "team_odds",
            "away_odds": "opp_odds",
        },
        inplace=True,
    )
    home_previous_games["scenario"] = "H"
    home_previous_games["WL"] = home_previous_games.apply(
        lambda row: "W" if row.winner == row.scenario else "L", axis=1
    )

    away_previous_games.rename(
        columns={
            "away_id": "team_id",
            "away_name": "team_name",
            "away_pts": "team_pts",
            "away_fgm": "team_fgm",
            "away_fga": "team_fga",
            "away_fg_pct": "team_fg_pct",
            "away_fg3m": "team_fg3m",
            "away_fg3a": "team_fg3a",
            "away_fg3_pct": "team_fg3_pct",
            "away_ftm": "team_ftm",
            "away_fta": "team_fta",
            "away_ft_pct": "team_ft_pct",
            "away_oreb": "team_oreb",
            "away_dreb": "team_dreb",
            "away_reb": "team_reb",
            "away_ast": "team_ast",
            "away_stl": "team_stl",
            "away_blk": "team_blk",
            "away_tov": "team_tov",
            "away_pf": "team_pf",
            "away_off_rtg": "team_off_rtg",
            "away_def_rtg": "team_def_rtg",
            "home_id": "opp_id",
            "home_name": "opp_name",
            "home_pts": "opp_pts",
            "home_fgm": "opp_fgm",
            "home_fga": "opp_fga",
            "home_fg_pct": "opp_fg_pct",
            "home_fg3m": "opp_fg3m",
            "home_fg3a": "opp_fg3a",
            "home_fg3_pct": "opp_fg3_pct",
            "home_ftm": "opp_ftm",
            "home_fta": "opp_fta",
            "home_ft_pct": "opp_ft_pct",
            "home_oreb": "opp_oreb",
            "home_dreb": "opp_dreb",
            "home_reb": "opp_reb",
            "home_ast": "opp_ast",
            "home_stl": "opp_stl",
            "home_blk": "opp_blk",
            "home_tov": "opp_tov",
            "home_pf": "opp_pf",
            "home_off_rtg": "opp_off_rtg",
            "home_def_rtg": "opp_def_rtg",
            "home_odds": "opp_odds",
            "away_odds": "team_odds",
        },
        inplace=True,
    )
    away_previous_games["scenario"] = "A"
    away_previous_games["WL"] = away_previous_games.apply(
        lambda row: "W" if row.winner == row.scenario else "L", axis=1
    )

    previous_games = pd.concat(
        [home_previous_games, away_previous_games], axis=0, ignore_index=True
    )
    previous_games.sort_values("date", inplace=True)

    previous_season_games = previous_games.loc[previous_games["season"] == season]
    home_previous_season_games = home_previous_games.loc[
        home_previous_games["season"] == season
    ]
    away_previous_season_games = away_previous_games.loc[
        away_previous_games["season"] == season
    ]

    return (
        home_previous_games,
        away_previous_games,
        previous_games,
        previous_season_games,
        home_previous_season_games,
        away_previous_season_games,
    )


def get_match_info(
    game_info, stats_team_a, stats_team_b, winner, team_a_pts, team_b_pts
):
    return game_info + stats_team_a + stats_team_b + [winner, team_a_pts, team_b_pts]


def get_game_data(
    season_games,
    season_games_plyrs,
    teams_elo_dict,
    game,
    team_id,
    opp_id,
    teams_per,
    n_last_games,
    n_last_specific_games,
    scenario,
    fetch_per = True
):
    response = get_team_previous_games(
        season_games, team_id, game["date"], game["season"]
    )
    if not response:
        return None

    (
        _,
        _,
        previous_games,
        previous_season_games,
        home_previous_season_games,
        away_previous_season_games,
    ) = response

    if len(previous_season_games.index) < n_last_games:
        return None

    last_n_games = previous_season_games.iloc[-n_last_games:, :]

    # Get last game ELO
    elo = teams_elo_dict[team_id]

    # Last n games pct
    pct_last_n_games = get_wl_pct(last_n_games)[0]

    # Getting Previous A x B Matchups
    last_matchups = previous_games[previous_games["opp_id"] == opp_id].iloc[-10:, :]

    # Getting player information
    if fetch_per:
        teams_per[team_id] = get_team_per_mean(
            team_id, game["id"], game["date"], game["season"], season_games_plyrs
        )

    # Season Win Percentage
    season_pct = get_wl_pct(previous_season_games)[0]
    
    day_before = game["date"] - timedelta(days=1)
    is_b2b = len(previous_season_games[previous_season_games["date"] >= day_before]) > 0

    # Last n/2 games pct and Season H/A Win Percentage
    if scenario == "H":
        ha_pct_last_n_games = get_wl_pct(
            home_previous_season_games.iloc[-n_last_specific_games:, :]
        )[0]
        ha_pct = get_wl_pct(home_previous_season_games)[0]
    else:
        ha_pct_last_n_games = get_wl_pct(
            away_previous_season_games.iloc[-n_last_specific_games:, :]
        )[0]
        ha_pct = get_wl_pct(away_previous_season_games)[0]

    # Matchup Win Percentage
    matchup_pct = get_wl_pct(last_matchups)[0]

    # Calculating Current Streak
    streak = current_streak(previous_season_games)

    stats_team = get_team_stats(
        last_n_games,
        season_pct,
        teams_per[team_id],
        elo,
        matchup_pct,
        ha_pct,
        streak,
        pct_last_n_games,
        ha_pct_last_n_games,
        is_b2b
    )

    return stats_team
