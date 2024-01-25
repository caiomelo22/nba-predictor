from statistics import mean


def current_streak(previous_games):
    if len(previous_games.index) > 0:
        previous_games["start_of_streak"] = previous_games.WL.ne(
            previous_games["WL"].shift()
        )
        previous_games["streak_id"] = previous_games["start_of_streak"].cumsum()
        previous_games["streak_counter"] = (
            previous_games.groupby("streak_id").cumcount() + 1
        )

        last_outcome = previous_games.iloc[-1]["WL"]

        if last_outcome == "W":
            return previous_games.iloc[-1, -1]
        else:
            return -1 * previous_games.iloc[-1, -1]
    else:
        return 0


def get_player_mean_per(playerLastGames):
    perValues = []
    for _, game in playerLastGames.iterrows():
        perValues.append(
            (
                game["fgm"] * 85.910
                + game["stl"] * 53.897
                + game["fg3m"] * 51.757
                + game["ftm"] * 46.845
                + game["blk"] * 39.190
                + game["oreb"] * 39.190
                + game["ast"] * 34.677
                + game["dreb"] * 14.707
                - game["pf"] * 17.174
                - (game["fta"] - game["ftm"]) * 20.091
                - (game["fga"] - game["fgm"]) * 39.190
                - game["tov"] * 53.897
            )
            * (1 / game["minutes"])
        )
    if len(perValues) > 0:
        return mean(perValues)
    return 0


def get_team_per_mean(teamId, gameId, gameDate, seasonId, seasonAllPlayers):
    gamePlayers = seasonAllPlayers.loc[
        (seasonAllPlayers["game_id"] == gameId)
        & (seasonAllPlayers["team_id"] == teamId)
    ].nlargest(5, "minutes")
    seasonPlayers = seasonAllPlayers.loc[
        (seasonAllPlayers["date"] < gameDate)
        & (seasonAllPlayers["team_id"] == teamId)
        & (seasonAllPlayers["season"] == seasonId)
        & (seasonAllPlayers["minutes"] > 0)
    ]
    perValues = []
    for _, player in gamePlayers.iterrows():
        playerLastTenGames = seasonPlayers.loc[
            seasonPlayers["player_id"] == player["player_id"]
        ].iloc[-10:]
        perValues.append(get_player_mean_per(playerLastTenGames))
    if len(perValues) > 0:
        return mean(perValues)
    else:
        return 0


def get_wl_pct(previous_games):
    if len(previous_games.index) > 0:
        wl = previous_games["WL"].value_counts(normalize=True)
        if "W" in wl and "L" in wl:
            win_pct = wl["W"]
            loss_pct = wl["L"]
        elif "W" not in wl and "L" in wl:
            win_pct = 0
            loss_pct = wl["L"]
        elif "W" in wl and "L" not in wl:
            win_pct = wl["W"]
            loss_pct = 0
        return win_pct, loss_pct
    return 0, 0


def get_team_possessions(game, scenario):
    if scenario == "H":
        return (
            game["home_fga"]
            - game["home_oreb"]
            + game["home_tov"]
            + (0.4 * game["home_fta"])
        )
    else:
        return (
            game["away_fga"]
            - game["away_oreb"]
            + game["away_tov"]
            + (0.4 * game["away_fta"])
        )


def get_team_offensive_rating_game(game, scenario):
    possessions = get_team_possessions(game, scenario)
    return (
        ((game["home_pts"] / possessions) * 100)
        if scenario == "H"
        else ((game["away_pts"] / possessions) * 100)
    )


def get_team_defensive_rating_game(game, scenario):
    possessions = get_team_possessions(game, scenario)
    return (
        ((game["away_pts"] / possessions) * 100)
        if scenario == "H"
        else ((game["home_pts"] / possessions) * 100)
    )
