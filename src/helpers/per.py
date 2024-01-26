from statistics import mean


def get_player_mean_per(player_last_games):
    per_values = []

    for _, game in player_last_games.iterrows():
        per_values.append(
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

    if len(per_values) > 0:
        return mean(per_values)
    return 0


def get_team_per_mean(team_id, game_id, game_date, season_id, season_all_players):
    game_players = season_all_players.loc[
        (season_all_players["game_id"] == game_id)
        & (season_all_players["team_id"] == team_id)
    ].nlargest(5, "minutes")

    season_players = season_all_players.loc[
        (season_all_players["date"] < game_date)
        & (season_all_players["team_id"] == team_id)
        & (season_all_players["season"] == season_id)
        & (season_all_players["minutes"] > 0)
    ]

    per_values = []

    for _, player in game_players.iterrows():
        player_last_ten_games = season_players.loc[
            season_players["player_id"] == player["player_id"]
        ].iloc[-10:]
        per_values.append(get_player_mean_per(player_last_ten_games))

    if len(per_values) > 0:
        return mean(per_values)
    else:
        return 0


def get_realtime_team_per(season_games_plyrs, lineup, team):
    # Getting players PER
    per_values = []
    for player in lineup:
        player = player.replace("'", "")
        try:
            player_object = season_games_plyrs.loc[
                (season_games_plyrs["team_id"] == team["id"])
                & (
                    (season_games_plyrs["player_name"].str.contains(player))
                    | (season_games_plyrs["player_name"] == player)
                    | (
                        season_games_plyrs["player_name"].str.startswith(player[0])
                        & season_games_plyrs["player_name"].str.endswith(
                            player.split(" ")[1]
                        )
                    )
                )
            ].iloc[-1]
            
            last_ten_games = season_games_plyrs.loc[
                (season_games_plyrs["minutes"] > 0) &
                (season_games_plyrs["player_id"] == player_object["player_id"])
            ].iloc[-10:]
            
            per_values.append(get_player_mean_per(last_ten_games))
        except Exception as e:
            print(
                "Error when trying to get the games for {} of the {}: {}".format(
                    player, team["nickname"], e
                )
            )
            continue

    if len(per_values) > 0:
        per = mean(per_values)
    else:
        per = 0

    return per
