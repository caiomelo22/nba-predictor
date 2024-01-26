from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import teams
import pandas as pd


class NBAService:
    def __init__(self, season) -> None:
        self.season = season
        pass

    def get_season(self, season_type="Regular Season"):
        season_games = leaguegamelog.LeagueGameLog(
            season=str(self.season), season_type_all_star=season_type
        ).get_data_frames()[0]

        season_players = leaguegamelog.LeagueGameLog(
            season=str(self.season),
            player_or_team_abbreviation="P",
            season_type_all_star=season_type,
        ).get_data_frames()[0]

        season_games["IS_PLAYOFFS"] = True if season_type == "Playoffs" else False
        season_players["IS_PLAYOFFS"] = True if season_type == "Playoffs" else False

        season_games.dropna(subset=["FG_PCT", "FT_PCT", "FG3_PCT"], inplace=True)

        season_games["GAME_ID"] = pd.to_numeric(season_games["GAME_ID"])
        season_players["GAME_ID"] = pd.to_numeric(season_players["GAME_ID"])

        season_games["GAME_DATE"] = pd.to_datetime(season_games["GAME_DATE"])
        season_players["GAME_DATE"] = pd.to_datetime(season_players["GAME_DATE"])

        season_games = season_games.sort_values(
            ["GAME_DATE", "GAME_ID"], ascending=[True, True]
        ).reset_index(drop=True)
        return season_games, season_players
    
    def get_teams_df(self):
        teams_list = teams.get_teams()
        teams_df = pd.DataFrame(teams_list)

        return teams_df
