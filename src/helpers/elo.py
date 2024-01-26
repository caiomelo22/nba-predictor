def get_k(vic_margin, elo_diff_winner):
    return 20*((vic_margin+3)**0.8)/(7.5 + 0.006*elo_diff_winner)

def get_e_team(team_elo, opp_team_elo):
    return 1/(1+10**((opp_team_elo - team_elo)/400))

def reset_season_elo(elo_dic):
    for k, v in elo_dic.items():
        elo_dic[k] = round(v*0.75 + 0.25*1505, 2)
                
def update_elo(winner, elo_dict, team_a_id, team_b_id, team_a_pts, team_b_pts):
    elo_a = elo_dict[team_a_id]
    elo_b = elo_dict[team_b_id]

    if winner == 'H':
        vic_margin = team_a_pts - team_b_pts
        elo_diff_winner = elo_a - elo_b
        elo_dict[team_a_id] = round(get_k(vic_margin, elo_diff_winner)*(1 - get_e_team(elo_a, elo_b)) + elo_a, 2)
        elo_dict[team_b_id] = round(get_k(vic_margin, elo_diff_winner)*(0 - get_e_team(elo_b, elo_a)) + elo_b, 2)
    else:
        vic_margin = team_b_pts - team_a_pts
        elo_diff_winner = elo_b - elo_a
        elo_dict[team_a_id] = round(get_k(vic_margin, elo_diff_winner)*(0 - get_e_team(elo_a, elo_b)) + elo_a, 2)
        elo_dict[team_b_id] = round(get_k(vic_margin, elo_diff_winner)*(1 - get_e_team(elo_b, elo_a)) + elo_b, 2)
