import pandas as pd
import numpy as np

df = pd.read_csv('data/nfl_play_by_play_2009_2018.csv')

df = df[[
    'play_id',
    'game_id',
    'home_team',
    'away_team',
    'posteam',
    'posteam_type',
    'defteam',
    # 'side_of_field',
    'yardline_100',
    'game_date',
    'quarter_seconds_remaining',
    'game_seconds_remaining',
    'game_half',
    'qtr',
    'yrdln',
    'play_type',
    'field_goal_result',
    'kick_distance',
    'home_timeouts_remaining', # timeouts remaining is post timeout taken
    'away_timeouts_remaining',
    'timeout',
    'timeout_team',
    # 'posteam_timeouts_remaining',
    'defteam_timeouts_remaining',
    # 'total_home_score',
    # 'total_away_score',
    'posteam_score',
    'defteam_score',
    'fg_prob',
    'field_goal_attempt',
    'kicker_player_name',
    'kicker_player_id'
]]

df['prev_play_type'] = df.groupby('game_id')['play_type'].shift(1)

df['prev_play_id'] = df.groupby('game_id')['play_id'].shift(1)
df['prev_play_timeout'] = df.groupby('game_id')['timeout'].shift(1)
df['prev_play_timeout_team'] = df.groupby('game_id')['timeout_team'].shift(1)

# Filter to field goals in final minute of either half
fg = df.query('field_goal_attempt == 1 & qtr.isin([2, 4]) & quarter_seconds_remaining < 60')

# Create boolean for if defensive team called timeout
fg['is_def_team_timeout'] = np.where(fg.prev_play_timeout == 1 & (fg.prev_play_timeout_team != fg.posteam), 1, 0)