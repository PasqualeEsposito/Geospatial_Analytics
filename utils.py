import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def extract_teams(teams_df, type, area=""):
    if type == 'club':
        return teams_df[(teams_df['area'] == area) & (teams_df['type'] == 'club')]
    elif type == 'national':
        return teams_df[(teams_df['type'] == 'national')]


def prepare_teams(teams_df):
    teams_df['area'] = teams_df['area'].apply(lambda x: x.get('name'))
    return teams_df


def extract_events(events_df, area):
    events = events_df[area][events_df[area]['playerId'] != 0]
    return events


def prepare_events(events):
    events['x'] = events['positions'].apply(lambda x: x[0]['x'])
    events['y'] = events['positions'].apply(lambda x: x[0]['y'])
    events.drop(columns={'id', 'subEventName', 'positions', 'eventName'}, inplace=True)
    return events


def extract_role(role):
    return role.get('code2')


def prepare_players(players_df):
    players_df['role'] = players_df['role'].apply(extract_role)
    players_df.drop(columns={'passportArea','weight','height','foot','birthArea','birthDate','middleName','firstName','lastName', 'currentNationalTeamId', 'shortName', 'currentTeamId'},inplace=True)
    players_df.rename(columns={'wyId':'playerId'}, inplace=True)
    return players_df


def compute_distance(df):
    # Sort the dataframe by playerId, matchId, matchPeriod, gameweek, and eventSec
    df.sort_values(by=['matchId', 'playerId', 'matchPeriod', 'eventSec'], inplace=True)

    # Compute the distance between consecutive points
    df['distance'] = np.sqrt(((df.groupby(['matchId', 'playerId', 'matchPeriod'])['x'].diff()**2) + (df.groupby(['playerId', 'matchId', 'matchPeriod'])['y'].diff()**2)))

    # Fill NaN values with 0
    df['distance'].fillna(0, inplace=True)
    
    return df


def sum_distances(df):
    tmp_df = df.groupby(['playerId', 'matchId'])['distance'].sum()
    tmp_df = round(tmp_df, -2)
    return pd.DataFrame(tmp_df)


def count_distances(df):
    # Count the occurrences of each unique distance value
    distance_counts = df['distance'].value_counts().reset_index()

    # Sort the DataFrame by distance
    distance_counts = distance_counts.sort_values(by='distance')

    distance_counts['distance'] = distance_counts['distance'].astype(int)

    return distance_counts


def plot_histogram(df, x, y, xlabel, ylabel, title):
    plt.figure(figsize=(15, 6))
    sns.barplot(data=df, x=x, y=y, color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()


def group_by_distance_role(df):
    # Count the occurrences of each unique distance value
    distance_counts_by_role = df[['distance', 'role']].value_counts().reset_index()

    # Sort the DataFrame by distance
    distance_counts_by_role = distance_counts_by_role.sort_values(by='distance')

    distance_counts_by_role['distance'] = distance_counts_by_role['distance'].astype(int)

    return distance_counts_by_role


def plot_histogram_per_role(df):
    plt.figure(figsize=(25, 6))
    sns.barplot(data=df, x='distance', y='count', hue='role')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title('Histogram of Distances by Role')
    plt.legend(title='Role')
    plt.xticks(rotation=45)
    plt.show()


def get_passes_before_shot(events_df, passes_before_shot):
    for index, row in events_df.iterrows():
        if row['eventId'] == 10:
            match_id = row['matchId']
            event_sec = row['eventSec']
            
            passes_before_shot[match_id, event_sec] = 0
            
            for i in range(index - 1, -1, -1):
                if events_df.at[i, 'eventId'] in [3, 8] and events_df.at[i, 'teamId'] == row['teamId']:
                    passes_before_shot[match_id, event_sec] += 1
                elif events_df.at[i, 'teamId'] != row['teamId']:
                    break


def count_passes_before_shot(df):
    passes_counts = df['Passes'].value_counts()
    passes_counts = passes_counts.sort_index()
    passes_counts = pd.DataFrame(passes_counts)
    return pd.DataFrame(passes_counts)


def extract_tags(x):
    tags = []
    for tag in x:
        tags.append(tag['id'])
    return tags