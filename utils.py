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


# I multiply the x and y coordinates by 1.05 and 0.65 respectively to convert the pitch from 100x100 (percentage) to 105x65
def compute_distance(df):
    # Sort the dataframe by playerId, matchId, matchPeriod, gameweek, and eventSec
    df.sort_values(by=['matchId', 'playerId', 'matchPeriod', 'eventSec'], inplace=True)

    # Compute the distance between consecutive points
    df['distance'] = np.sqrt((((df.groupby(['matchId', 'playerId', 'matchPeriod'])['x'].diff()*1.05)**2) + ((df.groupby(['playerId', 'matchId', 'matchPeriod'])['y'].diff()*0.65)**2)))

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


def plot_histogram_pass_chain(df, x, y, xlabel, ylabel, title):
    plt.figure(figsize=(25, 6))
    colors = {False: '#18d17b', True: '#75bbfd'}
    sns.barplot(data=df, x='Passes', y='count', hue='duel', palette=colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title='With duels')
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
    colors = {'GK': '#00bfff', 'DF': '#ff80a6', 'MD': '#ff331a', 'FW': '#ffbf00'}
    sns.barplot(data=df, x='distance', y='count', hue='role', palette=colors)
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title('Histogram of Distances by Role')
    plt.legend(title='Role')
    plt.xticks(rotation=45)
    plt.show()


def get_passes_before_shot(events_df):
    passes_before_shot = {}
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
    return passes_before_shot


def count_passes_before_shot(dictionary):
    df = pd.DataFrame.from_dict(dictionary, orient='index', columns=['Passes'])
    passes_counts = df['Passes'].value_counts()
    passes_counts = passes_counts.sort_index()
    passes_counts = pd.DataFrame(passes_counts)
    return pd.DataFrame(passes_counts).rename_axis('Passes').reset_index()


def extract_tags(x):
    tags = []
    for tag in x:
        tags.append(tag['id'])
    return tags


def draw_pitch_tessellation():
    fig, ax = plt.subplots()

    # Draw the large rectangle
    rect = plt.Rectangle((0, 0), 3, 3, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # Draw horizontal and vertical lines to split the large rectangle into 9 smaller rectangles
    for i in range(1, 3):
        plt.plot([0, 3], [i, i], color='black', linewidth=2)
        plt.plot([i, i], [0, 3], color='black', linewidth=2)

    # Add numbers from 0 to 8 in the smaller rectangles
    for i in range(3):
        for j in range(3):
            ax.text(i + 0.5, j + 0.5, str(i * 3 + j), ha='center', va='center', fontsize=14)

    # Set aspect of the plot to equal, so the rectangles are square
    ax.set_aspect('equal')

    # Set axis limits and remove ticks
    plt.xlim(0, 3)
    plt.ylim(0, 3)
    plt.xticks([])
    plt.yticks([])

    # Show the plot
    plt.show()


def from_coords_to_tesselation(positions):
    squares = []
    for position in positions:
        if position['x'] < 33:
            if position['y'] < 33:
                squares.append(0)
            else:
                if position['y'] < 66:
                    squares.append(1)
                else:
                    squares.append(2)
                
        elif position['x'] < 66:
            if position['y'] < 33:
                squares.append(3)
            else:
                if position['y'] < 66:
                    squares.append(4)
                else:
                    squares.append(5)
            
        else:
            if position['y'] < 33:
                squares.append(6)
            else:
                if position['y'] < 66:
                    squares.append(7)
                else:
                    squares.append(8)
    return squares
