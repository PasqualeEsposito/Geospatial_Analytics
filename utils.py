from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def extract_events(events_df, area):
    """
        Input: events_df (DataFrame), area (string)
        Output: 'events' (DataFrame)

        The function retrieves all the events of a specific competition (the competition is given by the 'area') and it removes all the
        events which have 0 as playerId, which are mainly 'Ball out of the field'.
    """
    events = events_df[area][events_df[area]['playerId'] != 0]
    return events


def prepare_events(events):
    """
        Input: events (DataFrame)
        Output: events (DataFrame)

        The function adds two columns to the DataFrame, 'x' and 'y', which are the x and y coordinates of the starting point of the event.
        Then, it removes the columns 'id', 'subEventName', 'positions', 'eventName' from the DataFrame and returns it.
    """
    events['x'] = events['positions'].apply(lambda x: x[0]['x'])
    events['y'] = events['positions'].apply(lambda x: x[0]['y'])
    events.drop(columns={'id', 'subEventName', 'positions', 'eventName'}, inplace=True)
    return events


def extract_role(role):
    """
        Input: role (dictionary)
        Output: role (string)

        The function returns the value of the key 'code2' that is into the dictionary 'role' of the players in the players_df DataFrame.
    """
    return role.get('code2')


def prepare_players(players_df):
    """
        Input: players_df (DataFrame)
        Output: players_df (DataFrame)

        The function makes some transformations on the players_df DataFrame. Specifically, it transforms the 'role' column into a string
        value and removes all the columns but 'wyId' and 'role'. Furthermore, it changes the name of the 'wyId' column into 'playerId'.
    """
    players_df['role'] = players_df['role'].apply(extract_role)
    players_df.drop(columns={'passportArea','weight','height','foot','birthArea','birthDate','middleName','firstName','lastName', 'currentNationalTeamId', 'shortName', 'currentTeamId'},inplace=True)
    players_df.rename(columns={'wyId':'playerId'}, inplace=True)
    return players_df


def compute_distance(df):
    """
        Input: df (DataFrame)
        Output: df (DataFrame)

        The function computes the distance between two points. Specifically, it computes the distance between two consecutive events that
        have the same 'matchId', 'playerId', 'matchPeriod'. Also, it transforms the distance into meters, multiplying the difference for
        1.05 and 0.65, assuming that a football pitch size is 105x65 meters. 
    """
    # Sort the dataframe by playerId, matchId, matchPeriod, gameweek, and eventSec
    df.sort_values(by=['matchId', 'playerId', 'matchPeriod', 'eventSec'], inplace=True)

    # Compute the distance between consecutive points
    df['distance'] = np.sqrt((((df.groupby(['matchId', 'playerId', 'matchPeriod'])['x'].diff()*1.05)**2) + ((df.groupby(['playerId', 'matchId', 'matchPeriod'])['y'].diff()*0.65)**2)))

    # Fill NaN values with 0
    df['distance'].fillna(0, inplace=True)
    
    return df


def sum_distances(df):
    """
        Input: df (DataFrame)
        Output: tmp_df (DataFrame)

        The function sums all values present in the 'distance' column, grouping them by the 'playerId' and 'matchId'. Also, it discretizes
        the values, rounding them to the closest hundred.
    """
    tmp_df = df.groupby(['playerId', 'matchId'])['distance'].sum()
    tmp_df = round(tmp_df, -2)
    return pd.DataFrame(tmp_df)


def count_distances(df):
    """"
        Input: df (DataFrame)
        Output: distance_counts (DataFrame)

        The function counts all the distances and returns a count of each distance traveled by the players.
    """
    # Count the occurrences of each unique distance value
    distance_counts = df['distance'].value_counts().reset_index()

    # Sort the DataFrame by distance
    distance_counts = distance_counts.sort_values(by='distance')

    distance_counts['distance'] = distance_counts['distance'].astype(int)

    return distance_counts


def plot_histogram(df, x, y, xlabel, ylabel, title):
    """
        Input: df (DataFrame), x (string), y (string), xlabel (string), ylabel (string), title (string)

        The function plots the values present in the x and y axis of the df DataFrame.
    """
    plt.figure(figsize=(15, 6))
    sns.barplot(data=df, x=x, y=y, color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()


def plot_histogram_pass_chain(df, x, y, xlabel, ylabel, title):
    """
        Input: df (DataFrame), x (string), y (string), xlabel (string), ylabel (string), title (string)

        The function plots the values present in the x and y axis of the df DataFrame.
    """
    plt.figure(figsize=(25, 6))
    colors = {False: '#18d17b', True: '#75bbfd'}
    sns.barplot(data=df, x='Passes', y='count', hue='duel', palette=colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title='With duels', loc='upper right')
    plt.xticks(rotation=45)
    plt.show()


def plot_histogram_comparison_pass_chain(df, x, y, xlabel, ylabel, title):
    """
        Input: df (DataFrame), x (string), y (string), xlabel (string), ylabel (string), title (string)

        The function plots the values present in the x and y axis of the df DataFrame.
    """
    plt.figure(figsize=(25, 6))
    colors = {False: '#18d17b', True: '#75bbfd'}
    sns.barplot(data=df, x='Passes', y='count', hue='Goal', palette=colors, ci=None)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title='Goal', loc='upper right')
    plt.xticks(rotation=45)
    plt.show()


def group_by_distance_role(df):
    """
        Input: df (DataFrame)
        Output: distance_counts_by_role (DataFrame)

        This function groups by df by distance and role and counts the number of occurrences of this pair.

    """
    # Count the occurrences of each unique distance value
    distance_counts_by_role = df[['distance', 'role']].value_counts().reset_index()

    # Sort the DataFrame by distance
    distance_counts_by_role = distance_counts_by_role.sort_values(by='distance')

    distance_counts_by_role['distance'] = distance_counts_by_role['distance'].astype(int)

    distance_counts_by_role.rename(columns={0: 'count'}, inplace=True)

    return distance_counts_by_role


def plot_histogram_per_role(df, title):
    """
        Input: df (DataFrame), title (string)

        This function plots a histogram for the distances traveled by players, divided by role.
    """
    plt.figure(figsize=(25, 6))
    colors = {'GK': '#00bfff', 'DF': '#ff80a6', 'MD': '#ff331a', 'FW': '#ffbf00'}
    sns.barplot(data=df, x='distance', y='count', hue='role', palette=colors, ci=None)
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend(title='Role', loc='upper right')
    plt.xticks(rotation=45)
    plt.show()


def get_passes_before_shot(events_df, matches_df):
    """
        Input: events_df (DataFrame)
        Output: passes_before_shot (dictionary)

        The function counts all the passes before a shot.
    """
    passes_before_shot = {}
    for index, row in events_df.iterrows():
        if row['eventId'] == 10:
            match_id = row['matchId']
            event_sec = row['eventSec']
            team_id = row['teamId']
            
            passes_before_shot[match_id, team_id, event_sec, matches_df[matches_df['wyId'] == match_id]['winner'].iloc[0]] = 0

            for i in range(index - 1, -1, -1):
                if events_df.at[i, 'eventId'] in [3, 8] and events_df.at[i, 'teamId'] == row['teamId']:
                    passes_before_shot[match_id, team_id, event_sec, matches_df[matches_df['wyId'] == match_id]['winner'].iloc[0]] += 1
                elif events_df.at[i, 'teamId'] != row['teamId']:
                    break
    return passes_before_shot


def count_passes_before_shot(dictionary):
    """
        Input: dictionary (dictionary)
        Output: passes_counts (DataFrame)

        The function counts the how many passes counts with the same value are present in the dictionary and returns a DataFrame with
        these values.
    """
    df = pd.DataFrame.from_dict(dictionary, orient='index', columns=['Passes'])
    passes_counts = df['Passes'].value_counts()
    passes_counts = passes_counts.sort_index()
    passes_counts = pd.DataFrame(passes_counts)
    passes_counts.reset_index(inplace=True)
    passes_counts.columns = ['Passes', 'count']
    return passes_counts


def extract_tags(x):
    """
        Input: x (list)
        Output: tags (list)

        The function transforms a list of dictionaries into a list of integers.
    """
    tags = []
    for tag in x:
        tags.append(tag['id'])
    return tags


def draw_pitch_tessellation():
    """
        The function draws the tessellation applied for the third task of the project.
    """
    fig, ax = plt.subplots()

    rect = plt.Rectangle((0, 0), 3, 3, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    for i in range(1, 3):
        plt.plot([0, 3], [i, i], color='black', linewidth=2)
        plt.plot([i, i], [0, 3], color='black', linewidth=2)

    for i in range(3):
        for j in range(3):
            ax.text(i + 0.5, j + 0.5, str(i * 3 + j), ha='center', va='center', fontsize=14)

    ax.set_aspect('equal')

    plt.xlim(0, 3)
    plt.ylim(0, 3)
    plt.xticks([])
    plt.yticks([])

    plt.show() 


def from_coords_to_tesselation(positions):
    """
        Input: positions (list)
        Output: piece (integer)

        The function maps the starting point of an event into a piece of a tessellation.
    """
    position = positions[0]
    if position['x'] < 33:
        if position['y'] < 33:
            return 0
        else:
            if position['y'] < 66:
                return 1
            else:
                return 2
            
    elif position['x'] < 66:
        if position['y'] < 33:
            return 3
        else:
            if position['y'] < 66:
                return 4
            else:
                return 5
        
    else:
        if position['y'] < 33:
            return 6
        else:
            if position['y'] < 66:
                return 7
            else:
                return 8
            

def draw_pitch(pitch, line, probabilities, prev_tile):
    """
        Input: pitch (string), line (string), probabilities (list), prev_tile (integer)

        The function draws the pitch with the probabilities of predicting each tile.
    """
    line = line
    pitch = pitch
    probability_colour = 'green'
    
    fig,ax = plt.subplots(figsize=(10.4,6.8))
    plt.xlim(-1,105)
    plt.ylim(-1,69)
    ax.axis('off')

    # plot the probabilities of each tile
    left_corner = [(0, 136/3), (0, 68/3), (0,  0), (104/3, 136/3), (104/3, 68/3), (104/3, 0), (208/3, 136/3), (208/3, 68/3), (208/3, 0)]
    for i in range(len(probabilities)):
        center_x = left_corner[i][0] + 104/3 / 4
        center_y = left_corner[i][1] + 68/3 / 8

        # Create the text element with desired opacity
        text = plt.Text(center_x, center_y, str(round(probabilities[i]*100)) + '%', ha='center', va='center', color='black', alpha=1)

        rec_prob = plt.Rectangle(left_corner[i], 104/3, 68/3 ,ls='-',color='green', zorder=2, alpha=probabilities[i])

        if(i == prev_tile):
            txt = plt.Text(left_corner[i][0] + 104/3 / 2, left_corner[i][1] + 68/3 / 2, 'Previous tile', ha='center', va='center', color='Black', alpha=1)
            ax.add_artist(txt)
        ax.add_artist(rec_prob)
        ax.add_artist(text)

    # side and goal lines #
    ly1 = [0,0,68,68,0]
    lx1 = [0,104,104,0,0]

    plt.plot(lx1,ly1,color=line,zorder=5)

    # boxes, 6 yard box and goals

        #outer boxes#
    ly2 = [13.84,13.84,54.16,54.16] 
    lx2 = [104,87.5,87.5,104]
    plt.plot(lx2,ly2,color=line,zorder=5)

    ly3 = [13.84,13.84,54.16,54.16] 
    lx3 = [0,16.5,16.5,0]
    plt.plot(lx3,ly3,color=line,zorder=5)

        #goals#
    ly4 = [30.34,30.34,37.66,37.66]
    lx4 = [104,104.2,104.2,104]
    plt.plot(lx4,ly4,color=line,zorder=5)

    ly5 = [30.34,30.34,37.66,37.66]
    lx5 = [0,-0.2,-0.2,0]
    plt.plot(lx5,ly5,color=line,zorder=5)


        #6 yard boxes#
    ly6 = [24.84,24.84,43.16,43.16]
    lx6 = [104,99.5,99.5,104]
    plt.plot(lx6,ly6,color=line,zorder=5)

    ly7 = [24.84,24.84,43.16,43.16]
    lx7 = [0,4.5,4.5,0]
    plt.plot(lx7,ly7,color=line,zorder=5)

    #Halfway line, penalty spots, and kickoff spot
    ly8 = [0,68] 
    lx8 = [52,52]
    plt.plot(lx8,ly8,color=line,zorder=5)


    plt.scatter(93,34,color=line,zorder=5)
    plt.scatter(11,34,color=line,zorder=5)
    plt.scatter(52,34,color=line,zorder=5)

    circle1 = plt.Circle((93.5,34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=1,alpha=1)
    circle2 = plt.Circle((10.5,34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=1,alpha=1)
    circle3 = plt.Circle((52, 34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=2,alpha=1)

    ## Rectangles in boxes
    rec1 = plt.Rectangle((87.5,20), 16,30,ls='-',color=pitch, zorder=1,alpha=1)
    rec2 = plt.Rectangle((0, 20), 16.5,30,ls='-',color=pitch, zorder=1,alpha=1)

    ## Pitch rectangle
    rec3 = plt.Rectangle((-1, -1), 106,70,ls='-',color=pitch, zorder=1,alpha=1)

    ax.add_artist(rec3)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(rec1)
    ax.add_artist(rec2)
    ax.add_artist(circle3)


def winning_probability_plot(df, passes_value):
    """
        Input: df (DataFrame), passes_value (integer)
        Output: value_counts (DataFrame)

        The function plots the distribution of won matches related to the length of the pass chain.
    """
    grouped = df.groupby('matchId')['passes'].agg(lambda x: 0 if (x < passes_value).all() else (2 if (x > passes_value).all() else 1)).reset_index()

    # Rename columns and display the new DataFrame
    grouped.columns = ['matchId', 'class']

    value_counts = grouped['class'].value_counts()

    # Plot the results
    value_counts.plot(kind='bar', color='skyblue', figsize=(15, 6), rot=0)

    # Add labels and title
    txt="\n\nThe class 0 represents the matches won by the teams that scored all the match goals after a short pass chain, the class 1 represents\n" + \
        "the matches won by the teams that scored at least a match goal after a short pass chain and at least a match goal after a long pass\n" + \
        " chain, the class 2 represents the matches won by the teams that scored all the match goals after a short pass chain. The threshold of a\n" + \
        " short pass chain is a pass chain with less than %d passes." %passes_value
    plt.xlabel('Class' + txt)
    plt.ylabel('Count')
    plt.title('Distribution of won matches related to the length of the pass chain')

    # Show plot
    plt.show()
    return value_counts


def total_winning_probability_plot(df, passes_value):
    """
        Input: df (DataFrame), passes_value (integer)

        The function plots the total distribution of won matches related to the length of the pass chain. 
    """
    grouped = df.groupby(df.index)['count'].sum()
    # Rename columns and display the new DataFrame
    grouped.columns = ['class', 'count']

    # Plot the results
    grouped.plot(kind='bar', color='skyblue', figsize=(15, 6), rot=0)

    # Add labels and title
    txt="\n\nThe class 0 represents the matches won by the teams that scored all the match goals after a short pass chain, the class 1 represents\n" + \
        "the matches won by the teams that scored at least a match goal after a short pass chain and at least a match goal after a long pass\n" + \
        " chain, the class 2 represents the matches won by the teams that scored all the match goals after a short pass chain. The threshold of a\n" + \
        " short pass chain is a pass chain with less than %d passes." %passes_value
    plt.xlabel('Class' + txt)
    plt.ylabel('Count')
    plt.title('Distribution of won matches related to the length of the pass chain')

    # Show plot
    plt.show()


def winning_probability(dictionary, short_passes_value):
    """
        Input: dictionary (dictionary), short_passes_value (integer)
        Output: value_counts (DataFrame)

        The function plots the distribution of won matches related to the length of the pass chain.
    """
    keys_to_remove = []
    for key in dictionary.keys():
        if(key[1] !=  key[3]):
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del dictionary[key]

    new_list = []
    for key in dictionary.keys():
        new_list.append([key[0], key[3], dictionary[key]])

    df = pd.DataFrame(new_list, columns=['matchId', 'winner', 'passes'])

    prob = []
    for row in df.itertuples():
        if row.passes < short_passes_value:
            prob.append(True)
        else:
            prob.append(False)
    df['short_passes'] = prob

    value_counts = winning_probability_plot(df, short_passes_value)
    return value_counts


def draw_tessellation():
    """
        The function draws the tessellation of the pitch.

    """    
    line = 'black'
    pitch = 'green'
    
    fig,ax = plt.subplots(figsize=(10.4,6.8))
    plt.xlim(-1,105)
    plt.ylim(-1,69)
    ax.axis('off') # this hides the x and y ticks

    # plot the tessellations
    left_corner = [(0, 136/3), (0, 68/3), (0,  0), (104/3, 136/3), (104/3, 68/3), (104/3, 0), (208/3, 136/3), (208/3, 68/3), (208/3, 0)]
    for i in range(9):
        center_x = left_corner[i][0] + 104/3 / 2
        center_y = left_corner[i][1] + 68/3 / 2
        text = plt.Text(center_x, center_y, i, ha='center', weight='bold', va='center', color='white', alpha=1, zorder=10, fontsize=24)
        ax.add_artist(text)

    # side and goal lines #
    ly1 = [0,0,68,68,0]
    lx1 = [0,104,104,0,0]

    plt.plot(lx1,ly1,color=line,zorder=5)

    # boxes, 6 yard box and goals

        #outer boxes#
    ly2 = [13.84,13.84,54.16,54.16] 
    lx2 = [104,87.5,87.5,104]
    plt.plot(lx2,ly2,color=line,zorder=5)

    ly3 = [13.84,13.84,54.16,54.16] 
    lx3 = [0,16.5,16.5,0]
    plt.plot(lx3,ly3,color=line,zorder=5)

        #goals#
    ly4 = [30.34,30.34,37.66,37.66]
    lx4 = [104,104.2,104.2,104]
    plt.plot(lx4,ly4,color=line,zorder=5)

    ly5 = [30.34,30.34,37.66,37.66]
    lx5 = [0,-0.2,-0.2,0]
    plt.plot(lx5,ly5,color=line,zorder=5)


        #6 yard boxes#
    ly6 = [24.84,24.84,43.16,43.16]
    lx6 = [104,99.5,99.5,104]
    plt.plot(lx6,ly6,color=line,zorder=5)

    ly7 = [24.84,24.84,43.16,43.16]
    lx7 = [0,4.5,4.5,0]
    plt.plot(lx7,ly7,color=line,zorder=5)

    #Halfway line, penalty spots, and kickoff spot
    ly8 = [0,68] 
    lx8 = [52,52]
    plt.plot(lx8,ly8,color=line,zorder=5)


    plt.scatter(93,34,color=line,zorder=5)
    plt.scatter(11,34,color=line,zorder=5)
    plt.scatter(52,34,color=line,zorder=5)

    circle1 = plt.Circle((93.5,34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=1,alpha=1)
    circle2 = plt.Circle((10.5,34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=1,alpha=1)
    circle3 = plt.Circle((52, 34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=2,alpha=1)

    ## Rectangles in boxes
    rec1 = plt.Rectangle((87.5,20), 16,30,ls='-',color=pitch, zorder=1,alpha=1)
    rec2 = plt.Rectangle((0, 20), 16.5,30,ls='-',color=pitch, zorder=1,alpha=1)

    ## Pitch rectangle
    rec3 = plt.Rectangle((-1, -1), 106,70,ls='-',color=pitch, zorder=1,alpha=1)

    plt.plot([104/3, 104/3], [0, 68], color='white', zorder=5)
    plt.plot([(104/3)*2, (104/3)*2], [0, 68], color='white', zorder=5)
    plt.plot([0, 104], [68/3, 68/3], color='white', zorder=5)
    plt.plot([0, 104], [(68/3)*2, (68/3)*2], color='white', zorder=5)

    ax.add_artist(rec3)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(rec1)
    ax.add_artist(rec2)
    ax.add_artist(circle3)