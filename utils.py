import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Input: 'events_df' (DataFrame), 'area' (string)
#Output: 'events' (DataFrame)
#
#The function retrieves all the events of a specific competition (the competition is given by the 'area') and it removes all the
#events which have 0 as playerId, which are mainly 'Ball out of the field'.
def extract_events(events_df, area):
    events = events_df[area][events_df[area]['playerId'] != 0]
    return events


#Input: 'events' (DataFrame)
#Output: 'events' (DataFrame)
#
#The function adds two columns to the DataFrame, 'x' and 'y', which are the x and y coordinates of the starting point of the event.
#Then, it removes the columns 'id', 'subEventName', 'positions', 'eventName' from the DataFrame and returns it.
def prepare_events(events):
    events['x'] = events['positions'].apply(lambda x: x[0]['x'])
    events['y'] = events['positions'].apply(lambda x: x[0]['y'])
    events.drop(columns={'id', 'subEventName', 'positions', 'eventName'}, inplace=True)
    return events


#Input: 'role' (dictionary)
#Output: 'role' (string)
#
#The function returns the value of the key 'code2' that is into the dictionary 'role' of the players in the players_df DataFrame.
def extract_role(role):
    return role.get('code2')


#Input: players_df (DataFrame)
#Output: players_df (DataFrame)
#
#The function makes some transformations on the players_df DataFrame. Specifically, it transforms the 'role' column into a string
#value and removes all the columns but 'wyId' and 'role'. Furthermore, it changes the name of the 'wyId' column into 'playerId'.
def prepare_players(players_df):
    players_df['role'] = players_df['role'].apply(extract_role)
    players_df.drop(columns={'passportArea','weight','height','foot','birthArea','birthDate','middleName','firstName','lastName', 'currentNationalTeamId', 'shortName', 'currentTeamId'},inplace=True)
    players_df.rename(columns={'wyId':'playerId'}, inplace=True)
    return players_df


#Input: df (DataFrame)
#Output: df (DataFrame)
#
#The function computes the distance between two points. Specifically, it computes the distance between two consecutive events that
#have the same 'matchId', 'playerId', 'matchPeriod'. Also, it transforms the distance into meters, multiplying the difference for
#1.05 and 0.65, assuming that a football pitch size is 105x65 meters. 
def compute_distance(df):
    # Sort the dataframe by playerId, matchId, matchPeriod, gameweek, and eventSec
    df.sort_values(by=['matchId', 'playerId', 'matchPeriod', 'eventSec'], inplace=True)

    # Compute the distance between consecutive points
    df['distance'] = np.sqrt((((df.groupby(['matchId', 'playerId', 'matchPeriod'])['x'].diff()*1.05)**2) + ((df.groupby(['playerId', 'matchId', 'matchPeriod'])['y'].diff()*0.65)**2)))

    # Fill NaN values with 0
    df['distance'].fillna(0, inplace=True)
    
    return df


#Input: df (DataFrame)
#Output: tmp_df (DataFrame)
#
#The function sums all values present in the 'distance' column, grouping them by the 'playerId' and 'matchId'. Also, it discretizes
#the values, rounding them to the closest hundred.
def sum_distances(df):
    tmp_df = df.groupby(['playerId', 'matchId'])['distance'].sum()
    tmp_df = round(tmp_df, -2)
    return pd.DataFrame(tmp_df)


#Input: df (DataFrame)
#Output: distance_counts (DataFrame)
#
#The function counts all the distances and returns a count of each distance traveled by the players.
def count_distances(df):
    # Count the occurrences of each unique distance value
    distance_counts = df['distance'].value_counts().reset_index()

    # Sort the DataFrame by distance
    distance_counts = distance_counts.sort_values(by='distance')

    distance_counts.rename(columns={'index': 'distance', 'distance': 'count'}, inplace=True)
    distance_counts['distance'] = distance_counts['distance'].astype(int)

    return distance_counts


#Input: df (DataFrame), x (string), y (string), xlabel (string), ylabel (string), title (string)
#
#The function plots the values present in the x and y axis of the df DataFrame.
def plot_histogram(df, x, y, xlabel, ylabel, title):     
    plt.figure(figsize=(15, 6))
    sns.barplot(data=df, x=x, y=y, color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()


#Input: df (DataFrame), x (string), y (string), xlabel (string), ylabel (string), title (string)
#
#The function plots the values present in the x and y axis of the df DataFrame.
def plot_histogram_pass_chain(df, x, y, xlabel, ylabel, title):
    plt.figure(figsize=(25, 6))
    colors = {False: '#18d17b', True: '#75bbfd'}
    sns.barplot(data=df, x='Passes', y='count', hue='duel', palette=colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title='With duels', loc='upper right')
    plt.xticks(rotation=45)
    plt.show()

#Input: df (DataFrame), x (string), y (string), xlabel (string), ylabel (string), title (string)
#
#The function plots the values present in the x and y axis of the df DataFrame.
def plot_histogram_comparison_pass_chain(df, x, y, xlabel, ylabel, title):
    plt.figure(figsize=(25, 6))
    colors = {False: '#18d17b', True: '#75bbfd'}
    sns.barplot(data=df, x='Passes', y='count', hue='Goal', palette=colors, ci=None)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title='Goal', loc='upper right')
    plt.xticks(rotation=45)
    plt.show()


#Input: df (DataFrame)
#Output: distance_counts_by_role (DataFrame)
#
#This function groups by df by distance and role and counts the number of occurrences of this pair.
def group_by_distance_role(df):
    # Count the occurrences of each unique distance value
    distance_counts_by_role = df[['distance', 'role']].value_counts().reset_index()

    # Sort the DataFrame by distance
    distance_counts_by_role = distance_counts_by_role.sort_values(by='distance')

    distance_counts_by_role['distance'] = distance_counts_by_role['distance'].astype(int)

    distance_counts_by_role.rename(columns={0: 'count'}, inplace=True)

    return distance_counts_by_role


#Input: df (DataFrame), title (string)
#
#This function plots a histogram for the distances traveled by players, divided by role.
def plot_histogram_per_role(df, title):
    plt.figure(figsize=(25, 6))
    colors = {'GK': '#00bfff', 'DF': '#ff80a6', 'MD': '#ff331a', 'FW': '#ffbf00'}
    sns.barplot(data=df, x='distance', y='count', hue='role', palette=colors, ci=None)
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend(title='Role', loc='upper right')
    plt.xticks(rotation=45)
    plt.show()

#Input: events_df (DataFrame)
#Output: passes_before_shot (dictionary)
#
#The function counts all the passes before a shot.
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


#Input: dictionary (dictionary)
#Output: passes_counts (DataFrame)
#
#The function counts the how many passes counts with the same value are present in the dictionary and returns a DataFrame with
#these values.
def count_passes_before_shot(dictionary):
    df = pd.DataFrame.from_dict(dictionary, orient='index', columns=['Passes'])
    passes_counts = df['Passes'].value_counts()
    passes_counts = passes_counts.sort_index()
    passes_counts = pd.DataFrame(passes_counts)
    passes_counts.reset_index(inplace=True)
    passes_counts.columns = ['Passes', 'count']
    return passes_counts


#Input: x (list)
#Output: tags (list)
#
#The function transforms a list of dictionaries into a list of integers.
def extract_tags(x):
    tags = []
    for tag in x:
        tags.append(tag['id'])
    return tags


#The function draws the tessellation applied for the third task of the project.
def draw_pitch_tessellation():
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


#Input: positions (list)
#Output: piece (integer)
#
#The function maps the starting point of an event into a piece of a tessellation.
def from_coords_to_tesselation(positions):
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
            

#Input: pitch (string), line (string), probabilities (list), prev_tile (integer)
# Output: 
# The function draws the pitch with the probabilities of predicting each tile.
def draw_pitch(pitch, line, probabilities, prev_tile):
    line = line
    pitch = pitch
    probability_colour = 'green'
    
    fig,ax = plt.subplots(figsize=(10.4,6.8))
    plt.xlim(-1,105)
    plt.ylim(-1,69)
    ax.axis('off') # this hides the x and y ticks

    # plot the probabilities of each tile
    left_corner = [(0, 136/3), (0, 68/3), (0,  0), (104/3, 136/3), (104/3, 68/3), (104/3, 0), (208/3, 136/3), (208/3, 68/3), (208/3, 0)]
    for i in range(len(probabilities)):
        center_x = left_corner[i][0] + 104/3 / 4
        center_y = left_corner[i][1] + 68/3 / 8

        # Create the text element with desired opacity
        text = plt.Text(center_x, center_y, str(round(probabilities[i]*100)) + '%', ha='center', va='center', color='green', alpha=1)

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
