import pandas as pd
import warnings
import random
import matplotlib.pyplot as plt
df = pd.read_csv("nba_games.csv", index_col=0) #reads in saved dataframe
df = df.sort_values("date") #sort df by date
df = df.reset_index(drop=True) #reset index to follow order of dates
del df["mp.1"] #delete irrelevent columns
del df["mp_opp.1"]
del df["index_opp"]
def add_target(group): #adds column that indicates if team won next game with True else false
    group["target"] = group["won"].shift(-1)
    return group

warnings.filterwarnings('ignore') #ignore warnings


#keys,empty lists, lists of teams, etc. that will be used later on in loops or for plotting
p=0
k=0
j=0
q=0
r=0
progress1=1
progress2=16
teamwins = []
team=[]
eastsrsorder=[0]
westsrsorder=[0]
eastwinsorder=[0]
eastteamsorder=[0]
westwinsorder=[0]
westteamsorder=[0]
teamslist = ["ATL","BOS","BRK","CHI","CHO","CLE","DAL","DEN","DET","GSW","HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK","OKC","ORL","PHI","PHO","POR","SAC","SAS","TOR","UTA","WAS"]
westlist = ["DAL","DEN","GSW","HOU","LAC","LAL","MEM","MIN","NOP","OKC","PHO","POR","SAC","SAS","UTA"]
eastlist = ["ATL","BOS","BRK","CHI","CHO","CLE","DET","IND","MIA","MIL","NYK","ORL","PHI","TOR","WAS"]
teamslisteastwest=["ATL","BOS","BRK","CHI","CHO","CLE","DET","IND","MIA","MIL","NYK","ORL","PHI","TOR","WAS","DAL","DEN","GSW","HOU","LAC","LAL","MEM","MIN","NOP","OKC","PHO","POR","SAC","SAS","UTA"]

szn = input("Enter desired season to simulate (2018-2021): ") #input season from user
while p == 0:
    if szn not in ["2018","2019","2020","2021"]:
        szn = input("Invalid season, try again (2018-2021): ")
    else:
        p = 1

#selects list of real life team wins for each season and list of east and west srs ratings corresponding to each team in alphabetical order
if szn == "2018":
    realteamwins=[24, 55, 28, 27, 36, 50, 24, 46, 39, 58, 65, 48, 42, 35, 22, 44, 44, 47, 48, 29, 48, 25, 52, 21, 49, 27, 47, 59, 48, 43]
    eastsrslist=[-5.3,3.23,-3.67,-6.84,.07,.59,-.26,1.18,.15,-.45,-3.53,-4.92,4.3,7.29,.53]
    westsrslist=[-2.7,1.57,5.79,8.21,.15,-1.44,-5.81,2.35,1.48,3.42,-8.8,2.6,-6.6,2.89,4.47]
elif szn == "2019":
    realteamwins=[29, 49, 42, 22, 39, 19, 33, 54, 41, 57, 53, 48, 48, 37, 33, 39, 60, 36, 33, 17, 49, 42, 51, 19, 53, 39, 48, 58, 50, 32]
    eastsrslist=[-6.06,3.9,-.4,-8.32,-1.32,-9.39,-.56,2.76,-.45,8.04,-8.93,.28,2.25,5.49,-3.3]
    westsrslist=[-.87,4.19,6.42,4.96,1.09,-1.33,-2.08,-1.02,-1.1,3.56,-8.61,4.43,-.81,1.8,5.28]
elif szn == "2020":
    realteamwins=[20, 48, 35, 22, 23, 19, 43, 46, 20, 15, 44, 45, 49, 52, 34, 44, 56, 19, 30, 21, 44, 33, 43, 34, 35, 31, 32, 53, 44, 25]
    eastsrslist=[-7.71,5.83,-1.01,-4,-7.03,-7.77,-4.38,1.63,2.59,9.41,-6.72,-.93,2.25,5.97,-5.24]
    westsrslist=[4.87,2.35,-8.12,3.13,6.66,6.28,-.91,-4.02,-.55,2.33,.56,-.61,-1.59,-.65,2.52]
elif szn == "2021":
    realteamwins=[41, 36, 48, 31, 33, 22, 42, 47, 20, 39, 17, 34, 47, 42, 38, 40, 46, 23, 31, 41, 22, 21, 49, 51, 42, 31, 33, 27, 52, 34]
    eastsrslist=[2.14,1.32,4.24,-.94,-1.94,-8.19,-4.38,-.13,-.06,5.57,2.13,-9.02,5.28,-.54,-1.85]
    westsrslist=[2.26,4.82,1.1,-7.5,6.02,2.77,1.07,-5.25,-.2,-10.13,5.67,1.81,-3.45,-1.58,8.97]

df = df.groupby("team", group_keys=False).apply(add_target) #groups df by team so target works

def backtest(data, model, predictors, start=2, step=1): #uses first 2 seasons data (2016-2017) to predict 2018 season, the all past seasons to predict next seasons 
    all_predictions = []
    
    seasons = sorted(data["season"].unique()) #creates list of all seasons we have data for
    
    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season] #train data is all data before current season
        test = data[data["season"] == season] #test data is current season
        
        model.fit(train[predictors], train["target"]) #take predictors to calculate if team wins next game
        
        preds = model.predict(test[predictors]) #make predictions on test
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1) #combine actual target and predictions
        combined.columns = ["actual", "prediction"] #label columns
        
        all_predictions.append(combined) #append combined df in list
    return pd.concat(all_predictions) #append each season df below eachother

for teamabbrev in eastlist: #for loop for each team in east in alphabetical order
    i=0
    dft = df[df["team"] == teamabbrev] #takes df of a single team
    dft["target"][pd.isnull(dft["target"])] = 2 #replaces null values with 2
    dft["target"] = dft["target"].astype(int, errors="ignore") #makes True 1 and False 0
    nulls = pd.isnull(dft).sum()
    nulls = nulls[nulls > 0] #only looks at columns where count of nulls > 0
    valid_columns = dft.columns[~dft.columns.isin(nulls.index)] #valid columns are columns with no null values
    dft = dft[valid_columns].copy()
    from sklearn.linear_model import RidgeClassifier #use ridge regression to classify if team wins following games
    from sklearn.feature_selection import SequentialFeatureSelector #used to select best features to analyze
    from sklearn.model_selection import TimeSeriesSplit #used to split data up when doing feature selection and analyzing past seasons in dataframe

    rr = RidgeClassifier(alpha=1)

    split = TimeSeriesSplit(n_splits=3)

    sfs = SequentialFeatureSelector(rr, 
                                    n_features_to_select=30, 
                                    direction="forward",
                                    cv=split,
                                    n_jobs=1
                                   ) #selects 30 best features to select from most effective to less effective

    removed_columns = ["season", "date", "won", "target", "team", "team_opp"] #remove columns not to scale
    selected_columns = dft.columns[~dft.columns.isin(removed_columns)]
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    dft[selected_columns] = scaler.fit_transform(dft[selected_columns]) #scales columns from 0 to 1 so ridge regression works better
    sfs.fit(dft[selected_columns], dft["target"]) #picking best 30 features to predict target (i.e. if team won next game)
    predictors = list(selected_columns[sfs.get_support()]) #list of the 30 best stats to predict game results

    predictions = backtest(dft, rr, predictors) #run backtest()
    predictions = predictions[predictions["actual"] != 2] #ignores games where there was null values for if team won next game True or False

    #take segment of predictions df corresp. to inputted season
    if szn == "2018":
        seasonpredictions = predictions.iloc[:82,:]
    elif szn == "2019":
        seasonpredictions = predictions.iloc[82:164,:]
    elif szn == "2020":
        seasonpredictions = predictions.iloc[164:246,:]
    elif szn == "2021":
        seasonpredictions = predictions.iloc[246:318,:] #72 games in 2021 season

    wins = seasonpredictions["prediction"].sum() #calculate total team wins in season

    print(seasonpredictions)
    print(teamabbrev,":",wins,"wins")
    print(str(progress1)+"/30 teams completed") #progress on how many teams were simulated for season
    progress1=progress1+1


    #sorts srs rating, team name, and team wins based on highest to lowest wins
    if k==0: #first time through loop, put team info in east___order list
        eastwinsorder[0]=wins
        eastteamsorder[0]=teamabbrev
        eastsrsorder[0]=eastsrslist[q] #q correlates with index of team being analyzed
        k=1
    else:
        while i < len(eastwinsorder): #runs through each index in east__order list 
            if wins >= eastwinsorder[i]: #puts new team info in where wins is greater than the one to the right
                eastwinsorder.insert(i,wins)
                eastteamsorder.insert(i,teamabbrev)
                eastsrsorder.insert(i,eastsrslist[q])
                break
            else:
                i=i+1
        if i == len(eastwinsorder): #if wins not greater than any prev. teams, append to end of list
            eastwinsorder.append(wins)
            eastteamsorder.append(teamabbrev)
            eastsrsorder.append(eastsrslist[q])

    q=q+1

    teamwins.append(wins) #append team wins and name to lists in alphabetical order, as is the eastteams loop
    team.append(teamabbrev)
    #print(eastwinsorder)
    #print(eastteamsorder)
    #print(eastsrsorder)

for teamabbrev in westlist: #same as for teamabbreviation in eastlist, but for west teams
    i=0
    q=0
    dft = df[df["team"] == teamabbrev]
    dft["target"][pd.isnull(dft["target"])] = 2
    dft["target"] = dft["target"].astype(int, errors="ignore")
    nulls = pd.isnull(dft).sum()
    nulls = nulls[nulls > 0]
    valid_columns = dft.columns[~dft.columns.isin(nulls.index)]
    dft = dft[valid_columns].copy()
    from sklearn.linear_model import RidgeClassifier
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.model_selection import TimeSeriesSplit

    rr = RidgeClassifier(alpha=1)

    split = TimeSeriesSplit(n_splits=3)

    sfs = SequentialFeatureSelector(rr, 
                                    n_features_to_select=30, 
                                    direction="forward",
                                    cv=split,
                                    n_jobs=1
                                   )
    removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
    selected_columns = dft.columns[~dft.columns.isin(removed_columns)]
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    dft[selected_columns] = scaler.fit_transform(dft[selected_columns])
    sfs.fit(dft[selected_columns], dft["target"])
    predictors = list(selected_columns[sfs.get_support()])

    predictions = backtest(dft, rr, predictors)
    predictions = predictions[predictions["actual"] != 2]

    if szn == "2018":
        seasonpredictions = predictions.iloc[:82,:]
    elif szn == "2019":
        seasonpredictions = predictions.iloc[82:164,:]
    elif szn == "2020":
        seasonpredictions = predictions.iloc[164:246,:]
    elif szn == "2021":
        seasonpredictions = predictions.iloc[246:318,:]

    wins = seasonpredictions["prediction"].sum()

    print(seasonpredictions)
    print(teamabbrev,":",wins,"wins")
    print(str(progress2)+"/30 teams completed")
    progress2=progress2+1

    if j==0:
        westwinsorder[0]=wins
        westteamsorder[0]=teamabbrev
        westsrsorder[0]=westsrslist[r]
        j=1
    else:
        while i < len(westwinsorder):
            if wins >= westwinsorder[i]:
                westwinsorder.insert(i,wins)
                westteamsorder.insert(i,teamabbrev)
                westsrsorder.insert(i,westsrslist[r])
                break
            else:
                i=i+1
        if i == len(westwinsorder):
            westwinsorder.append(wins)
            westteamsorder.append(teamabbrev)
            westsrsorder.append(westsrslist[r])

    r=r+1

    teamwins.append(wins)
    team.append(teamabbrev)
    #print(westwinsorder)
    #print(westteamsorder)
    #print(westsrsorder)

#prints respective team stats in order from team wioth highest wins to lowest
print("\n")
print("East Wins Highest to Lowest:",eastwinsorder)
print("East Standings Corresp. to Wins:",eastteamsorder)
print("East SRS Rating Corresp. to Standings:",eastsrsorder,end="\n\n")
print("West Wins Highest to Lowest:",westwinsorder)
print("West Standings Corresp. to Wins:",westteamsorder)
print("West SRS Rating Corresp. to Standings:",westsrsorder,end="\n\n")

#print(teamwins)
#print(teamslist)

plt.figure(1) #compares real vs. simulated team win totals in on bar graph
plt.subplot(2,1,1)
plt.bar(team,teamwins) #plots team and their respective simulated wins
plt.xlabel("teams")
plt.ylabel("wins")
plt.title("Predicted Team Wins for "+szn)
plt.subplot(2,1,2)
plt.bar(teamslist,realteamwins) #plots real team wins for each team
plt.xlabel("teams")
plt.ylabel("realwins")
plt.title("Real Team Wins for "+szn)
plt.subplots_adjust(left=.05, 
                    bottom=.067, 
                    right=.95, 
                    top=.963, 
                    wspace=.2, 
                    hspace=.277)
plt.show()

#empty lists that will be used to simulate playoffs by comparing srs rating (a number based on points per game, opponent points per game, strength of schedule, etc. A higher number means a team is better)
eastteamsroundtwo=[]
eastsrsroundtwo=[]
eastteamsroundthree=[]
eastsrsroundthree=[]
westteamsroundtwo=[]
westsrsroundtwo=[]
westteamsroundthree=[]
westsrsroundthree=[]
finalsteams=[]
srsfinals=[]
eastteamsroundone=[]
westteamsroundone=[]
eastsrsroundone=[]
westsrsroundone=[]

t=0
h=0

if szn in ["2020","2021"]: #Run play-in tournament (added to NBA in 2020 season). 7 and 8 seed play for 7 seed in playoffs. Loser plays winner of 9 and 10 seed for 8 seed in playoffs.
    print("East Play-In:")
    while t < 6:
        eastteamsroundone.append(eastteamsorder[t]) #round one of east playoffs is first 6 east seeds
        eastsrsroundone.append(eastsrsorder[t])
        t=t+1
    print(eastteamsorder[6],"vs.",eastteamsorder[7])
    if eastsrsorder[6] >= eastsrsorder[7]: #if 7 seed wins, they are 7 seed in round one of eastern playoffs
        eastteamsroundone.append(eastteamsorder[6])
        eastsrsroundone.append(eastsrsorder[6])
        eastloserteampi=[eastteamsorder[7]]
        eastlosersrspi=[eastsrsorder[7]]
        print(eastteamsorder[6],"wins and remains the 7 seed.",eastteamsorder[7],"plays winner of 9th and 10th seed for the 8 seed.")
    else: #else 8 seed wins and becomes 7 seed for round one of eastern playoffs
        eastteamsroundone.append(eastteamsorder[7])
        eastsrsroundone.append(eastsrsorder[7])
        eastloserteampi=[eastteamsorder[6]]
        eastlosersrspi=[eastsrsorder[6]]
        print(eastteamsorder[7],"wins and becomes the 7 seed.",eastteamsorder[6],"plays winner of 9th and 10th seed for the 8 seed.")
    print(eastteamsorder[8],"vs.",eastteamsorder[9])
    if eastsrsorder[8] >= eastsrsorder[9]: #if 9 seed wins, they play loser of original 7 and 8 seed for 8 seed in east playoffs
        eastwinnerteampi=[eastteamsorder[8]]
        eastwinnersrspi=[eastsrsorder[8]]
        print(eastteamsorder[8],"wins and plays",eastloserteampi[0],"for 8 seed.")
    else: #if 10 seed wins, they play loser of original 7 and 8 seed for 8 seed in playoffs
        eastwinnerteampi=[eastteamsorder[9]]
        eastwinnersrspi=[eastsrsorder[9]]
        print(eastteamsorder[9],"wins and plays",eastloserteampi[0],"for 8 seed.")
    print(eastwinnerteampi[0],"vs.",eastloserteampi[0])
    if eastlosersrspi[0] >= eastwinnersrspi[0]: #if original 7 and 8 seed game loser wins against winner of original 9 and 10 seed, they become 8 seed for playoffs
        eastteamsroundone.append(eastloserteampi[0])
        eastsrsroundone.append(eastlosersrspi[0])
        print(eastteamsroundone[7],"wins and is 8 seed.",end="\n\n")
    else: #if winner of original 9 and 10 seed game wins against loser of original 7 and 8 seed, they become the 8 seed for playoffs
        eastteamsroundone.append(eastwinnerteampi[0])
        eastsrsroundone.append(eastwinnersrspi[0])
        print(eastteamsroundone[7],"wins and is 8 seed.",end="\n\n")

    print("West Play-In:") #same as east play in but west teams
    while h < 6:
        westteamsroundone.append(westteamsorder[h])
        westsrsroundone.append(westsrsorder[h])
        h=h+1
    print(westteamsorder[6],"vs.",westteamsorder[7])
    if westsrsorder[6] >= westsrsorder[7]:
        westteamsroundone.append(westteamsorder[6])
        westsrsroundone.append(westsrsorder[6])
        westloserteampi=[westteamsorder[7]]
        westlosersrspi=[westsrsorder[7]]
        print(westteamsorder[6],"wins and remains the 7 seed.",westteamsorder[7],"plays winner of 9th and 10th seed for the 8 seed.")
    else:
        westteamsroundone.append(westteamsorder[7])
        westsrsroundone.append(westsrsorder[7])
        westloserteampi=[westteamsorder[6]]
        westlosersrspi=[westsrsorder[6]]
        print(westteamsorder[7],"wins and becomes the 7 seed.",westteamsorder[6],"plays winner of 9th and 10th seed for the 8 seed.")
    print(westteamsorder[8],"vs.",westteamsorder[9])
    if westsrsorder[8] >= westsrsorder[9]:
        westwinnerteampi=[westteamsorder[8]]
        westwinnersrspi=[westsrsorder[8]]
        print(westteamsorder[8],"wins and plays",westloserteampi[0],"for 8 seed.")
    else:
        westwinnerteampi=[westteamsorder[9]]
        westwinnersrspi=[westsrsorder[9]]
        print(westteamsorder[9],"wins and plays",westloserteampi[0],"for 8 seed.")
    print(westwinnerteampi[0],"vs.",westloserteampi[0])
    if westlosersrspi[0] >= westwinnersrspi[0]:
        westteamsroundone.append(westloserteampi[0])
        westsrsroundone.append(westlosersrspi[0])
        print(westteamsroundone[7],"wins and is 8 seed.",end="\n\n")
    else:
        westteamsroundone.append(westwinnerteampi[0])
        westsrsroundone.append(westwinnersrspi[0])
        print(westteamsroundone[7],"wins and is 8 seed.",end="\n\n")

else: #if no play in (i.e. 2018-2019 season) round one teams seedings are same as regular season wins order
    eastteamsroundone=eastteamsorder
    eastsrsroundone=eastsrsorder
    westteamsroundone=westteamsorder
    westsrsroundone=westsrsorder

print("East Playoffs Round 1:")
print(eastteamsroundone[0],"vs.",eastteamsroundone[7],end=": ") #if seed 1 beats 8, they move to round 2
if eastsrsroundone[0] >= eastsrsroundone[7]:
    eastteamsroundtwo.append(eastteamsroundone[0])
    eastsrsroundtwo.append(eastsrsroundone[0])
    print(eastteamsroundone[0],"wins.")
else: #if 8 beats 1, they move to round 2
    eastteamsroundtwo.append(eastteamsroundone[7])
    eastsrsroundtwo.append(eastsrsroundone[7])
    print(eastteamsroundone[7],"wins.")
print(eastteamsroundone[1],"vs.",eastteamsroundone[6],end=": ")
if eastsrsroundone[1] >= eastsrsroundone[6]: #if sed 2 beats seed 7, they move to round 2
    eastteamsroundtwo.append(eastteamsroundone[1])
    eastsrsroundtwo.append(eastsrsroundone[1])
    print(eastteamsroundone[1],"wins.")
else: #if 7 beats 2, they move to round 2
    eastteamsroundtwo.append(eastteamsroundone[6])
    eastsrsroundtwo.append(eastsrsroundone[6])
    print(eastteamsroundone[6],"wins.")
print(eastteamsroundone[2],"vs.",eastteamsroundone[5],end=": ")
if eastsrsroundone[2] >= eastsrsroundone[5]: #if seed 3 beats 6, they move to round 2
    eastteamsroundtwo.append(eastteamsroundone[2])
    eastsrsroundtwo.append(eastsrsroundone[2])
    print(eastteamsroundone[2],"wins.")
else: #if 6 beats 3, they move to round 2
    eastteamsroundtwo.append(eastteamsroundone[5])
    eastsrsroundtwo.append(eastsrsroundone[5])
    print(eastteamsroundone[5],"wins.")
print(eastteamsroundone[3],"vs.",eastteamsroundone[4],end=": ")
if eastsrsroundone[3] >= eastsrsroundone[4]: #if seed 4 beats seed 5, they move to round 2
    eastteamsroundtwo.append(eastteamsroundone[3])
    eastsrsroundtwo.append(eastsrsroundone[3])
    print(eastteamsroundone[3],"wins.",end="\n\n")
else: #if 5 beats 4, they move to round 2
    eastteamsroundtwo.append(eastteamsroundone[4])
    eastsrsroundtwo.append(eastsrsroundone[4])
    print(eastteamsroundone[4],"wins.",end="\n\n")
print("East Playoffs Round 2:")
print(eastteamsroundtwo[0],"vs.",eastteamsroundtwo[3],end=": ")
if eastsrsroundtwo[0] >= eastsrsroundtwo[3]: #if winner of seed 1vs8 beats winner of 4vs5, they move to round 3
    eastteamsroundthree.append(eastteamsroundtwo[0])
    eastsrsroundthree.append(eastsrsroundtwo[0])
    print(eastteamsroundtwo[0],"wins.")
else: #else winner of 4vs5 moves to round 3
    eastteamsroundthree.append(eastteamsroundtwo[3])
    eastsrsroundthree.append(eastsrsroundtwo[3])
    print(eastteamsroundtwo[3],"wins.")
print(eastteamsroundtwo[1],"vs.",eastteamsroundtwo[2],end=": ")
if eastsrsroundtwo[1] >= eastsrsroundtwo[2]: #if winner of seed 2vs7 beats winner of 3vs6, they move to round 3
    eastteamsroundthree.append(eastteamsroundtwo[1])
    eastsrsroundthree.append(eastsrsroundtwo[1])
    print(eastteamsroundtwo[1],"wins.",end="\n\n")
else:#else winner of 3vs6 moves to round 3
    eastteamsroundthree.append(eastteamsroundtwo[2])
    eastsrsroundthree.append(eastsrsroundtwo[2])
    print(eastteamsroundtwo[2],"wins.",end="\n\n")
print("East Playoffs Round 3:")
print(eastteamsroundthree[0],"vs.",eastteamsroundthree[1],end=": ")
if eastsrsroundthree[0] >= eastsrsroundthree[1]: #if winner of 1vs8vs4vs5 beats winner of 2vs7vs3vs6, they move to finals
    finalsteams.append(eastteamsroundthree[0])
    srsfinals.append(eastsrsroundthree[0])
    print(eastteamsroundthree[0],"wins the east.",end="\n\n")
else: #else other team wins and moves to finals
    finalsteams.append(eastteamsroundthree[1])
    srsfinals.append(eastsrsroundthree[1])
    print(eastteamsroundthree[1],"wins the east.",end="\n\n")

print("West Playoffs Round 1:") #same as east but for west teams
print(westteamsroundone[0],"vs.",westteamsroundone[7],end=": ")
if westsrsroundone[0] >= westsrsroundone[7]:
    westteamsroundtwo.append(westteamsroundone[0])
    westsrsroundtwo.append(westsrsroundone[0])
    print(westteamsroundone[0],"wins.")
else:
    westteamsroundtwo.append(westteamsroundone[7])
    westsrsroundtwo.append(westsrsroundone[7])
    print(westteamsroundone[7],"wins.")
print(westteamsroundone[1],"vs.",westteamsroundone[6],end=": ")
if westsrsroundone[1] >= westsrsroundone[6]:
    westteamsroundtwo.append(westteamsroundone[1])
    westsrsroundtwo.append(westsrsroundone[1])
    print(westteamsroundone[1],"wins.")
else:
    westteamsroundtwo.append(westteamsroundone[6])
    westsrsroundtwo.append(westsrsroundone[6])
    print(westteamsroundone[6],"wins.")
print(westteamsroundone[2],"vs.",westteamsroundone[5],end=": ")
if westsrsroundone[2] >= westsrsroundone[5]:
    westteamsroundtwo.append(westteamsroundone[2])
    westsrsroundtwo.append(westsrsroundone[2])
    print(westteamsroundone[2],"wins.")
else:
    westteamsroundtwo.append(westteamsroundone[5])
    westsrsroundtwo.append(westsrsroundone[5])
    print(westteamsroundone[5],"wins.")
print(westteamsroundone[3],"vs.",westteamsroundone[4],end=": ")
if westsrsroundone[3] >= westsrsroundone[4]:
    westteamsroundtwo.append(westteamsroundone[3])
    westsrsroundtwo.append(westsrsroundone[3])
    print(westteamsroundone[3],"wins.",end="\n\n")
else:
    westteamsroundtwo.append(westteamsroundone[4])
    westsrsroundtwo.append(westsrsroundone[4])
    print(westteamsroundone[4],"wins.",end="\n\n")
print("West Playoffs Round 2:")
print(westteamsroundtwo[0],"vs.",westteamsroundtwo[3],end=": ")
if westsrsroundtwo[0] >= westsrsroundtwo[3]:
    westteamsroundthree.append(westteamsroundtwo[0])
    westsrsroundthree.append(westsrsroundtwo[0])
    print(westteamsroundtwo[0],"wins.")
else:
    westteamsroundthree.append(westteamsroundtwo[3])
    westsrsroundthree.append(westsrsroundtwo[3])
    print(westteamsroundtwo[3],"wins.")
print(westteamsroundtwo[1],"vs.",westteamsroundtwo[2],end=": ")
if westsrsroundtwo[1] >= westsrsroundtwo[2]:
    westteamsroundthree.append(westteamsroundtwo[1])
    westsrsroundthree.append(westsrsroundtwo[1])
    print(westteamsroundtwo[1],"wins.",end="\n\n")
else:
    westteamsroundthree.append(westteamsroundtwo[2])
    westsrsroundthree.append(westsrsroundtwo[2])
    print(westteamsroundtwo[2],"wins.",end="\n\n")
print("West Playoffs Round 3:")
print(westteamsroundthree[0],"vs.",westteamsroundthree[1],end=": ")
if westsrsroundthree[0] >= westsrsroundthree[1]: 
    finalsteams.append(westteamsroundthree[0])
    srsfinals.append(westsrsroundthree[0])
    print(westteamsroundthree[0],"wins the west.",end="\n\n")
else:
    finalsteams.append(westteamsroundthree[1])
    srsfinals.append(westsrsroundthree[1])
    print(westteamsroundthree[1],"wins the west.",end="\n\n")

print("NBA Finals:")
print(finalsteams[0],"vs.",finalsteams[1])
if srsfinals[0] > srsfinals[1]: #if winner of eastern conference beats winner of western conf. east team wins championship
    print(finalsteams[0],"won and is the simulated",szn,"champion!")
    winner=finalsteams[0]
elif srsfinals[1] > srsfinals[0]: #if winner of west conf beats east conf winner, west team wins championship
    print(finalsteams[1],"won and is the simulated",szn,"champion!")
    winner=finalsteams[1]
else: #if both teams' srs ratings are equal
    num = random.randint(1,2) #generate a either 1 or 2
    if num == 1: #if number is 1, east team wins
        print("It was a close match up, but",finalsteams[0],"won and is the simulated",szn,"champions!")
        winner=finalsteams[0]
    else: #else west team wins
        print("It was a close match up, but",finalsteams[1],"won and is the simulated",szn,"champions!")
        winner=finalsteams[1]

import numpy as np

if szn in ["2020","2021"]: #plot diagram of play in bracket if season is 2020 onward
    fig2 = plt.figure(figsize=(10,8)) #plot figure with a given size 0f 10x8
    ax = fig2.add_axes((0,0,1,.95)) #axes for plot go from bottom left corner to top right corner, but a little downward so title can fit
    ax.set_xlim(0,10) #set x and y scale so circles scale to not be ellipses
    ax.set_ylim(0,8)
    ax.set_facecolor("lightgray") #make color of background light gray
    centers=[(.5,5.5),(.5,4.5),(.5,1.5),(.5,.5),(2,5),(2,3),(2,1),(3.5,2),
            (9.5,5.5),(9.5,4.5),(9.5,1.5),(9.5,.5),(8,5),(8,3),(8,1),(6.5,2)] #designates points for where bracket circles will be drawn on plot
    radii=.5
    texts=['(7)'+" "+str(westteamsorder[6]),'(8)'+" "+str(westteamsorder[7]),'(9)'+" "+str(westteamsorder[8]),'(10)'+" "+str(westteamsorder[9]),'(7)'+" "+str(westteamsroundone[6]),westloserteampi[0],westwinnerteampi[0],
            '(8)'+" "+str(westteamsroundone[7]),'(7)'+" "+str(eastteamsorder[6]),'(8)'+" "+str(eastteamsorder[7]),'(9)'+" "+str(eastteamsorder[8]),'(10)'+" "+str(eastteamsorder[9]),'(7)'+" "+str(eastteamsroundone[6]),eastloserteampi[0],
            eastwinnerteampi[0],'(8)'+" "+str(eastteamsroundone[7])] #designated text to go on each center

    for i, center in enumerate(centers): #draw circles around each center and put text on top of center
        x, y = center
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(x + radii * np.cos(theta), y + radii * np.sin(theta), color="black")
        ax.text(x, y, texts[i], horizontalalignment="center",verticalalignment="center",color="black")

    plt.title("Simulated Play-In Bracket for "+szn)
    plt.show()

#same concept as previous diagram, but there will be more "centers" for the full, simulated playoff bracket
fig = plt.figure(figsize=(10,8))
ax = fig.add_axes((0,0,1,.95))
ax.set_xlim(0,10)
ax.set_ylim(0,8)
ax.set_facecolor("lightgray")
centers=[(.5,7.5),(.5,6.5),(.5,5.5),(.5,4.5),(.5,3.5),(.5,2.5),(.5,1.5),(.5,.5),(2,7),(2,5),(2,3),(2,1),(3.5,6),(3.5,2),(4,4),
        (9.5,7.5),(9.5,6.5),(9.5,5.5),(9.5,4.5),(9.5,3.5),(9.5,2.5),(9.5,1.5),(9.5,.5),(8,7),(8,5),(8,3),(8,1),(6.5,6),(6.5,2),(6,4),
        (5,4)]
radii=.5
texts=['(1)'+" "+str(westteamsroundone[0]),'(8)'+" "+str(westteamsroundone[7]),'(4)'+" "+str(westteamsroundone[3]),'(5)'+" "+str(westteamsroundone[4]),'(3)'+" "+str(westteamsroundone[2]),'(6)'+" "+str(westteamsroundone[5]),
        '(2)'+" "+str(westteamsroundone[1]),'(7)'+" "+str(westteamsroundone[6]),westteamsroundtwo[0],westteamsroundtwo[3],westteamsroundtwo[2],westteamsroundtwo[1],
        westteamsroundthree[0],westteamsroundthree[1],finalsteams[1],'(1)'+" "+str(eastteamsroundone[0]),'(8)'+" "+str(eastteamsroundone[7]),'(4)'+" "+str(eastteamsroundone[3]),
        '(5)'+" "+str(eastteamsroundone[4]),'(3)'+" "+str(eastteamsroundone[2]),'(6)'+" "+str(eastteamsroundone[5]),'(2)'+" "+str(eastteamsroundone[1]),'(7)'+" "+str(eastteamsroundone[6]),eastteamsroundtwo[0],
        eastteamsroundtwo[3],eastteamsroundtwo[2],eastteamsroundtwo[1],eastteamsroundthree[0],eastteamsroundthree[1],finalsteams[0],str(winner)+"\nwins!"]

for i, center in enumerate(centers):
    x, y = center
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(x + radii * np.cos(theta), y + radii * np.sin(theta), color="black")
    ax.text(x, y, texts[i], horizontalalignment="center",verticalalignment="center",color="black")

plt.title("Simulated Playoff Bracket for "+szn)
plt.show()


#Real Playoff brackets
fig3 = plt.figure(figsize=(10,8))
ax = fig3.add_axes((0,0,1,.95))
ax.set_xlim(0,10)
ax.set_ylim(0,8)
ax.set_facecolor("lightgray")
centers=[(.5,7.5),(.5,6.5),(.5,5.5),(.5,4.5),(.5,3.5),(.5,2.5),(.5,1.5),(.5,.5),(2,7),(2,5),(2,3),(2,1),(3.5,6),(3.5,2),(4,4),
        (9.5,7.5),(9.5,6.5),(9.5,5.5),(9.5,4.5),(9.5,3.5),(9.5,2.5),(9.5,1.5),(9.5,.5),(8,7),(8,5),(8,3),(8,1),(6.5,6),(6.5,2),(6,4),
        (5,4)]
radii=.5
if szn == "2018":
    texts = ["(1) HOU","(8) MIN","(4) OKC","(5) UTA","(3) POR","(6) NOP","(2) GSW","(7) SAS","HOU","UTA","NOP","GSW","HOU","GSW","GSW","(1) TOR","(8) WAS","(4) CLE","(5) IND","(3) PHI","(6) MIA","(2) BOS","(7) MIL","TOR","CLE","PHI","BOS","CLE","BOS","CLE","GSW\nwins!"]
elif szn == "2019":
    texts = ["(1) GSW","(8) LAC","(4) HOU","(5) UTA","(3) POR","(6) OKC","(2) DEN","(7) SAS","GSW","HOU","POR","DEN","GSW","POR","GSW","(1) MIL","(8) DET","(4) BOS","(5) IND","(3) PHI","(6) BRK","(2) TOR","(7) ORL","MIL","BOS","PHI","TOR","MIL","TOR","TOR","TOR\nwins!"]
elif szn == "2020":
    texts = ["(1) LAL","(8) POR","(4) HOU","(5) OKC","(3) DEN","(6) UTA","(2) LAC","(7) DAL","LAL","HOU","DEN","LAC","LAL","DEN","LAL","(1) MIL","(8) ORL","(4) IND","(5) MIA","(3) BOS","(6) PHI","(2) TOR","(7) BRK","MIL","MIA","BOS","TOR","MIA","BOS","MIA","LAL\nwins!"]
elif szn == "2021":
    texts = ["(1) UTA","(8) MEM","(4) LAC","(5) DAL","(3) DEN","(6) POR","(2) PHO","(7) LAL","UTA","LAC","DEN","PHO","LAC","PHO","PHO","(1) PHI","(8) WAS","(4) NYK","(5) ATL","(3) MIL","(6) MIA","(2) BRK","(7) BOS","PHI","ATL","MIL","BRK","ATL","MIL","MIL","MIL\nwins!"]

for i, center in enumerate(centers):
    x, y = center
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(x + radii * np.cos(theta), y + radii * np.sin(theta), color="black")
    ax.text(x, y, texts[i], horizontalalignment="center",verticalalignment="center",color="black")

plt.title("Real Playoff Bracket for "+szn)
plt.show()