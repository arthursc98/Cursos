# loading csv file
statesInfo<-read.csv('stateData.csv')

# creating a subset(like querying data on Python using Pandas)
subset(statesInfo,state.region==1)

# another way to create a subset
stateSubset<-statesInfo[statesInfo$state.region==1,]