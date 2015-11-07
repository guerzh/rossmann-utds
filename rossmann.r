DATA_DIR = 'C:/Users/Guerzhoy/Desktop/utds/rossmann/'
setwd(DATA_DIR)
set.seed(0)

data = read.csv("data/train.csv")
data_stores = read.csv("data/store.csv")

data_merged = merge(data, data_stores)

get_month <- function(date_str){
  return(as.numeric(strsplit(date_str[[1]], '-')[[1]][2]))
}


get_season_col <- function(month_col){
  season_col = month_col
  season_col[month_col == 12 | month_col == 1 | month_col == 2] = "Winter"
  season_col[month_col == 3 | month_col == 4 | month_col == 5] = "Spring"
  season_col[month_col == 6 | month_col == 7 | month_col == 8] = "Summer"
  season_col[month_col == 9 | month_col == 10 | month_col == 11] = "Fall"
  return(season_col)
}


all_valid_perf = c()
for(i in 100:120){
  d = data_merged[data_merged["Store"]==i,]
  date_col = unlist(lapply(d[,"Date"], as.character))
  d[,"Month"] = unlist(lapply(date_col, get_month))
  d[,"Season"] = get_season_col(d["Month"])
  
  
  
  d = d[d["Sales"] != 0, ]
  row.names(d) = 1:nrow(d)
  
  idx = sample(1:nrow(d), nrow(d), replace=FALSE, prob=NULL)
  
  
  valid_data = d[idx[1:round(nrow(d)*.1)],]
  test_data = d[idx[round(nrow(d)*.1)+1:idx[round(nrow(d)*.2)],]
  train_data = d[idx[round(nrow(d)*.2)+1:length(idx)],]
  
  
  
  ml = lm(Sales ~   factor(DayOfWeek) + factor(Promo) + factor(Month), data=train_data)
  pred = predict.lm(ml, valid_data)
  
  perf = sqrt(mean(((pred-valid_data["Sales"])/valid_data["Sales"])^2)) 
  all_valid_perf = c(all_valid_perf, perf)
  print(perf)
}

#mean(all_valid_perf)
#[1] 0.1557868









#sqrt(mean(((predict.lm(ml, train_data)-train_data["Sales"])/train_data["Sales"])^2))
