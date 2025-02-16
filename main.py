import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("games.csv")

label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse_output=False)
encoded_data = one_hot_encoder.fit_transform(df[['team']])
encoded_df = pd.DataFrame(encoded_data,columns=one_hot_encoder.categories_[0])
df = pd.concat([df, encoded_df], axis=1)

df["home_away"]=label_encoder.fit_transform(df['home_away'])


df["win"] = (df["team_score"] > df["opponent_team_score"]).astype(float)
df["win"] = df["win"].mask(df["team_score"] < df["opponent_team_score"], 0)
df["win"] = df["win"].mask(df["team_score"] == df["opponent_team_score"], 0.5)

x = df.drop(columns=["win","notD1_incomplete","team","game_id","game_date","rest_days","FGA_3","F_tech","prev_game_dist"])
y = df["win"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.5,random_state=42)

x_train = x_train.fillna(0)
x_test = x_test.fillna(0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
