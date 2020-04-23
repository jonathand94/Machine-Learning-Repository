import pandas as pd

df = pd.read_csv('h.csv')
t = df['timestamp']
dt = pd.to_datetime(t)

max_count = 0
min_count = 0

for i in range(24):

    bools = dt.between(str(i) + ':00:00', str(i+1)+':59:59').tolist()

    count = 0
    for b in bools:
        if b:
            count+=1

    print(str(i) + ':00:00')
    print(count)
    if count >= max_count:
        max_count = count
    if count <= min_count:
        min_count = count

