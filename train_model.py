import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data.pickle', 'rb'))

print("Sample feature lengths:", [len(sample) for sample in data_dict['data'][:10]])


fixed_length = min(len(sample) for sample in data_dict['data'])  # Use minimum length
data = np.array([sample[:fixed_length] for sample in data_dict['data']])  # Truncate
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f"{score * 100:.2f}% of samples were classified correctly")

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
