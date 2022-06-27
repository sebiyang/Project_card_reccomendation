import pandas as pd

customer = pd.read_csv('C:\\Users\\sebiy\\section3_project\\customer_fin.csv')

customer['sales_sum'] = customer['selling_price'] * customer['quantity']
customer = customer.loc[:,['sales_sum','rented','family_size','no_of_children','income_bracket','age_range']]

customer['no_of_children'].replace('3+','3',inplace=True)
customer['family_size'].replace('5+','5',inplace=True)

customer['age_range'].replace('18-25','20',inplace=True)
customer['age_range'].replace('26-35','30',inplace=True)
customer['age_range'].replace('36-45','40',inplace=True)
customer['age_range'].replace('46-55','50',inplace=True)
customer['age_range'].replace('56-70','60',inplace=True)
customer['age_range'].replace('70+','70',inplace=True)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd



# K-means train & Elbow method
X = customer[['sales_sum', 'rented', 'family_size', 'no_of_children','income_bracket', 'age_range']]

k_list = []
cost_list = []
for k in range (2, 7):
    kmeans = KMeans(n_clusters=k).fit(X)
    interia = kmeans.inertia_  #evaluation metrics of k-means
    print ("k:", k, "| cost:", interia)
    k_list.append(k)
    cost_list.append(interia)
    
plt.plot(k_list, cost_list)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# selected by elbow method (5)
kmeans = KMeans(n_clusters=5, algorithm="elkan").fit(X)

cluster_num = kmeans.predict(X)

cluster = pd.Series(cluster_num)
customer['cluster_num'] = cluster.values
customer.head()



#split
customer_encoded = customer

target = 'cluster_num'
features = customer_encoded.drop(columns=[target]).columns

# split into train/test set
# no validation set needed (used k-fold evaluation rather)
from sklearn.model_selection import train_test_split


train, test = train_test_split(customer_encoded, train_size=0.80, test_size=0.20, 
                              shuffle=False, random_state=2)


X_train = train[features]
y_train = train[target]
X_test= test[features]
y_test = test[target]


from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

n_classes=5

multi = MultiOutputClassifier(RandomForestClassifier(class_weight='balanced', n_jobs=-1,
                                                     oob_score=True, random_state=42, 
                                                     min_samples_split=4
                                                     ,min_samples_leaf=1, max_depth=19))



#Multioutput Classification needs to be reshaped 
multi.fit(X_train, y_train.values.reshape(-1, 1));


import pickle

with open ('multi.pkl','wb') as pickle_file:
    pickle.dump(multi, pickle_file)
