
import numpy as np
import pandas as pd
import pickle
#initially I have taken only 2000 records for my analysis not the full set of data 
df_capu =pd.read_csv("model_data.csv",sep=',',engine='python')
#df_new = df.sample(2000000)



from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from sklearn.utils import resample
from sklearn import metrics


is_purcahase_dataset = df_capu[df_capu['is_purchased']== 1]
n_samples_x = int(len((is_purcahase_dataset.index)))
is_purcahase_dataset.shape[0]

not_purcahase_dataset = df_capu[df_capu['is_purchased']== 0]
n_samples_y = len((not_purcahase_dataset.index))
not_purcahase_dataset.shape[0]

is_purchase_downsampled = resample(is_purcahase_dataset,
                                replace = False, 
                                n_samples = n_samples_x ,
                                random_state = 27)
not_purcahase_set_downsampled = resample(not_purcahase_dataset,
                                replace = False,
                                n_samples = n_samples_y,
                                random_state = 27)

downsampled = pd.concat([is_purchase_downsampled, not_purcahase_set_downsampled])
downsampled['is_purchased'].value_counts()

downsampled.columns

features = downsampled[['brand', 'price', 'dayofweek' , 'category_code_split1', 'category_code_split2','activity_count','Time_Spend']]


features.loc[:,'brand'] = LabelEncoder().fit_transform(downsampled.loc[:,'brand'].copy())
features.loc[:,'dayofweek'] = LabelEncoder().fit_transform(downsampled.loc[:,'dayofweek'].copy())
features.loc[:,'category_code_split1'] = LabelEncoder().fit_transform(downsampled.loc[:,'category_code_split1'].copy())
features.loc[:,'category_code_split2'] = LabelEncoder().fit_transform(downsampled.loc[:,'category_code_split2'].copy())
features.loc[:,'Time_Spend'] = LabelEncoder().fit_transform(downsampled.loc[:,'Time_Spend'].copy())
is_purchased = LabelEncoder().fit_transform(downsampled['is_purchased'])
features.head()

X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    is_purchased, 
                                                    test_size = 0.2, 
                                                    random_state = 0)


from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

pickle.dump(clf, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


# is_purcahase_set = df[df['is_purchased']== 1]
# is_purcahase_set.shape[0]

# not_purcahase_set = df[df['is_purchased']== 0]
# not_purcahase_set.shape[0]

# n_samples_x = 553
# n_samples_y = 4456
# is_purchase_downsampled = resample(is_purcahase_set,
#                                 replace = False, 
#                                 n_samples = n_samples_x ,
#                                 random_state = 27)
# not_purcahase_set_downsampled = resample(not_purcahase_set,
#                                 replace = False,
#                                 n_samples = n_samples_y,
#                                 random_state = 27)

# downsampled = pd.concat([is_purchase_downsampled, not_purcahase_set_downsampled])
# downsampled['is_purchased'].value_counts()

# downsampled.columns

# features = downsampled[['brand', 'price', 'event_weekday' , 'category_code_level1', 'category_code_level2','activity_count']]

# features.loc[:,'brand'] = LabelEncoder().fit_transform(downsampled.loc[:,'brand'].copy())
# features.loc[:,'event_weekday'] = LabelEncoder().fit_transform(downsampled.loc[:,'event_weekday'].copy())
# features.loc[:,'category_code_level1'] = LabelEncoder().fit_transform(downsampled.loc[:,'category_code_level1'].copy())
# features.loc[:,'category_code_level2'] = LabelEncoder().fit_transform(downsampled.loc[:,'category_code_level2'].copy())
# #features.loc[:,'Time_Spend'] = LabelEncoder().fit_transform(downsampled.loc[:,'Time_Spend'].copy())
# #features.loc[:,'is_viewed'] = LabelEncoder().fit_transform(downsampled.loc[:,'is_viewed'].copy())
# is_purchased = LabelEncoder().fit_transform(downsampled['is_purchased'])
# # features.head()

# X_train, X_test, y_train, y_test = train_test_split(features, 
#                                                     is_purchased, 
#                                                     test_size = 0.2, 
#                                                     random_state = 0)

# from sklearn import svm
# clf = svm.SVC()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# pickle.dump(clf, open('model.pkl','wb'))
# model = pickle.load(open('model.pkl','rb'))
#print(model.predict([['samsung',503.09,4,'electronics','smartphone',1]]))
# input= [[174,179,5,7,27,3]]
# output = model.predict(X_test)
# for i in output:
#          print(output)


































