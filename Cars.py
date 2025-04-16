#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[27]:


car_df = pd.read_csv('data/cars.csv')
brand_df = pd.read_csv('data/brand.csv')


# In[9]:


car_df.head()


# In[28]:


brand_df.head()


# In[30]:


car_df['title']


# In[35]:


car_df['title'].str.split()


# In[36]:


car_df['title'].str.split(expand=True)


# In[43]:


car_df['brand'] = car_df['title'].str.split(expand=True)[0]


# In[47]:


car_df.head()


# In[51]:


brand_df['title'] = brand_df['title'].str.upper()


# In[67]:


brand_df.head()


# In[68]:


car_df = car_df.merge(brand_df, left_on='brand', right_on='title', how='left')


# In[70]:


car_df


# In[73]:


car_df.drop('title_y', axis=1, inplace = True)


# In[76]:


car_df.head()


# In[81]:


car_df.rename({'title_x':'title'}, axis = 1, inplace = True)


# In[84]:


car_df.head()


# In[86]:


bonus_df = car_df.copy()


# In[104]:


car_df.info()


# In[105]:


car_df.drop(['country_y'], axis=1, inplace = True)


# In[108]:


car_df.info()


# In[121]:


car_df.columns


# In[127]:


car_df = car_df.loc[:, ~car_df.columns.duplicated()]


# In[130]:


car_df.info()


# In[134]:


car_df.rename({'country_x':'country'}, axis=1)


# In[136]:


car_df = car_df.rename({'country_x':'country'}, axis=1)


# In[138]:


car_df.info()


# In[146]:


car_df['Engine'] = car_df['Engine'].str.split('L', expand = True)[0]


# In[148]:


car_df['Engine']


# In[150]:


car_df['Emission Class'].str.split()


# In[153]:


car_df['Emission Class'].str.split(expand = True)


# In[159]:


car_df['Emission Class'] = car_df['Emission Class'].str.split(expand = True)[1]


# In[168]:


car_df


# In[175]:


car_df = bonus_df.loc[:, ~bonus_df.columns.duplicated()]


# In[177]:


car_df.info()


# In[181]:


car_df  = car_df.rename({'country_x': 'country'}, axis=1)


# In[183]:


car_df


# In[191]:


car_df = car_df['Engine'].str.split('L', expand= True)[0]


# In[198]:


car_df = bonus_df


# In[200]:


car_df


# In[202]:


car_df = bonus_df.loc[:, ~bonus_df.columns.duplicated()]


# In[204]:


car_df.info()


# In[211]:


car_df = car_df.rename({'country_x': 'country'}, axis = 1)


# In[213]:


car_df.info()


# In[218]:


car_df['Engine'] = car_df['Engine'].str.split('L', expand=True) [0]


# In[222]:


car_df['Emission Class'] = car_df['Emission Class'].str.split(expand = True) [1]


# In[225]:


car_df1 = car_df.copy()


# In[228]:


car_df


# In[230]:


car_df.info()


# In[232]:


car_df['Engine']=pd.to_numeric(car_df['Engine'])


# In[234]:


car_df['Emission Class']=pd.to_numeric(car_df['Emission Class'])


# In[237]:


car_df.info()


# In[239]:


car_df.describe()


# In[242]:


car_df.isna().mean()


# In[250]:


car_df["Service history"].unique()


# In[257]:


car_df.groupby('Service history')['Price'].mean()


# In[262]:


car_df['Service history'] = car_df['Service history'].fillna('unknown')


# In[267]:


car_df.groupby('Service history')['Price'].mean()


# In[273]:


car_df.isna().sum()


# In[275]:


car_df.isna().sum(axis=1)


# In[287]:


car_df.isna().sum()


# In[279]:


car_df


# In[291]:


car_df['na_values'].value_counts().sort_index()


# In[296]:


len(car_df[car_df['na_values'] == 2])


# In[300]:


car_df = car_df[car_df['na_values'] <= 4]


# In[303]:


car_df


# In[305]:


car_df2 = car_df.copy()


# In[307]:


car_df2


# In[309]:


car_df


# In[313]:


car_df.drop('na_values', axis = 1)


# In[315]:


car_df = car_df.drop('na_values', axis = 1)


# In[317]:


car_df


# In[320]:


car_df = car_df2


# In[324]:


car_df = car_df[car_df['na_values'] < 4]


# In[327]:


car_df.drop('na_values', axis=1, inplace = True)


# In[331]:


car_df.isna().mean()


# In[337]:


sns.displot(car_df['Previous Owners'])


# In[339]:


car_df['Previous Owners'].mean()


# In[341]:


car_df['Previous Owners'].median()


# In[343]:


sns.displot(car_df['Engine'])


# In[348]:


car_df.isna().mean()


# In[350]:


sns.displot(car_df['Doors'])


# In[352]:


car_df['Doors'].mean()


# In[355]:


car_df['Doors'].median()


# In[359]:


sns.displot(car_df['Emission Class'])


# In[361]:


car_df['Emission Class'].mean()


# In[363]:


car_df['Emission Class'].median()


# In[365]:


sns.displot(car_df['Seats'])


# In[367]:


car_df['Seats'].mean()


# In[370]:


car_df['Seats'].median()


# In[374]:


car_df = car_df.fillna(car_df.median())


# In[376]:


car_df3=car_df.copy()


# In[378]:


car_df.describe()


# In[381]:


car_df['Price'].sort_values()


# In[383]:


car_df['Mileage(miles)'].sort_values()


# In[387]:


car_df[~car_df['Mileage(miles)']<1000]


# In[389]:


car_df = car_df[~car_df['Mileage(miles)']<1000]


# In[392]:


car_df[car_df['Registration_Year'] == 2025]


# In[397]:


car_df['Registration_Year'].sort_values()


# In[403]:


car_df = car_df[car_df['Registration_Year'] < 2025]


# In[407]:


car_df['Previous Owners'].sort_values()


# In[410]:


car_df[car_df['Previous Owners'] == 9]


# In[413]:


car_df.groupby('brand')['Price'].agg(['mean', 'std'])


# In[416]:


pd.pivot_table(car_df, index = 'brand', columns = 'Fuel type', values ='Price')


# In[418]:


car_df.head()


# In[422]:


sns.scatterplot( x= car_df['Previous Owners'], y=car_df['Price'])


# In[426]:


sns.scatterplot( x = car_df['Registration_Year'], y=car_df['Price'])


# In[429]:


sns.scatterplot(x=car_df['Registration_Year'], y=np.log(car_df['Price']))


# In[431]:


car_df


# In[438]:


car_df[['title','Fuel type','Body type', 'Emission Class','Service history','brand','country' ]].nunique(0)


# In[440]:


car_df.drop('title', axis =1, inplace = True)


# In[442]:


car_df


# In[450]:


car_df.groupby('brand')['Price'].mean()


# In[455]:


car_df['brand'].value_counts()


# In[457]:


pd.DataFrame(car_df['brand'].value_counts()).join(car_df.groupby('brand')['Price'].mean())


# In[460]:


pd.get_dummies(car_df)


# In[481]:



car_df = pd.get_dummies(car_df, drop_first = True)


# In[467]:


from sklearn.preprocessing import RobustScaler


# In[469]:


rs = RobustScaler()


# In[482]:


rs.fit_transform(car_df)


# In[475]:


car_df.info()


# In[479]:


from sklearn.preprocessing import RobustScaler

rs = RobustScaler()

# 숫자형 컬럼만 골라서 스케일링
num_cols = car_df.select_dtypes(include=['number'])
scaled_data = rs.fit_transform(num_cols)

# 결과를 DataFrame으로 다시 만들기
scaled_df = pd.DataFrame(scaled_data, columns=num_cols.columns)


# In[483]:


scaled_df


# In[485]:


car_df


# In[511]:


car_df = pd.DataFrame(rs.fit_transform(car_df), columns=car_df.columns)


# In[512]:


from sklearn.decomposition import PCA


# In[513]:


pca = PCA(5)


# In[514]:


pca.fit(car_df)


# In[515]:


pca.explained_variance_ratio_.sum()


# In[518]:


for i in range(0, 11):
    pca = PCA(i)
    pca.fit(car_df)
    print(i, round(pca.explained_variance_ratio_.sum(),2))


# In[519]:


pac = PCA(7)


# In[522]:


pd.DataFrame(pac.fit_transform(car_df), columns =['PC1','PC2','PC3','PC4','PC5','PC6','PC7'])


# In[530]:


bonus_df


# In[539]:


# 1. 중복 country_x들 추출
bonus_df.loc[:, bonus_df.columns == 'country_x']


# In[542]:


bonus_df = bonus_df.loc[:, ~bonus_df.columns.duplicated()]


# In[552]:


bonus_df = bonus_df.rename({'country_x':'country'}, axis=1)


# In[554]:


bonus_df


# In[560]:


bonus_df.groupby('country')['brand'].nunique()


# In[562]:


bonus_df.corr()


# In[564]:


car_df.corr()


# In[ ]:




