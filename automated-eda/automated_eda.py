import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
from sklearn.preprocessing import StandardScaler

@st.cache
def load_data(data):
  if data.name.endswith('.csv'):
    df = pd.read_csv(data)
  else:
    df = pd.read_excel(data)
  return df

def encode_categorical(df):
  return pd.get_dummies(df)

def fill_nans(df):
  return df.fillna(df.mean())

def drop_duplicates(df):
  return df.drop_duplicates() 

def scale_features(df):
  scaler = StandardScaler()
  return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


def main():
    st.header("Exploratory Data Analysis")
    
    data = st.file_uploader("upload a dataset", type=['csv', 'xlsx'])
    if data:
      df = load_data(data)
      
      # 1- Before processing data
      st.subheader('Before processing data')
      st.dataframe(df.head())
          
      if st.checkbox("shape"): 
        st.write(df.shape)

      if st.checkbox("show values counts") : 
        st.write(df.iloc[:, -1].value_counts())
              
      if st.checkbox("description"): 
        st.write(df.describe())
            
      if st.checkbox("columns datatypes"):
        dtypes_df = pd.DataFrame({'col': df.dtypes.index, 'type': df.dtypes.values})
        st.write(dtypes_df)
            
      if st.checkbox("columns"): 
        st.write(df.columns.tolist())
            
      if st.checkbox("show specific column"): 
        cols_df = df[st.multiselect('select column', df.columns.tolist())]
        st.dataframe(cols_df)
              
      if st.checkbox("correlation"):
        fig, ax = plt.subplots()
        st.write(sns.heatmap(df.corr(),annot=True))
        st.pyplot(fig)
        plt.close(fig)
        
            
      # 2- after preprocessing
      st.subheader("Data Pre-processing")
             
      with st.spinner("process data..."):
        
        # 1- Encode categorical features
        preprocessed_df = encode_categorical(df)
        st.write('categorical encoded ✅')
          
        # 2- handle missing values
        preprocessed_df = fill_nans(preprocessed_df)
        st.write('NaNs filled ✅')
          
        # 3- drop duplicated values
        preprocessed_df = drop_duplicates(preprocessed_df)
        st.write('duplicates dropped ✅')  
          
        # 4- scale features
        preprocessed_df = scale_features(preprocessed_df)      
        st.write('features scaled ✅')      

        st.write("data pre-processed successfully ✅")
                
        st.dataframe(preprocessed_df.head())
                
        if st.checkbox("preprocessed data shape"):
          st.write(preprocessed_df.shape)
          
        if st.checkbox('preprocessed data description'):
          st.write(preprocessed_df.describe())
              
                         
      # 3- Data visualization        
      st.subheader('Data visualization')        
      cols_names = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
      
      if cols_names:
        plot_type = st.selectbox("select plot type", ['bar', 'line', 'area', 'hist', 'box'])
        selected_cols = st.multiselect('select cols to plot', cols_names)
      
        if st.button("plot the column"):
          st.success(f"{plot_type} plot for {selected_cols}")
   
          if plot_type == 'bar':
            st.bar_chart(df[selected_cols])
            
          elif plot_type == 'line':
            st.line_chart(df[selected_cols])
            
          elif plot_type == 'area':
            st.area_chart(df[selected_cols])
          
          elif plot_type:
              fig, ax = plt.subplots()
              df[selected_cols].plot(kind=plot_type, ax=ax)
              st.pyplot(fig)


if __name__ == '__main__':
	main()
