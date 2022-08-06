import streamlit as st
from streamlit_option_menu import option_menu
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
from matplotlib import colors
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#########################################
st.set_page_config(layout="wide")
#Remove the Hamburger menu for better user interface
# st.markdown(""" <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style> """, unsafe_allow_html=True)

#Initiate the option menu that will be called in every section later on
rad = option_menu(
            menu_title=None,  # required
            options=["Home Page", "Dataset Info", "Numerical Features", "Categorical Features", "Segmentation", "Predictions"],  # required
            icons=["house", "book","123","bookmark","person", "arrow-up-right"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
# padding=0
# hide_streamlit_style = """
# <style>
# .css-18e3th9 {padding-top: 0rem;}
# </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)
#<div data-testid="stToolbar" class="css-r698ls e8zbici2"><span id="MainMenu" aria-haspopup="true" aria-expanded="false"><button kind="icon" class="css-ex7byz edgvbvh6"><svg viewBox="0 0 8 8" aria-hidden="true" focusable="false" fill="currentColor" xmlns="http://www.w3.org/2000/svg" color="inherit" class="e1fb0mya0 css-xq1lnh-EmotionIconBase ex0cdmw0"><path d="M0 1v1h8V1H0zm0 2.97v1h8v-1H0zm0 3v1h8v-1H0z"></path></svg></button></span></div>
# st.markdown(f""" <style>
# .reportview-container .main .block-container{{
# padding-top: {padding}rem;
# padding-right: {padding}rem;
# padding-left: {padding}rem;
# padding-bottom: {padding}rem;
# }} </style> """, unsafe_allow_html=True)

#Create the upload data button and assign the data as a global component so it can be used in every section
uploaded_file=st.sidebar.file_uploader(label="Upload your Data", accept_multiple_files=False, type=['csv', 'xlsx'])

global data
if uploaded_file is not None:
    uploaded_file.seek(0)
    print(uploaded_file)
    try:
        data=pd.read_csv(uploaded_file)
        num_features = data.select_dtypes(include=['int64', 'float64'])
        cat_features = data.select_dtypes(include=['object','category'])
    except Exception as e:
        print(e)
        data=pd.read_excel(uploaded_file)
        num_features = data.select_dtypes(include=['int64', 'float64'])
        cat_features = data.select_dtypes(include=['object','category'])
    # except Exception as e:
    #     print(e)
    #     st.header("Please Upload a File in the Sidebar")

######################################### HOME PAGE
if rad=="Home Page":
#     st.markdown(f""" <style>
#         .reportview-container .main .block-container{{
#             padding-top: {padding}rem;
#             padding-right: {padding}rem;
#             padding-left: {padding}rem;
#             padding-bottom: {padding}rem;
#         }} </style> """, unsafe_allow_html=True)
    image, title= st.columns((1,10))
    with image:
        st.image("https://i0.wp.com/gbsn.org/wp-content/uploads/2020/07/AUB-logo.png?ssl=1", width=200)
    with title:
        st.markdown("<h1 style='text-align: center; color: black;'>Welcome to the Customer Segmentation App! </h2>", unsafe_allow_html=True)
    col1,col2,col3= st.columns((2,5,2))
    with col1:
        st.write(' ')
    with col2:
        st.image("https://static.wixstatic.com/media/0bbc45_89b19cbc33864e719a69e25b81d96218~mv2.jpg/v1/fit/w_1000%2Ch_1000%2Cal_c%2Cq_80/file.jpg", use_column_width=True)
    with col3:
        st.write(' ')
    st.markdown("This app was created with a marketing agency in mind as a client. The app will tackle the topic of customer segmentation using machine learning to help clients correctly segment their customers based on the demographic scope of proper marketing customer segmentation. Demographic segmentation divides clients into segments based on age, profession, gender, income, education and marital status. Cutting to the chase, segmenting customers is not only important for a business but vital! This process provides a clear view on different customer behaviors and helps the business achieve better results on several scales ranging from value proposition to cost efficiency: By correctly segmenting customers, the business can better cater for the need of each group and thus provide the 'delightful' experience which is the North star of marketing. Also, a correct segmentation yields better return on investment of marketing campaigns; instead of spending money on several campaigns with a holistic approach on all customers, segmentation provides a clear guidance on the taste and preference of each target group. You most probably know the true value of customer segmentation but it is not as easy as it seems! If customer segmentation was as easy as explored in marketing books, every business would be making tons of profit and no marketing agency would have a competitive edge against its competitors. However, Data Analytics and Machine Learning tools provide an untapped potential in terms of customer segmentation CD and that is the purpose of this app! **You now have one of the most powerful tools at your fingertips to help you segment customers based on their demographics and especially based on the factors previously mentioned.**")

######################################### DATASET INFO
if rad=="Dataset Info":
    c1,c2= st.columns(2)
    y1,y2= st.columns(2)
    try:
        with c1:
            """##### Your Dataset"""
            AgGrid(data, height=300)
            rows= data.shape[0]
            columns= data.shape[1]
            st.write(f'Your dataset has {rows} rows and {columns} columns')

        with c2:
            """##### Quick Summary"""
            num_features.drop(columns=["ID"], inplace=True)
            stats= num_features.describe()
            st.table(stats)
            #table.update_layout(width=500, height=400)
            #print(stats)
            #st.write(table)
            ######################################### Remove Missing Values & ID Column

        with y1:
            """##### Missing Values"""
            fig = plt.figure(figsize=(10, 4))
            plt.ylabel('Rows (Position)', fontsize=16)
            s=sns.heatmap(data.isna(), cmap='YlGnBu', cbar_kws={'label': 'Missing Data'})
            s.set_xlabel('Column Name', fontsize=12)
            s.set_ylabel('Row (position of nan)', fontsize=12)
            st.write(fig, use_container_width=True)
            nan= data.isnull().sum().sum()
            st.write(f"The uploaded dataset has a total of {nan} missing values accross all columns!")

        with y2:
            """##### Data Cleaning"""
            data.dropna(inplace=True)
            data.drop(columns=['ID'], inplace=True)
            fig = plt.figure(figsize=(10, 4))
            plt.ylabel('Rows (Position)', fontsize=16)
            s=sns.heatmap(data.isna(), cmap='YlGnBu', cbar_kws={'label': 'Missing Data'})
            s.set_xlabel('Column Name', fontsize=12)
            s.set_ylabel('Row (position of nan)', fontsize=12)
            st.write(fig, use_container_width=True)
            rows=data.shape[0]
            columns=data.shape[1]
            st.markdown(f"After cleaning, your dataset has {rows} rows and {columns} columns. ID column was dropped since it does not add any value to this study.")

    except Exception as e:
        print(e)
        st.header("Please Upload a File in the Sidebar")
######################################### Explore Dataset (Outliers, distributions, relationships)
################################################################################    Numerical Features Analysis ###################################################################
if rad=="Numerical Features":
######################################### Density Plots
# I used a for loop on numerical variables to make this app as flexible as possible. If another data is uplaoded, the distribution plots wof all numerical columns would still appear
    num_features.dropna(inplace=True)
    num_features.drop(columns=['ID'], inplace=True)
    a1,a2,a3 =st.columns(3)
    with a1:

        """##### Density Plots"""

        # Scale the data using robust scaler to take into account outliers and render all data within same scale
        distribution=st.selectbox(" ", ["Income", "Age", "Family Size", "Work Experience"])
        if distribution == "Income":
            color_discrete_map = {'Income': 'rgb(70,130,180)'}
            dist = px.histogram(num_features["Income"], color_discrete_map=color_discrete_map)
            dist.update_layout(showlegend= False,  autosize=False,
        width=500,
        height=300,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=10,
            pad=0
        ))
            st.plotly_chart(dist)
        if distribution == "Age":
            color_discrete_map = {'Age': 'rgb(70,130,180)'}
            dist = px.histogram(num_features["Age"], color_discrete_map= color_discrete_map )
            dist.update_layout(showlegend= False,  autosize=False,
        width=500,
        height=300,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=10,
            pad=0
        ))
            st.plotly_chart(dist)

        if distribution == "Family Size":
            color_discrete_map = {'Family_Size': 'rgb(70,130,180)'}
            dist = px.histogram(num_features["Family_Size"], color_discrete_map=color_discrete_map)
            dist.update_layout(showlegend= False,  autosize=False,
        width=500,
        height=300,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=10,
            pad=0
        ))
            st.plotly_chart(dist)

        if distribution == "Work Experience":
            color_discrete_map = {'Work_Experience':'rgb(31,119,180)'}
            dist = px.histogram(num_features["Work_Experience"], color_discrete_map=color_discrete_map)
            dist.update_layout(showlegend= False, autosize=False,
        width=500,
        height=300,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=10,
            pad=0
        ))
            st.plotly_chart(dist)
######################################### Income Distribution by Gender
        """##### Income Distribution by Gender"""
        color_discrete_map = {'Male': 'rgb(31,119,180)', 'Female': 'rgb(214,39,40)'}
        hist1= px.histogram(data, x="Income", color="Gender", color_discrete_map=color_discrete_map,category_orders={'Gender':['Female', 'Male']})
        hist1.update_traces(opacity=0.75)
        hist1.update_layout(legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.6), autosize=False,
    width=500,
    height=400,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=10,
        pad=0
    ))
        st.write(hist1, use_container_width=True)
######################################### Correlation Matrix
    with a2:
        #Heatmap- Correlation Matrix to check the relationships between numerical features
        """##### Correlation"""
        # fig1=plt.figure(figsize=(10,8))
        corr = num_features.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        # sns.heatmap(data.corr(),mask=mask, cmap='YlGnBu',annot=True)
        dt=go.Heatmap(
        z=corr.mask(mask),
        x=corr.columns,
        y=corr.columns,
        colorscale='Blues',
        zmin=-1,
        zmax=1
        )
        layout=go.Layout(width=500, height=400, yaxis_autorange='reversed',margin=dict(
                l=0,
                r=0,
                b=0,
                t=10,
                pad=0
            ))
        correlation=go.Figure(data=[dt], layout=layout)
        st.write(correlation, use_container_width=True)
#########################################Age vs Income
        """#####  Age and Income"""
        color_discrete_map = {'Male': 'rgb(31,119,180)', 'Female': 'rgb(31,119,180)'}
        scatter= px.scatter(data_frame=data, x="Income", y="Age", color='Gender', color_discrete_map=color_discrete_map)
        scatter.update_layout(showlegend= False,width=500, height=400,margin=dict(
                l=0,
                r=0,
                b=0,
                t=10,
                pad=0
            ))
        st.write(scatter, use_container_width=True)

#########################################Age distribution by Gender
    with a3:
        """##### Age Distribution by Gender"""
        color_discrete_map = {'Male': 'rgb(31,119,180)', 'Female': 'rgb(214,39,40)'}
        hist2= px.histogram(data, x="Age", color="Gender", color_discrete_map=color_discrete_map, category_orders={'Gender':['Female', 'Male']})
        hist2.update_traces(opacity=0.75)
        hist2.update_layout(legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.75), autosize=False,
    width=450,
    height=400,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=10,
        pad=0
    ))
        st.write(hist2)

######################################### Income and Work Experience
        """##### Work Experience and Income"""
        color_discrete_map = {'Male':'rgb(31,119,180)', 'Female':'rgb(31,119,180)'}
        scatter2= px.scatter(data_frame=data, x="Income", y="Work_Experience", color='Gender',color_discrete_map=color_discrete_map)
        scatter2.update_layout(showlegend= False,width=450, height=400,margin=dict(
                l=0,
                r=0,
                b=0,
                t=10,
                pad=0
            ))
        st.write(scatter2, use_container_width=True)
################################################################################   Categorical Features Analysis ###################################################################
if rad == "Categorical Features":
######################################### Analysis of Categorical Features

######################################### Gender and Marriage Barplot
    aa,bb= st.columns(2)
    cc,dd=st.columns(2)

    with aa:

        """##### Gender and Marriages"""
        color_discrete_map = {'Yes': 'rgb(31,119,180)', 'No': 'rgb(214,39,40)'}
        b = cat_features.groupby(by=["Gender", "Ever_Married"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
        #b=cat_features[["Gender", "Ever_Married"]].value_counts(normalize=True).reset_index(name="Counts").sort_values(by='Counts', ascending=False)
        fig2=px.bar(b, x="Counts", y="Gender", color="Ever_Married", text_auto=True, color_discrete_map=color_discrete_map)
        fig2.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83),autosize=False, width=600, height=400,margin=dict(
                l=0,
                r=0,
                b=0,
                t=10,
                pad=0
            ))
        st.write(fig2, use_container_width=True)

######################################### Gender and Spending Score
    with bb:
        """##### Gender and Spending Score"""
        color_discrete_map = {'Low': 'rgb(31,119,180)', 'Average': 'rgb(211,211,211)', 'High': 'rgb(214,39,40)'}
        c = cat_features.groupby(by=["Gender", "Spending_Score"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
        #c["Percentage"]=100* c["C"]
        fig4=px.bar(c, x="Counts", y="Gender", color="Spending_Score", text_auto=True, color_discrete_map=color_discrete_map)
        fig4.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=400,margin=dict(
                l=0,
                r=0,
                b=0,
                t=10,
                pad=0
            ))
        st.write(fig4, use_container_width=True)
######################################### Gender and profession barplot
    with cc:
        """##### Gender and Profession"""
        color_discrete_map = {'Male': 'rgb(31,119,180)', 'Female': 'rgb(214,39,40)'}
        a = cat_features.groupby(by=["Gender", "Profession"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
        fig3=px.bar(a, x="Counts", y="Profession", color="Gender", text_auto=True, color_discrete_map=color_discrete_map)
        fig3.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83),autosize=False, width=600, height=400,margin=dict(
                l=0,
                r=0,
                b=0,
                t=10,
                pad=0
            ))
        st.write(fig3, use_container_width=True)


#########################################Profession and spending score
    with dd:
        """##### Education & Profession"""  #groupby spending score and profession then sort to have a good looking barplot
        color_discrete_map = {'Yes': 'rgb(31,119,180)', 'No': 'rgb(214,39,40)'}
        d = cat_features.groupby(by=["Graduated", "Profession"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
        fig5=px.bar(d, x="Counts", y="Profession", color="Graduated", color_discrete_map= color_discrete_map)
        fig5.update_traces(textfont_size=12, textangle=90, textposition="inside", cliponaxis=False)
        fig5.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.76),autosize=False,width=600, height=400,margin=dict(
                l=0,
                r=0,
                b=0,
                t=10,
                pad=0
            ))
        st.write(fig5, use_container_width=True)


################################################################################    Machine Learning ###################################################################
######################################################################################################### Machine Learning, PCA and Segmentation
if rad == "Segmentation":
    """##### Pick the Algorithm """
    option = st.selectbox(
     '',
     ('KMeans', "Agglomerative Clustering", 'KMeans with PCA'))
    d,e= st.columns(2)
    f,g=st.columns(2)
    data.dropna(inplace=True)
    data.drop(columns=['ID', 'Graduated', 'Work_Experience'], inplace=True)
    #Creating a copy of data
    df = data.copy()
    df=pd.get_dummies(df)
    #Scaling
    scaler = MinMaxScaler()  #Used MinMaxScaler to keep the order of 1 and 0 after dummy encoding
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df),columns= df.columns )

################################################################################    KMEANS ALGORITHM --> CLUSTERS BASED ON CREATING GROUPS OF EQUAL VARIANCES ###################################################################
####################################################### ROW 1
    if option == "KMeans":
        with d:
            elbow= plt.figure(figsize=(5,3))
            Elbow_M = KElbowVisualizer(KMeans(random_state=1), k=10,metric='silhouette', timings=False)
            Elbow_M.fit(df)
            Elbow_M.show()

            """##### Optimal Number of Clusters for Your Data"""
            st.write(elbow,use_container_width=True)
            st.write(f"The optimal number of clusters is {Elbow_M.elbow_value_} with a silhouette score of 0.284%")

        with e:
            Elbow_M = KElbowVisualizer(KMeans(random_state=1), k=10, metric='silhouette')
            Elbow_M.fit(df)
            AC = KMeans(n_clusters= Elbow_M.elbow_value_, random_state=1)
            # fit model and predict clusters
            y_pred = AC.fit_predict(df)
            df["Clusters"] = y_pred
            #Adding the Clusters feature to the orignal dataframe.
            data["Clusters"]= y_pred
            object1 = go.Scatter3d(
            x=data["Age"], y=data["Income"], z=data["Gender"],
            mode='markers',
            opacity=0.5,
            marker=dict(
            size=5,
            color=data["Clusters"],
            #colorscale='Viridis'
            )
            )
            # Create an object for graph layout
            segments = go.Figure(data=[object1])
            segments.update_layout(scene = dict(
            xaxis_title = "Age",
            yaxis_title = "Income",
            zaxis_title = "Gender"),
            autosize=False,
            width=700,
            height=500,
            margin=dict(r=20, b=10, l=10, t=10)
            )
            """##### The Customer Segments"""
            # Plot on the dashboard on streamlit
            st.plotly_chart(segments, use_container_width=True)
    #################################################################ROW 2
        with f:
            """##### Clusters Relationships"""
            attributes= st.selectbox(" ", ("Age and Income", "Spending Score", "Gender", "Occupation", "Marital Status"))
            if attributes=="Age and Income":
                data["Clusters"]=data["Clusters"].astype('str')
                color_discrete_map = {'0': 'rgb(31,119,180)', '1': 'rgb(214,39,40)'}
                scatter= px.scatter(data_frame=data, x="Income", y="Age", color="Clusters", opacity=0.5, color_discrete_map=color_discrete_map)
                layout=go.Layout(width=500, height=400, yaxis_autorange='reversed',margin=dict(
                l=0,
                r=0,
                b=0,
                t=0,
                pad=0
                ))
                scatter.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83),width=600, height=400,margin=dict(
                l=0,
                r=0,
                b=0,
                t=0,
                pad=0
                ))
                st.write(scatter, use_container_width=True)
            if attributes == "Spending Score":
                data["Clusters"]=data["Clusters"].astype('str')
                color_discrete_map = {'0': 'rgb(31,119,180)', '1': 'rgb(214,39,40)'}
                c = data.groupby(by=["Clusters", "Spending_Score"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
                color_discrete_map = {'0': 'rgb(31,119,180)', '1': 'rgb(214,39,40)'}
                fig4=px.bar(c, x="Counts", y="Spending_Score", color="Clusters", text_auto=True, color_discrete_map=color_discrete_map)
                fig4.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=400,margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0,
                        pad=0
                    ))
                st.write(fig4, use_container_width=True)

            if attributes == "Gender":
                data["Clusters"]=data["Clusters"].astype('category')
                color_discrete_map = {'Male': 'rgb(31,119,180)', 'Female': 'rgb(214,39,40)'}
                c = data.groupby(by=["Clusters", "Gender"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
                fig4=px.bar(c, x="Counts", y="Clusters", color="Gender", text_auto=True,color_discrete_map=color_discrete_map)
                fig4.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=400,margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0,
                        pad=0
                    ))
                st.write(fig4, use_container_width=True)

            if attributes == "Occupation":
                data["Clusters"]=data["Clusters"].astype('str')
                color_discrete_map = {'0': 'rgb(31,119,180)', '1': 'rgb(214,39,40)'}
                c = data.groupby(by=["Clusters", "Profession"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
                fig4=px.bar(c, x="Counts", y="Profession", color="Clusters", text_auto=True, color_discrete_map=color_discrete_map)
                fig4.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=400,margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0,
                        pad=0
                    ))
                st.write(fig4, use_container_width=True)

            if attributes == "Marital Status":
                data["Clusters"]=data["Clusters"].astype('category')
                color_discrete_map = {'Yes': 'rgb(31,119,180)', 'No': 'rgb(214,39,40)'}
                c = data.groupby(by=["Clusters", "Ever_Married"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
                fig4=px.bar(c, x="Counts", y="Clusters", color="Ever_Married", text_auto=True)
                fig4.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=400,margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0,
                        pad=0
                    ))
                st.write(fig4, use_container_width=True)
        with g:
            data["Clusters"]=data["Clusters"].astype('category')
            data["Family_Size"]= data["Family_Size"].astype("category")
            v= data.groupby(by=["Clusters"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
            fig5=px.bar(v, x="Clusters", y="Counts", text_auto=True, color_discrete_sequence=['rgb(31,119,180)'])
            fig5.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=500,margin=dict(
                    l=0,
                    r=0,
                    b=0,
                    t=100,
                    pad=0
                ))
            """##### Clusters Distribution"""
            st.write(fig5,use_container_width=True)

################################################################################    TRY AGGLOMERATIVE CLUSTERING (hierarchical) ###################################################################
####################################################### ROW 1
    if option == "Agglomerative Clustering":
        with d:
            elbow= plt.figure(figsize=(5,3))
            Elbow_M = KElbowVisualizer(AgglomerativeClustering(), k=10,metric='silhouette', timings=False)
            Elbow_M.fit(df)
            Elbow_M.show()

            """##### Optimal Number of Clusters for Your Data"""
            st.write(elbow,use_container_width=True)
            st.write(f"The optimal number of clusters is {Elbow_M.elbow_value_} with a silhouette score of 0.281%")
        with e:
            Elbow_M = KElbowVisualizer(AgglomerativeClustering(), k=10, metric='silhouette')
            Elbow_M.fit(df)
            AC = AgglomerativeClustering(n_clusters= Elbow_M.elbow_value_)
            # fit model and predict clusters
            y_pred = AC.fit_predict(df)
            df["Clusters"] = y_pred
            #Adding the Clusters feature to the orignal dataframe.
            data["Clusters"]= y_pred
            object1 = go.Scatter3d(
            x=data["Age"], y=data["Income"], z=data["Gender"],
            mode='markers',
            opacity=0.5,
            marker=dict(
            size=5,
            color=data["Clusters"],
            #colorscale='Viridis'
            )
            )
            # Create an object for graph layout
            segments = go.Figure(data=[object1])
            segments.update_layout(scene = dict(
            xaxis_title = "Age",
            yaxis_title = "Income",
            zaxis_title = "Gender"),
            autosize=False,
            width=700,
            height=500,
            margin=dict(r=20, b=10, l=10, t=10)
            )
            """##### The Customer Segments"""
            # Plot on the dashboard on streamlit
            st.plotly_chart(segments, use_container_width=True)
    #################################################################ROW 2
        with f:
            """##### Clusters Relationships"""
            attributes= st.selectbox(" ", ("Age and Income", "Spending Score", "Gender", "Occupation", "Marital Status"))
            if attributes=="Age and Income":
                data["Clusters"]=data["Clusters"].astype('str')
                color_discrete_map = {'0': 'rgb(31,119,180)', '1': 'rgb(214,39,40)'}
                scatter= px.scatter(data_frame=data, x="Income", y="Age", color="Clusters", opacity=0.5, color_discrete_map=color_discrete_map)
                layout=go.Layout(width=500, height=400, yaxis_autorange='reversed',margin=dict(
                l=0,
                r=0,
                b=0,
                t=0,
                pad=0
                ))
                scatter.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83),width=600, height=400,margin=dict(
                l=0,
                r=0,
                b=0,
                t=0,
                pad=0
                ))
                st.write(scatter, use_container_width=True)
            if attributes == "Spending Score":
                data["Clusters"]=data["Clusters"].astype('str')
                color_discrete_map = {'0': 'rgb(31,119,180)', '1': 'rgb(214,39,40)'}
                c = data.groupby(by=["Clusters", "Spending_Score"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
                color_discrete_map = {'0': 'rgb(31,119,180)', '1': 'rgb(214,39,40)'}
                fig4=px.bar(c, x="Counts", y="Spending_Score", color="Clusters", text_auto=True, color_discrete_map=color_discrete_map)
                fig4.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=400,margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0,
                        pad=0
                    ))
                st.write(fig4, use_container_width=True)

            if attributes == "Gender":
                data["Clusters"]=data["Clusters"].astype('category')
                color_discrete_map = {'Male': 'rgb(31,119,180)', 'Female': 'rgb(214,39,40)'}
                c = data.groupby(by=["Clusters", "Gender"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
                fig4=px.bar(c, x="Counts", y="Clusters", color="Gender", text_auto=True,color_discrete_map=color_discrete_map)
                fig4.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=400,margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0,
                        pad=0
                    ))
                st.write(fig4, use_container_width=True)

            if attributes == "Occupation":
                data["Clusters"]=data["Clusters"].astype('str')
                color_discrete_map = {'0': 'rgb(31,119,180)', '1': 'rgb(214,39,40)'}
                c = data.groupby(by=["Clusters", "Profession"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
                fig4=px.bar(c, x="Counts", y="Profession", color="Clusters", text_auto=True, color_discrete_map=color_discrete_map)
                fig4.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=400,margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0,
                        pad=0
                    ))
                st.write(fig4, use_container_width=True)

            if attributes == "Marital Status":
                data["Clusters"]=data["Clusters"].astype('category')
                color_discrete_map = {'Yes': 'rgb(31,119,180)', 'No': 'rgb(214,39,40)'}
                c = data.groupby(by=["Clusters", "Ever_Married"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
                fig4=px.bar(c, x="Counts", y="Clusters", color="Ever_Married", text_auto=True)
                fig4.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=400,margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0,
                        pad=0
                    ))
                st.write(fig4, use_container_width=True)
        with g:
            data["Clusters"]=data["Clusters"].astype('category')
            data["Family_Size"]= data["Family_Size"].astype("category")
            v= data.groupby(by=["Clusters"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
            fig5=px.bar(v, x="Clusters", y="Counts", text_auto=True, color_discrete_sequence=['rgb(31,119,180)'])
            fig5.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=500,margin=dict(
                    l=0,
                    r=0,
                    b=0,
                    t=100,
                    pad=0
                ))
            """##### Clusters Distribution"""
            st.write(fig5,use_container_width=True)


################################################################################    NO OUTLIERS ASSUMPTION --> USE STANDARSCALER() INSTEAD OF RobustScaler() ###################################################################
####################################################### ROW 1
    if option == "KMeans with PCA":
        #Define and fit PCA on dataset
        pca = PCA(n_components=3, random_state=1)
        pca.fit(df)
        PCA_df = pd.DataFrame(pca.transform(df), columns=(["col1","col2","col3"]))
        #Assign names to the created columns by PCA
        x =PCA_df["col1"]
        y =PCA_df["col2"]
        z =PCA_df["col3"]
        with d:
            elbow= plt.figure(figsize=(5,3))
            Elbow_M = KElbowVisualizer(KMeans(random_state=1), k=10, metric='silhouette',timings=False)
            Elbow_M.fit(PCA_df)
            Elbow_M.show()
            """##### Optimal Number of Clusters for Your Data"""
            st.write(elbow,use_container_width=True)
            st.write(f"The optimal number of clusters is {Elbow_M.elbow_value_} with a silhouette score of 0.567")
        with e:
            Elbow_M = KElbowVisualizer(KMeans(random_state=1), k=10, metric='silhouette')
            Elbow_M.fit(PCA_df)
            AC = KMeans(n_clusters= Elbow_M.elbow_value_, random_state=1)
            # fit model and predict clusters on PCA dataframe
            y_pred = AC.fit_predict(PCA_df)
            PCA_df["Clusters"] = y_pred
            #Adding the Clusters feature to the orignal dataframe.
            data["Clusters"]= y_pred
            object1 = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            opacity=0.5,
            marker=dict(
            size=5,
            color=data["Clusters"],
            #colorscale='Viridis'
            )
            )
            # Create an object for graph layout
            segments = go.Figure(data=[object1])
            segments.update_layout(scene = dict(
            xaxis_title = "Feature 1",
            yaxis_title = "Feature 2",
            zaxis_title = "Feature 3"),
            autosize=False,
            width=700,
            height=500,
            margin=dict(r=20, b=10, l=10, t=10)
            )
            """##### The Customer Segments Based on PCA Features"""
            # Plot on the dashboard on streamlit
            st.plotly_chart(segments, use_container_width=True)
    #################################################################ROW 2
        with f:
            """##### Clusters Relationships"""
            attributes= st.selectbox(" ", ("Age and Income", "Spending Score", "Gender", "Occupation", "Marital Status"))
            if attributes=="Age and Income":
                data["Clusters"]=data["Clusters"].astype('str')
                color_discrete_map = {'0': 'rgb(31,119,180)', '1': 'rgb(214,39,40)'}
                scatter= px.scatter(data_frame=data, x="Income", y="Age", color="Clusters", opacity=0.5, color_discrete_map=color_discrete_map)
                layout=go.Layout(width=500, height=400, yaxis_autorange='reversed',margin=dict(
                l=0,
                r=0,
                b=0,
                t=0,
                pad=0
                ))
                scatter.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83),width=600, height=400,margin=dict(
                l=0,
                r=0,
                b=0,
                t=0,
                pad=0
                ))
                st.write(scatter, use_container_width=True)
            if attributes == "Spending Score":
                data["Clusters"]=data["Clusters"].astype('str')
                color_discrete_map = {'0': 'rgb(31,119,180)', '1': 'rgb(214,39,40)'}
                c = data.groupby(by=["Clusters", "Spending_Score"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
                #color_discrete_map = {'0': 'rgb(31,119,180)', '1': 'rgb(214,39,40)'}
                fig4=px.bar(c, x="Counts", y="Spending_Score", color="Clusters", text_auto=True)
                fig4.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=400,margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0,
                        pad=0
                    ))
                st.write(fig4, use_container_width=True)

            if attributes == "Gender":
                data["Clusters"]=data["Clusters"].astype('category')
                #color_discrete_map = {'Male': 'rgb(31,119,180)', 'Female': 'rgb(214,39,40)'}
                c = data.groupby(by=["Clusters", "Gender"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
                fig4=px.bar(c, x="Counts", y="Clusters", color="Gender", text_auto=True)
                fig4.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=400,margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0,
                        pad=0
                    ))
                st.write(fig4, use_container_width=True)

            if attributes == "Occupation":
                data["Clusters"]=data["Clusters"].astype('str')
                #color_discrete_map = {'0': 'rgb(31,119,180)', '1': 'rgb(214,39,40)'}
                c = data.groupby(by=["Clusters", "Profession"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
                fig4=px.bar(c, x="Counts", y="Profession", color="Clusters", text_auto=True)
                fig4.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=400,margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0,
                        pad=0
                    ))
                st.write(fig4, use_container_width=True)

            if attributes == "Marital Status":
                data["Clusters"]=data["Clusters"].astype('category')
                #color_discrete_map = {'Yes': 'rgb(31,119,180)', 'No': 'rgb(214,39,40)'}
                c = data.groupby(by=["Clusters", "Ever_Married"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
                fig4=px.bar(c, x="Counts", y="Clusters", color="Ever_Married", text_auto=True)
                fig4.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=400,margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0,
                        pad=0
                    ))
                st.write(fig4, use_container_width=True)
        with g:
            data["Clusters"]=data["Clusters"].astype('category')
            data["Family_Size"]= data["Family_Size"].astype("category")
            v= data.groupby(by=["Clusters"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
            fig5=px.bar(v, x="Clusters", y="Counts", text_auto=True, color_discrete_sequence=['rgb(31,119,180)'])
            fig5.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.83), autosize=False, width=600, height=500,margin=dict(
                    l=0,
                    r=0,
                    b=0,
                    t=100,
                    pad=0
                ))
            """##### Clusters Distribution"""
            st.write(fig5,use_container_width=True)
################################################################################   PREDICTIONS ###################################################################
if rad == "Predictions":
    predzz=st.selectbox(" ", ("Spending Score Prediction", "Segment Prediction"))
    #Repeat all steps done in Segmentation part to have the data with clusters
    if predzz == "Segment Prediction":
        data.dropna(inplace=True)
        data.drop(columns=['ID', 'Graduated', 'Work_Experience'], inplace=True)
        df = data.copy()
        df=pd.get_dummies(df)
        #Scaling
        scaler = MinMaxScaler()  #Used MinMaxScaler to keep the order of 1 and 0 after dummy encoding
        scaler.fit(df)
        df = pd.DataFrame(scaler.transform(df),columns= df.columns )

        Elbow_M = KElbowVisualizer(KMeans(random_state=1), k=10, metric='silhouette')
        Elbow_M.fit(df)
        AC = KMeans(n_clusters= Elbow_M.elbow_value_, random_state=1)
        # fit model and predict clusters
        y_pred = AC.fit_predict(df)
        df["Clusters"] = y_pred
        #Adding the Clusters feature to the orignal dataframe.
        data["Clusters"]= y_pred
        #Give the client the option to download the created data with clusters
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        csv = convert_df(data)
        st.download_button("Download the New Dataset with Clusters", data=csv, file_name= "Data with Clusters.csv",mime='text/csv')
        #We now have A dataset with clusters gotten from KMeans clustering ready.
        #The purpose of this section is to let the client input the features of a potential or current customer and predict to which cluster he/she belongs
        #To do so, I will use a classification model
        X=data.loc[:, data.columns != 'Clusters']
        y=data['Clusters'].astype("category")
        #Define XGBoost Model

        num_features = X.select_dtypes(include=['int64', 'int32', 'float64'])
        cat_features = X.select_dtypes(include=['object','category'])
        numerical_transformer = Pipeline(steps=[
        ('outliers_removal',RobustScaler(with_centering=False,with_scaling=True)),
        ('num_imputer', SimpleImputer(missing_values = np.nan, strategy='mean')),
        ('normalizer', Normalizer())
        ])
        #Pipeline for categorical features preprocessing (imputing missing values by mode, encoding using OHE Dummies)
        categorical_transformer = Pipeline(steps=[
            ('cat_imputer', SimpleImputer(missing_values = np.nan, strategy='most_frequent')),
            ('encoder',OneHotEncoder(drop='first',handle_unknown='ignore',sparse=False))
            ])
        #Fitting the numerical and categorical features datasets into their corresponding transformers
        transformer = ColumnTransformer( transformers=[
                ('numerical', numerical_transformer, num_features.columns),
                ('categorical',categorical_transformer,cat_features.columns)]
                ,remainder='passthrough')

        def input_features():
            gender= st.radio("Gender", ("Female", "Male"))
            age= st.number_input("Age", 1, 90)
            profession= st.selectbox("Profession", ["Artist", "Healthcare", "Lawyer", "Homemaker", "Marketing", "Engineer", "Doctor", "Executive", "Entertainment"])
            married= st.radio("Married?", ("Yes", "No"))
            family_size= st.number_input("Family Size", 0, 10)
            income= st.number_input("Income in USD", 0, 100000)
            spending= st.radio("Spending Score", ("Low", "Average", "High"))

            data ={'Gender':gender,'Age':age,'Profession':profession, 'Income':income,'Family_Size':family_size, 'Ever_Married':married, 'Spending_Score':spending}

            features=pd.DataFrame(data, index=[0])
            return features

        df1=input_features()
        # Encode the target variable since it is as type integer
        encoder= LabelEncoder()
        y=encoder.fit_transform(y)

        #Split data between X (all features) and y (target feature --> clusters)
        model = LogisticRegression(max_iter=1000, random_state=1)
        ultimate_pipeline= Pipeline(steps=[('transformer', transformer), ('model', model)])
        ultimate_pipeline.fit(X, y)
        pred=ultimate_pipeline.predict(df1)
        #calculate the accuracy of prediction
        X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=1)
        ultimate_pipeline.fit(X_train, y_train)
        y_pred=ultimate_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if st.button("Predict"):
            st.success("Done !")
            st.subheader(f"Your Customer Belongs to Cluster: {str(pred)} with an accuracy of {acc*100}% ")

    if predzz == "Spending Score Prediction":
        X=data.drop(['ID', 'Graduated', 'Work_Experience', 'Spending_Score'], axis=1)
        y=data['Spending_Score'].astype("category")

        #Define XGBoost Model

        num_features = X.select_dtypes(include=['int64', 'int32', 'float64'])
        cat_features = X.select_dtypes(include=['object','category'])
        numerical_transformer = Pipeline(steps=[
        ('outliers_removal',RobustScaler(with_centering=False,with_scaling=True)),
        ('num_imputer', SimpleImputer(missing_values = np.nan, strategy='mean')),
        ('normalizer', Normalizer())
        ])
        #Pipeline for categorical features preprocessing (imputing missing values by mode, encoding using OHE Dummies)
        categorical_transformer = Pipeline(steps=[
            ('cat_imputer', SimpleImputer(missing_values = np.nan, strategy='most_frequent')),
            ('encoder',OneHotEncoder(drop='first',handle_unknown='ignore',sparse=False))
            ])
        #Fitting the numerical and categorical features datasets into their corresponding transformers
        transformer = ColumnTransformer( transformers=[
                ('numerical', numerical_transformer, num_features.columns),
                ('categorical',categorical_transformer,cat_features.columns)]
                ,remainder='passthrough')

        def input_features():
            gender= st.radio("Gender", ("Female", "Male"))
            age= st.number_input("Age", 1, 90)
            profession= st.selectbox("Profession", ["Artist", "Healthcare", "Lawyer", "Homemaker", "Marketing", "Engineer", "Doctor", "Executive", "Entertainment"])
            married= st.radio("Married?", ("Yes", "No"))
            family_size= st.number_input("Family Size", 0, 10)
            income= st.number_input("Income in USD", 0, 100000)
            #spending= st.radio("Spending Score", ("Low", "Average", "High"))

            data ={'Gender':gender,'Age':age,'Profession':profession, 'Income':income,'Family_Size':family_size, 'Ever_Married':married}

            features=pd.DataFrame(data, index=[0])
            return features

        df1=input_features()
        # Encode the target variable since it is as type integer
        encoder= LabelEncoder()
        y=encoder.fit_transform(y)

        #Split data between X (all features) and y (target feature --> clusters)
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', max_iter=1000)
        ultimate_pipeline= Pipeline(steps=[('transformer', transformer), ('model', model)])
        ultimate_pipeline.fit(X, y)
        pred=ultimate_pipeline.predict(df1)
        #calculate the accuracy of prediction
        X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=1)
        ultimate_pipeline.fit(X_train, y_train)
        y_pred=ultimate_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if st.button("Predict"):
            st.success("Done !")
            st.subheader(f"Your Customer's predicted spending score is: {str(pred)} with an accuracy of {acc*100:.2f}% ")
            st.markdown("Legend: 0:Low Spending Score | 1:Average Spending Score | 2:High Spending Score")
