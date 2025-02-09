import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

##################################################
#                    SETTINGS                    #
##################################################

app_logo = 'nmf_logo.png'
st.set_page_config(layout='wide', page_title='NMF Monitoring', page_icon=app_logo)

########################################################
#                    SESSION STATES                    #
########################################################

if 'dataset' not in st.session_state:
    st.session_state['dataset'] = pd.DataFrame()

###################################################
#                    FUNCTIONS                    #
###################################################

def expanding_mad(serie):
    mad_values = []
    for i in range(1, len(serie) + 1):
        subset = serie[:i]
        subset_median = np.nanmedian(subset)

        if np.isnan(subset_median):
            mad = 0
        else:
            deviations = np.absolute(subset - subset_median)
            mad = np.nanmedian(deviations)
        mad_values.append(mad)
    
    return mad_values

def apply_style_df(x, colors_list):
    colors = colors_list
    df_empty = pd.DataFrame(colors, index=x.index, columns=x.columns)
    return df_empty

def plot_rsi(data, player):
    data_p = data[data['Name']==player].copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=data_p['Date'], y=data_p['RSI'], opacity=.7,
                         marker=dict(color='#026bcf'), name='RSI'))
    fig.add_trace(go.Scatter(x=data_p['Date'], y=data_p['RSI_exp_median'],
                             mode='lines', line=dict(shape='spline', color='grey'), name='Cumulative median',
                             stackgroup='one'))
    fig.update_layout(hovermode='x', height=300, 
                      legend=dict(orientation='h', yanchor='bottom', xanchor='right', y=1.01, x=1),
                      margin=dict(l=10, r=10, b=50, t=40))
    return fig

def plot_relpow(data, player):
    data_p = data[data['Name']==player].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_p['Date'], y=data_p['Relative Power'], opacity=.7,
                             mode='lines+markers', line=dict(shape='spline', color='#FFCC33'), name='Relative Power'))
    fig.update_layout(hovermode='x', height=345, title='Relative Power', yaxis_range=[0, 90],
                      margin=dict(l=10, r=10, b=50, t=40))
    return fig

def plot_cumul_cv(data, player, metric):
    data_p = data[data['Name']==player].copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=data_p['Date'], y=data_p[f'{metric}_CV'], opacity=.7,
                         marker=dict(color='#FFCC33'), name=metric))
    fig.update_layout(hovermode='x', height=150, title=f'Cumulative CV(%) - {metric}',
                      margin=dict(l=10, r=10, b=50, t=40))
    return fig

def load_file():
    try:
        st.session_state['dataset'] = pd.read_excel(st.session_state['file'])
    except:
        st.session_state['dataset'] = pd.DataFrame()
    
    if not st.session_state['dataset'].empty:
        #Change Player Names
        st.session_state['dataset']['Name'] = st.session_state['dataset']['Name'].replace({n: f'Player_{i}' for i, n in enumerate(st.session_state['dataset']['Name'].unique())})

        st.session_state['dataset'] = st.session_state['dataset'][st.session_state['dataset']['RSI'].notna()].reset_index(drop=True).copy()
        st.session_state['dataset'] = st.session_state['dataset'].sort_values(by='Date', ignore_index=True)

        #Cumulative MAD
        st.session_state['dataset']['RSI_mad'] = st.session_state['dataset'].groupby('Name')['RSI'].transform(expanding_mad)
        st.session_state['dataset']['Relative Power_mad'] = st.session_state['dataset'].groupby('Name')['Relative Power'].transform(expanding_mad)
        #Expanding median
        st.session_state['dataset']['RSI_exp_median'] = st.session_state['dataset'].groupby('Name')['RSI'].transform(lambda x: x.expanding().median())
        st.session_state['dataset']['Relative Power_exp_median'] = st.session_state['dataset'].groupby('Name')['Relative Power'].transform(lambda x: x.dropna().expanding().median())
        #CV
        st.session_state['dataset']['RSI_CV'] = round(st.session_state['dataset']['RSI_mad'] / st.session_state['dataset']['RSI_exp_median'] *100, 1)
        st.session_state['dataset']['Relative Power_CV'] = round(st.session_state['dataset']['Relative Power_mad'] / st.session_state['dataset']['Relative Power_exp_median'] *100, 1)
        #Color codes
        st.session_state['dataset'].loc[st.session_state['dataset']['RSI'] <= (st.session_state['dataset']['RSI_exp_median'] - st.session_state['dataset']['RSI_mad']), 'RSI_color'] = '#ffc09f' #orange
        st.session_state['dataset'].loc[(st.session_state['dataset']['RSI'] > (st.session_state['dataset']['RSI_exp_median'] - st.session_state['dataset']['RSI_mad']))
                                        & (st.session_state['dataset']['RSI'] < (st.session_state['dataset']['RSI_exp_median'] + st.session_state['dataset']['RSI_mad'])), 'RSI_color'] = '#a0ced9' #bleu
        st.session_state['dataset'].loc[st.session_state['dataset']['RSI'] >= (st.session_state['dataset']['RSI_exp_median'] + st.session_state['dataset']['RSI_mad']), 'RSI_color'] = '#adf7b6' #vert

        st.session_state['dataset'].loc[st.session_state['dataset']['Relative Power'] <= (st.session_state['dataset']['Relative Power_exp_median'] - st.session_state['dataset']['Relative Power_mad']), 'Relative Power_color'] = '#ffc09f' #orange
        st.session_state['dataset'].loc[(st.session_state['dataset']['Relative Power'] > (st.session_state['dataset']['Relative Power_exp_median'] - st.session_state['dataset']['Relative Power_mad']))
                                        & (st.session_state['dataset']['Relative Power'] < (st.session_state['dataset']['Relative Power_exp_median'] + st.session_state['dataset']['Relative Power_mad'])), 'Relative Power_color'] = '#a0ced9' #bleu
        st.session_state['dataset'].loc[st.session_state['dataset']['Relative Power'] >= (st.session_state['dataset']['Relative Power_exp_median'] + st.session_state['dataset']['Relative Power_mad']), 'Relative Power_color'] = '#adf7b6' #vert

        for p in st.session_state['dataset']['Name'].unique():
            st.session_state[f'{p}_rsi_graph'] = plot_rsi(st.session_state['dataset'], p)
            st.session_state[f'{p}_rp_graph'] = plot_relpow(st.session_state['dataset'], p)
            st.session_state[f'{p}_rsi_ccv'] = plot_cumul_cv(st.session_state['dataset'], p, 'RSI')
            st.session_state[f'{p}_rp_ccv'] = plot_cumul_cv(st.session_state['dataset'], p, 'Relative Power')

            p_data = st.session_state['dataset'][st.session_state['dataset']['Name']==p].sort_values(by='Date', ascending=False).copy()
            colors_ = list()
            for i, row in p_data.iterrows():
                colors_.append([f"background-color: {row['RSI_color']}; color: black",
                                f"background-color: {row['Relative Power_color']}; color: black"])
                
            st.session_state[f'{p}_tbl'] = (p_data.set_index('Date')[['RSI', 'Relative Power']]
                                            .style.apply(apply_style_df, colors_list=colors_, axis=None)
                                            .format(precision=2))
            
        st.session_state['last_cv'] = st.session_state['dataset'].groupby('Name', as_index=False).tail(1)

#############################################
#                    APP                    #
#############################################

with st.sidebar:
    st.file_uploader('Drop File', key='file', on_change=load_file)

with st.container(border=True):
    c_title = st.columns([.1, .9])
    with c_title[1]:
        st.title('Force Plates - Neuromuscular Fatigue Monitoring')
        st.write('')

if not st.session_state['dataset'].empty:
    c_dash = st.columns([.3, .7])
    with c_dash[0]:
        st.selectbox('Player Name', st.session_state['dataset']['Name'].unique(), index=0, key='player_sel')
        st.dataframe(st.session_state[f"{st.session_state['player_sel']}_tbl"], height=362)

        p_df = st.session_state['last_cv'][st.session_state['last_cv']['Name']==st.session_state['player_sel']].copy()

        c_left = st.columns(2)
        with c_left[0]:
            st.metric('CV% RSI - Player', value=str(p_df['RSI_CV'].values[0]), border=True)
            st.metric('CV% Rel.Pwr - Player', value=str(p_df['Relative Power_CV'].values[0]), border=True)
        with c_left[1]:
            st.metric('CV% RSI - Team', value=str(np.nanmedian(st.session_state['last_cv']['RSI_CV'])), border=True)
            st.metric('CV% Rel.Pwr - Team', value=str(np.nanmedian(st.session_state['last_cv']['Relative Power_CV'])), border=True)

    with c_dash[1]:
        st.plotly_chart(st.session_state[f"{st.session_state['player_sel']}_rsi_graph"])

        c_right = st.columns(2, vertical_alignment='center')
        with c_right[0]:
            with st.container(border=True):
                st.plotly_chart(st.session_state[f"{st.session_state['player_sel']}_rsi_ccv"])
            with st.container(border=True):
                st.plotly_chart(st.session_state[f"{st.session_state['player_sel']}_rp_ccv"])
        with c_right[1]:
            with st.container(border=True):
                st.plotly_chart(st.session_state[f"{st.session_state['player_sel']}_rp_graph"])