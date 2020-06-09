import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px
import dash
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import dash_table
import pandas as pd
import json
from dash.exceptions import PreventUpdate
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# all the dash component follo the naming as camel case like dropHiddenLayer1
# where drop signifies its a dropdown, hidden layer 1 species where and for what
# purpose the component has been kept
# component has been placed sequentially in the order of sequence in the actual layout
# same is true for the callback
# Materialise CSS has been used with S12 Grid system

#%%
class NormalizationTensor():
    '''normalize the data to be inserted into neural network and store
    mean and std for usage later on in an object, also exponential to target is applied by default,
    during conversion'''
    def __init__(self, feature_array, target_array):
        #fit the feature_array
        self.feature_mean = np.mean(feature_array, axis = 0, keepdims = True)
        self.feature_std = np.std(feature_array, axis = 0, keepdims = True)
        
        #fit the target_array
        target_array = np.log(target_array)
        self.target_mean = np.mean(target_array, axis = 0, keepdims = True)
        self.target_std = np.std(target_array, axis = 0, keepdims = True)
    def f2n(self,input_array):
        '''convert actual feature to normalised feature'''
        return (input_array - self.feature_mean) / self.feature_std
    def t2n(self,input_array):
        '''convert actual target to normalised target,
        also natural log is applied by default
        (because target of engine should be handeled in exponential space)'''
        array = np.log(input_array)
        return (array - self.target_mean) / self.target_std
    def n2f(self,input_array):
        '''convert normalised feature to actual feature'''
        return input_array*self.feature_std + self.feature_mean
    def n2t(self,input_array):
        '''convert normalised target to actual target,
        also exponential function is applied by default
        (because target of engine should be handeled in exponential space)'''
        array = input_array*self.target_std + self.target_mean
        return np.exp(array)
    def t_t2n(self,input_array):
        '''convert actual target tensor to normalised target tensor,
        also natural log is applied by default
        (because target of engine should be handeled in exponential space)'''
        tensor = tf.math.log(input_array)
        return (tensor - self.target_mean) / self.target_std
    def t_n2t(self,input_array):
        '''convert normalised target tensor to actual target tensor,
        also exponential function is applied by default
        (because target of engine should be handeled in exponential space)'''
        tensor = input_array*self.target_std + self.target_mean
        return tf.math.exp(tensor)
    def t_f2n(self,input_array):
        '''convert actual feature to normalised feature'''
        return (input_array - self.feature_mean) / self.feature_std
    def t_n2f(self,input_array):
        '''convert normalised feature to actual feature'''
        return input_array*self.feature_std + self.feature_mean
    @classmethod
    def from_stored_value(cls, feature_mean, feature_std, target_mean, target_std):
        instance = cls([0], [1])
        instance.feature_mean = feature_mean
        instance.feature_std = feature_std
        instance.target_mean = target_mean
        instance.target_std = target_std
        return instance

def model1_prep(layer_nums, add_dropout):
    '''
    layer_nums is like [10,20,30]
    means neural network with 3 layers
    10 input nodes
    20 hidden nodes
    30 outputs
    '''
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    first_hidden_layer = [Dense(layer_nums[1], input_shape=(layer_nums[0],), activation = 'tanh')]
    if add_dropout == True:
        dropout_layer = [Dropout(0.2)]
    else:
        dropout_layer = []
    other_layers = [Dense(i, activation='tanh') for i in layer_nums[2:-1]]
    other_layers = other_layers + [Dense(layer_nums[-1], activation='linear')]
    model1 = Sequential(first_hidden_layer + dropout_layer + other_layers)
    model1.compile('adam', 'mse')
    return model1

def model_weights_to_json(numpy_weights):
    json_weights = []
    for weights in numpy_weights:
        json_weights.append(weights.tolist())
    return json.dumps(json_weights)

def json_to_model_weights(json_weights):
    json_weights = json.loads(json_weights)
    numpy_weights = []
    for weights in json_weights:
        numpy_weights.append(np.array(weights))
    return numpy_weights

def load_app_state(to_be_updated, filename):
    with open(filename, "r") as f:
        saved_data = json.load(f)
    output = []
    for comp, prop in to_be_updated:
        for saved_dict in saved_data:
            if (saved_dict['component'] == comp) and (saved_dict['property'] == prop):
                output.append(saved_dict['value'])
    return output

def consolidated_df(x_trn, x_vld, y_trn, y_vld, model1, N):
    '''
    take train and test set from data frame, predict the model output on same using feature
    and gives out dataframe as tidy dataframe , for the plotly express
    '''
    y_trn = y_trn.copy()
    y_trn_pred = y_trn.copy()
    y_trn_pred.loc[:,:] = N.n2t(model1.predict(N.f2n(x_trn.values)))
    y_trn_pred = y_trn_pred.astype('float')
    y_trn['set'] = 'train'
    y_trn_pred['set'] = 'train'
    
    y_vld = y_vld.copy()
    y_vld_pred = y_vld.copy()
    y_vld_pred.loc[:,:] = N.n2t(model1.predict(N.f2n(x_vld.values)))
    y_vld_pred = y_vld_pred.astype('float')
    y_vld['set'] = 'test'
    y_vld_pred['set'] = 'test'
    
    y_actual = pd.concat([y_trn, y_vld], axis=0)
    y_pred = pd.concat([y_trn_pred, y_vld_pred], axis=0)
    
    df_actual = y_actual.melt(id_vars=['set'], value_name='actual', var_name='emission')
    df_pred = y_pred.melt(id_vars=['set'], value_name='predict', var_name='emission')
    df = df_actual.copy()
    df['predict'] = df_pred['predict']
    return df

def operating_point_plotter(rpm_max, inj_max, rpm, inj, rpm_text, inj_text):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[rpm], y=[inj],
        name='operating point',
        mode='markers',
        marker_color='red',
        marker_size=15
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, rpm_max, rpm_max], y=[inj_max, inj_max, 0],
        name='extreme limits',
        mode='lines',
        marker_color='blue'
    ))
    
    fig.update_layout(height=350, width=400,
                      xaxis_title_text=rpm_text, yaxis_title_text=inj_text)
    fig.update_xaxes(range=(0, rpm_max*1.5), showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(range=(0, inj_max*1.5), showline=True, linewidth=2, linecolor='black', mirror=True)
    return fig


#%%dash app
app = dash.Dash(__name__)
app.title = 'AI DieselEngine'

nameProject = dcc.Input(id='nameProject')
loadProject = dcc.Upload(id='loadProject')

dataColumnFeature = dash_table.DataTable(
        #Column id is just for visibility on the webpage
        #While constructing data id is used instead of name and same goes as dataframe
        id='dataColumnFeature',
        columns=([{'id': 'Column 1', 'name': 'Column 1'},
                  {'id': 'Column 2', 'name': 'Column 2'}]),
        data=[{'Column 1': 'Column 1', 'Column 2': 'Column 2'}],
        editable=True,
        style_table={'overflowX': 'auto'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
        style_header={'backgroundColor': 'rgb(230, 230, 230)','fontWeight': 'bold'})

dataColumnTarget = dash_table.DataTable(
        id='dataColumnTarget',
        columns=([{'id': 'Column 1', 'name': 'Column 1'},
                  {'id': 'Column 2', 'name': 'Column 2'}]),
        data=[{'Column 1': 'Column 1', 'Column 2': 'Column 2'}],
        editable=True,
        style_table={'overflowX': 'auto'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
        style_header={'backgroundColor': 'rgb(230, 230, 230)','fontWeight': 'bold'})

dataTableFeature = dash_table.DataTable(
        id='dataTableFeature',
        columns=([{'id': 'Column 1', 'name': 'Column 1'},
                  {'id': 'Column 2', 'name': 'Column 2'}]),
        data=[{'Column 1': 1.1, 'Column 2': 1.2}],
        editable=True,
        style_table={'overflowX': 'auto', 'height': '300px'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
        style_header={'backgroundColor': 'rgb(230, 230, 230)','fontWeight': 'bold'})

dataTableTarget = dash_table.DataTable(
        id='dataTableTarget',
        columns=([{'id': 'Column 1', 'name': 'Column 1'},
                  {'id': 'Column 2', 'name': 'Column 2'}]),
        data=[{'Column 1': 1.3, 'Column 2': 1.4}],
        editable=True,
        style_table={'overflowX': 'auto', 'height': '300px'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
        style_header={'backgroundColor': 'rgb(230, 230, 230)','fontWeight': 'bold'})


saveDataButton = html.Button('Save Session', id='saveDataButton',className='btn waves-effect waves-light')
openDataButton = dcc.Upload('Load session', id='openDataButton',className='btn waves-effect waves-light', multiple=False)
transferButton = html.Button('transfer column names', id='transferButton', className='btn waves-effect waves-light')

dropDataX = dcc.Dropdown(id='dropDataX',
                              options=[{'label': str(i), 'value': i} for i in range(1,3)],
                              value=None)
dropDataY = dcc.Dropdown(id='dropDataY',
                              options=[{'label': str(i), 'value': i} for i in range(1,3)],
                              value=None)
dropDataColor = dcc.Dropdown(id='dropDataColor',
                              options=[{'label': str(i), 'value': i} for i in range(1,3)],
                              value=None)

dropHiddenLayer1 = dcc.Dropdown(id='dropHiddenLayer1',
                              options=[{'label': str(i), 'value': i} for i in range(1,50)],
                              value=None)
dropHiddenLayer2 = dcc.Dropdown(id='dropHiddenLayer2',
                              options=[{'label': str(i), 'value': i} for i in range(1,50)],
                              value=None)
dropHiddenLayer3 = dcc.Dropdown(id='dropHiddenLayer3',
                              options=[{'label': str(i), 'value': i} for i in range(1,50)],
                              value=None)
dropHiddenLayer4 = dcc.Dropdown(id='dropHiddenLayer4',
                              options=[{'label': str(i), 'value': i} for i in range(1,50)],
                              value=None)
dropHiddenLayer5 = dcc.Dropdown(id='dropHiddenLayer5',
                              options=[{'label': str(i), 'value': i} for i in range(1,50)],
                              value=None)
dropEngineRpm = dcc.Dropdown(id='dropEngineRpm')
dropInjectionQty = dcc.Dropdown(id='dropInjectionQty')
dropCoolantTemp = dcc.Dropdown(id='dropCoolantTemp')

slideTrainTestSplit = dcc.Slider(id='slideTrainTestSplit',min=0, max=1,step=0.01,value=0.8,
                                 marks={i: f'{i*100} %' for i in [0,0.25,0.5,0.75]})
dropNumIter = dcc.Dropdown(id='dropNumIter',
                              options=[{'label': str(i), 'value': i} for i in [10, 100, 500, 1000]],
                              value=None)
butTrain = html.Button('Train the maodel', id='butTrain',className='btn waves-effect waves-light')

divSlideModelExploration = html.Div(id='divSlideModelExploration')
divGaugeModelExploration = html.Div(id='divGaugeModelExploration')

storeSave = html.Div(id='storeSave') #store string as 'saved' if value is saved in the session, almost like adummy
storeModelSize = html.Div(id='storeModelSize') #list of numbers of nodes in neural network including the feature and target
storeTrainFeature = html.Div(id='storeTrainFeature') #string column name
storeTrainTarget = html.Div(id='storeTrainTarget') #dataframe ordered
storeTestFeature = html.Div(id='storeTestFeature') #dataframe ordered
storeTestTarget = html.Div(id='storeTestTarget') #dataframe ordered
storeModel1Weights = html.Div(id='storeModel1Weights') #dataframe ordered
storeN = html.Div(id='storeN', children='None') #feature and target normaliser
storePredictionFeature = html.Div(id='storePredictionFeature', children='None')


storeAll = html.Div([storeSave, storeModelSize, storeTrainFeature, storeTrainTarget, storeTestFeature,
                     storeTestTarget, storeModel1Weights, storeN,
                     storePredictionFeature], style={'display': 'none'})

app.layout = html.Div([html.Nav(html.H3('Artificial Intelligence for Diesel Engine Calibration'), className='nav-wraper indigo'), 
                           html.Div([html.Div(saveDataButton, className='col offset-s9'),
                                     html.Div(openDataButton)], className='row'),
                           html.H4('Load the engine doe data', className='blue darken-3 white-text'),
                           html.Div([html.Div('Features :', className='col s6 card-panel teal lighten-2'),
                                     html.Div('Targets :', className='col s6 card-panel teal lighten-2')], className='row'),
                           html.Div([html.Div(dataColumnFeature, className='col s6'),
                                     html.Div(dataColumnTarget, className='col s6')], className='row grey lighten-4'),
                           html.Div([html.Div(transferButton, className='col s2 offset-s5')] ,className='row'),
                           html.Div([html.Div(dataTableFeature, className='col s6'),
                                     html.Div(dataTableTarget, className='col s6')], className='row grey lighten-4'),
                           html.Div(className='divider'),
                           html.H4('Explore Engine DOE data', className='blue darken-3 white-text'),
                           html.Div([html.Div(['X Axis Value: ',
                                               dropDataX,
                                               'Y Axis Value',
                                               dropDataY,
                                               'Color Selection',
                                               dropDataColor],className='col s2'),
                                     html.Div(id = 'graphDataScatter',className='col s10')], className='row grey lighten-4'),
                           html.Div(className='divider'),
                           html.H4('Make Artificial Intelligence Neural Network', className='blue darken-3 white-text'),
                           html.Div([html.Div(['Hidden Layer1',
                                               dropHiddenLayer1,
                                               'Hidden Layer 2',
                                               dropHiddenLayer2,
                                               'Hidden Layer 3',
                                               dropHiddenLayer3,
                                               'Hidden Layer 4',
                                               dropHiddenLayer4,
                                               'Hidden Layer 5',
                                               dropHiddenLayer5],className='col s2'),
                                     html.Div(id = 'graphNetwork',className='col s10')], className='row grey lighten-4'),
                           html.Div(className='divider'),
                           html.H4('Train Artificial Intelligence Neural Network', className='blue darken-3 white-text'),
                           html.Div([html.Div(['Engine RPM Label :',
                                              dropEngineRpm,
                                              'Injection Quantity label :',
                                              dropInjectionQty,
                                              'Coolant Temperature Label :',
                                              dropCoolantTemp,
                                              'No of samples for training the model :',
                                              slideTrainTestSplit,
                                              'No of Model training steps',
                                              dropNumIter,
                                              html.Div('.'),
                                              butTrain], className='col s2'),
                                    html.Div(id = 'graphTraining', className='col s10')],className='row grey lighten-4'),
                           html.Div(className='divider'),
                           html.H4('Offline Engine Dyno', className='blue darken-3 white-text'),
                           html.Div([html.Div(divSlideModelExploration, className='col s2'),
                                     html.Div([html.Div(id='divOPModelExploration'),
                                               divGaugeModelExploration],
                                              className='col s10')],className='row grey lighten-4'),
                           storeAll])

@app.callback([Output('dataTableFeature', 'data'),
               Output('dataTableFeature', 'columns'),
               Output('dataTableTarget', 'data'),
               Output('dataTableTarget', 'columns')],
              [Input('transferButton', 'n_clicks'),
               Input('openDataButton', 'filename')],
              [State('dataColumnFeature', 'data'),
               State('dataTableFeature', 'data'),
               State('dataColumnTarget', 'data'),
               State('dataTableTarget', 'data')])
def update_base_data(n_clicks, filename, df_from1, df_to1, df_from2, df_to2):
    ctx = dash.callback_context
    if ctx.triggered:
        button_pressed = ctx.triggered[0]['prop_id'].split('.')[0]
    else:
        button_pressed = None
    
    if button_pressed == 'transferButton' or button_pressed == None:
        def update_feature_column(n_clicks, df_from, df_to):
            df_from = pd.DataFrame(df_from)
            df_to = pd.DataFrame(df_to)
            df = pd.DataFrame(df_to.values, columns=df_from.values.flatten().tolist())
            columns = [{'id': str(c), 'name': str(c)} for c in df.columns]
            return df.to_dict('records'), columns
        output1 = update_feature_column(n_clicks, df_from1, df_to1)
        output2 = update_feature_column(n_clicks, df_from2, df_to2)
        return output1 + output2
    
    if button_pressed == 'openDataButton':
        if filename != None:
            to_be_updated = [('dataTableFeature', 'data'),
                             ('dataTableFeature', 'columns'),
                             ('dataTableTarget', 'data'),
                             ('dataTableTarget', 'columns')]            
            return load_app_state(to_be_updated, filename)

@app.callback([Output('dropDataX', 'options'),
               Output('dropDataY', 'options'),
               Output('dropDataColor', 'options')],
              [Input('dataTableFeature', 'data'),
               Input('dataTableTarget', 'data')])
def update_dropdown_data_exploration(data_feature, data_target):
    df_feature = pd.DataFrame(data_feature, dtype='float')
    df_target = pd.DataFrame(data_target, dtype='float')
    df = pd.concat([df_feature, df_target], axis=1)
    options = [{'label': str(i), 'value': i} for i in list(df.columns)]
    return [options]*3


@app.callback([Output('graphDataScatter', 'children')],
              [Input('dropDataX', 'value'),
               Input('dropDataY', 'value'),
               Input('dropDataColor', 'value')],
              [State('dataTableFeature', 'data'),
               State('dataTableTarget', 'data')])
def update_data_graph(value_x, value_y, value_color, data_feature, data_target):
    df_feature = pd.DataFrame(data_feature,dtype='float')
    df_target = pd.DataFrame(data_target,dtype='float')
    df = pd.concat([df_feature, df_target], axis=1)
    if ((value_x != None) and (value_y != None) and (value_color != None)):
        fig = px.scatter(df, x=value_x, y=value_y, color=value_color, color_continuous_scale=px.colors.diverging.RdYlGn_r, height=600)
        graph1 = dcc.Graph(figure = fig)
        
        fig = go.Figure(go.Histogram2dContour(
        x = df[value_x],
        y = df[value_y],
        z= df[value_color],
        histfunc='avg',
        colorscale = 'hot_r',
        contours = dict(
            start=0,
            end=df[value_color].max()*1.1,
            size=df[value_color].max()/10,
            showlabels = True,
            labelfont = dict(
                family = 'Raleway',
                color = 'blue',
                size = 22)),
        hoverlabel = dict(
            bgcolor = 'white',
            bordercolor = 'black',
            font = dict(
                family = 'Raleway',
                color = 'black'))))
        fig.update_layout(xaxis_title_text=value_x, yaxis_title_text=value_y, height=600)
        graph2 = dcc.Graph(figure = fig)
        
        return [[graph1, graph2]]
    else:
        raise PreventUpdate

@app.callback([Output('graphNetwork', 'children'),
               Output('storeModelSize', 'children')],
               [Input('dropHiddenLayer1', 'value'),
                Input('dropHiddenLayer2', 'value'),
                Input('dropHiddenLayer3', 'value'),
                Input('dropHiddenLayer4', 'value'),
                Input('dropHiddenLayer5', 'value'),
                Input('dataTableFeature', 'data'),
                Input('dataTableTarget', 'data')])
def update_network(*layers):
    len_feature = [len(layers[-2][0])]
    len_target = [len(layers[-1][0])]
    layers = list(layers[:-2])
    layers = len_feature + layers + len_target
    layers = [i for i in layers if i is not None]
    if len(layers) > 0:
        max_nodes = max(layers)
        def gen_ypoint(num_nodes, max_nodes):
            empty_spaces = max_nodes - num_nodes
            empty_space_bottom = int(empty_spaces/2)
            y_point = list(range(empty_space_bottom, empty_space_bottom+num_nodes))
            y_point_modified = []
            offset = (empty_spaces%2)/2
            for points in y_point:
                y_point_modified.append(points + offset)
            return y_point_modified
        
        point_traces = []
        for i, _ in enumerate(layers[:-1]):
            x_points_curr = [i]*layers[i]
            y_points_curr = gen_ypoint(layers[i], max_nodes)
            x_points_next = [i+1]*layers[i+1]
            y_points_next = gen_ypoint(layers[i+1], max_nodes)
            x_points = []
            y_points = []
            for id_curr in range(layers[i]):
                for id_next in range(layers[i+1]):
                    x_points.append(x_points_curr[id_curr])
                    y_points.append(y_points_curr[id_curr])
                    x_points.append(x_points_next[id_next])
                    y_points.append(y_points_next[id_next])
            point_traces.append(go.Scatter(
                x=x_points, y=y_points,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines+markers',
                marker_size=5,
                marker_color='rgb(31, 119, 180)'))
        fig = go.Figure(data=point_traces)
        fig.update_layout(height=700)
        graph = dcc.Graph(figure = fig, config={'staticPlot': True})
        return [graph], json.dumps(layers)
    else:
        raise PreventUpdate

@app.callback([Output('dropEngineRpm', 'options'),
               Output('dropInjectionQty', 'options'),
               Output('dropCoolantTemp', 'options')],
              [Input('dataTableFeature', 'data')])
def update_dropdown_rpm_selection(data_feature):
    df_feature = pd.DataFrame(data_feature,dtype='float')
    options = [{'label': str(i), 'value': i} for i in list(df_feature.columns)]
    return [options]*3


@app.callback([Output('storeTrainFeature', 'children'),
               Output('storeTestFeature', 'children'),
               Output('storeTrainTarget', 'children'),
               Output('storeTestTarget', 'children'),
               Output('storeModel1Weights', 'children'),
               Output('storeN', 'children'),
               Output('graphTraining', 'children')],
              [Input('butTrain', 'n_clicks')],
              [State('dropEngineRpm', 'value'),
               State('dropInjectionQty', 'value'),
               State('dropCoolantTemp', 'value'),
               State('slideTrainTestSplit', 'value'),
               State('dropNumIter', 'value'),
               State('dataTableFeature', 'data'),
               State('dataTableTarget', 'data'),
               State('storeModelSize', 'children')])
def train_the_model_plot_the_graph(n_clicks, rpm_name, inj_name, tmp_name, train_size, epochs, data_feature, data_target, model_size):
    if n_clicks is not None:
        #aligning engine rpm, injection qty and coolant temperature
        df_feature = pd.DataFrame(data_feature,dtype='float')
        df_target = pd.DataFrame(data_target,dtype='float')        
        r_i_t = [rpm_name, inj_name, tmp_name]
        revised_feature_column = r_i_t + [i for i in df_feature.columns if i not in r_i_t]
        df_feature_revised = df_feature.reindex(columns=revised_feature_column)
        
        #filter in only non-zero positive value in target
        filt = df_target.min(axis=1) > 0
        df_feature_revised = df_feature_revised.loc[filt, :]
        df_target = df_target.loc[filt, :]
        x_trn, x_vld, y_trn, y_vld = train_test_split(df_feature_revised, df_target, test_size=1-train_size)
        N = NormalizationTensor(x_trn.values, y_trn.values)
        model_size = json.loads(model_size)
        model1 = model1_prep(model_size, add_dropout=True)
        model1.fit(N.f2n(x_trn.values), N.t2n(y_trn.values),
                   validation_data=(N.f2n(x_vld.values), N.t2n(y_vld.values)),
                   epochs=epochs, batch_size=500)
        model1_weights = model1.get_weights()
        model1_weights_as_json = model_weights_to_json(model1_weights)
        
        df = consolidated_df(x_trn, x_vld, y_trn, y_vld, model1, N)
        fig = px.scatter(df, x='actual', y='predict', facet_col='emission', facet_col_wrap=5, color='set')
        fig.update_layout(height=1700)
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_xaxes(matches=None, showticklabels=True)
        graph = dcc.Graph(figure = fig)
        N_json = json.dumps({'feature_mean' : N.feature_mean.tolist(),
                              'feature_std' : N.feature_std.tolist(),
                              'target_mean' : N.target_mean.tolist(),
                              'target_std' : N.target_std.tolist()})
        return [i.to_json() for i in [x_trn, x_vld, y_trn, y_vld]] + [model1_weights_as_json] + [N_json] + [graph]
    else:
        raise PreventUpdate

@app.callback([Output('divSlideModelExploration', 'children')],
              [Input('storeTrainFeature', 'children')])
def explore_model_make_sliders(data_feature):
    df_feature = pd.read_json(data_feature)
    def make_sliders(df_feature):
        column_names = list(df_feature.columns)
        min_features = df_feature.min(axis=0).values
        max_features = df_feature.max(axis=0).values
        sliders = []
        for i, (name, min_value, max_value) in enumerate(zip(column_names, min_features, max_features)):
            slide_name = html.Div([html.Div(name + ' : ', className='col s8'),
                                   html.Div(id = {'type' : 'slideNameModelExploration',
                                                 'index' : name},children = 0, className='col s4 blue-text')], className='row')
            slider = dcc.Slider(id={'type':'slideModelExploration',
                                    'index': name},
                                min=min_value,
                                max=max_value,
                                value=min_value,
                                step=(max_value-min_value)/100,
                                marks={i: str(np.round(i,2)) for i in np.linspace(min_value, max_value, 5, endpoint=True)})
            sliders.append(html.Div([slide_name, slider]))
        return [html.Div(sliders)]
    return make_sliders(df_feature)

@app.callback([Output({'type': 'slideNameModelExploration', 'index': MATCH}, 'children')],
               [Input({'type': 'slideModelExploration', 'index': MATCH}, 'value')])
def explore_model_update_slider_name(value):
    return [str(np.round(value,2))]

@app.callback([Output('storePredictionFeature', 'children')],
               [Input({'type': 'slideModelExploration', 'index': ALL}, 'value')])
def explore_model_update_feature_to_be_predicted(values):
    if len(values) == 0:
        raise PreventUpdate
    else:
        return [json.dumps(values)]

@app.callback([Output('divOPModelExploration', 'children')],
               [Input('storePredictionFeature', 'children')],
               [State('storeTrainFeature', 'children')])
def explore_model_update_operating_point(values, df_feature):
    if values == None:
        raise PreventUpdate
    else:
        df_feature = pd.read_json(df_feature)
        rpm_max = df_feature.iloc[:,0].max()
        inj_max = df_feature.iloc[:,1].max()
        rpm = json.loads(values)[0]
        inj = json.loads(values)[1]
        rpm_text = df_feature.columns[0]
        inj_text = df_feature.columns[1]
        return [dcc.Graph(figure = operating_point_plotter(rpm_max, inj_max, rpm, inj, rpm_text, inj_text),
                          config={'staticPlot': True})]

@app.callback([Output('divGaugeModelExploration', 'children')],
               [Input('storePredictionFeature', 'children')],
               [State('storeTrainTarget', 'children'),
                State('storeModelSize', 'children'),
                State('storeModel1Weights', 'children'),
                State('storeN', 'children')])
def explore_model_make_gauges(*values):
    if values[-5] == None:
        raise PreventUpdate
    feature_values = json.loads(values[-5])
    
    data_target = values[-4]
    model_length = json.loads(values[-3])
    model_weights = json.loads(values[-2])
    model_weights = [np.array(i) for i in model_weights]
    N_values = json.loads(values[-1])
    N = NormalizationTensor.from_stored_value(np.array(N_values['feature_mean']),
                                              np.array(N_values['feature_std']),
                                              np.array(N_values['target_mean']),
                                              np.array(N_values['target_std']))

    model1 = model1_prep(model_length, add_dropout=False)
    model1.set_weights(model_weights)
    
    prediction = N.n2t(model1.predict(N.f2n(feature_values)))

    df_target = pd.read_json(data_target)
    column_names = list(df_target.columns)
    min_targets = df_target.min(axis=0).values
    max_targets = df_target.max(axis=0).values
    gauges = []
    for i, (name, min_value, max_value) in enumerate(zip(column_names, min_targets, max_targets)):
        gap = (max_value - min_value)/3
        green = [min_value, min_value+gap]
        yellow = [min_value+gap, min_value + 2*gap]
        red = [min_value + 2*gap, min_value + 3*gap]
        gauge = daq.Gauge(
            id={'type' : 'gaugeModelExploration',
                'index': name},
            label=name,
            min = min_value,
            max = max_value,
            value=prediction[0,i],
            showCurrentValue=True,
            color={"gradient":True,"ranges":{"green":green,"yellow":yellow,"red":red}},
            size=100,
            scale={'start': min_value, 'interval': (max_value-min_value)/8}
        )
        gauges.append(html.Div(gauge, className='col s1'))
    return [html.Div(gauges, className='row')]

save_list = [('dataTableFeature', 'data'),
             ('dataTableFeature', 'columns'),
             ('dataTableTarget', 'data'),
             ('dataTableTarget', 'columns'),
             ('dataColumnFeature', 'data'),
             ('dataColumnFeature', 'columns'),
             ('dataColumnTarget', 'data'),
             ('dataColumnTarget', 'columns'),
             ('dropDataX', 'value'),
             ('dropDataY', 'value'),
             ('dropDataColor', 'value'),
             ('dropHiddenLayer1', 'value'),
             ('dropHiddenLayer2', 'value'),
             ('dropHiddenLayer3', 'value'),
             ('dropHiddenLayer4', 'value'),
             ('dropHiddenLayer5', 'value'),
             ('dropEngineRpm', 'value'),
             ('dropInjectionQty', 'value'),
             ('dropCoolantTemp', 'value'),
             ('slideTrainTestSplit', 'value'),
             ('dropNumIter', 'value')]
app_states = [State(comp, prop) for comp, prop in save_list]
@app.callback(Output('storeSave', 'children'),
              [Input('saveDataButton', 'n_clicks')],
              app_states)
def save_app_state(n_clicks, *states):
    if n_clicks != None:
        saved_data = []
        for i, (comp, prop) in enumerate(save_list):
            saved_data.append({'component': comp,'property': prop, 'value': states[i]})
        with open("app_state.json", "w") as write_file:
            json.dump(saved_data, write_file)
    return 'saved'

upload_list = [('dataColumnFeature', 'data'),
               ('dataColumnFeature', 'columns'),
               ('dataColumnTarget', 'data'),
               ('dataColumnTarget', 'columns'),
               ('dropDataX', 'value'),
               ('dropDataY', 'value'),
               ('dropDataColor', 'value'),
               ('dropHiddenLayer1', 'value'),
               ('dropHiddenLayer2', 'value'),
               ('dropHiddenLayer3', 'value'),
               ('dropHiddenLayer4', 'value'),
               ('dropHiddenLayer5', 'value'),
               ('dropEngineRpm', 'value'),
               ('dropInjectionQty', 'value'),
               ('dropCoolantTemp', 'value'),
               ('slideTrainTestSplit', 'value'),
               ('dropNumIter', 'value')]
app_output = [Output(comp, prop) for comp, prop in upload_list]
@app.callback(app_output,
              [Input('openDataButton', 'filename'),
               Input('dataTableFeature', 'data')]) #this is to give sequence to update plot callback
def update_app_state(filename, dummy):
    if filename != None:
        return load_app_state(upload_list, filename)
    else:
        raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True)
