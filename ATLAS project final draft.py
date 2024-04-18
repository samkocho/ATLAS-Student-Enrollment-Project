import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.formula.api as smf
import random
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from statsmodels.tsa.deterministic import DeterministicProcess
import gradio as gr
import io
import matplotlib.ticker as ticker
import plotly.graph_objects as go


# Read and organize data

# Undergrad
data = pd.read_excel("studentsbycollege.xlsx",sheet_name="data").fillna(0)
campuswide = pd.read_excel("studentsbycollege.xlsx",sheet_name="campuswide").fillna(0)
agr = pd.read_excel("studentsbycollege.xlsx",sheet_name="agr").fillna(0)
ahs = pd.read_excel("studentsbycollege.xlsx",sheet_name="ahs").fillna(0)
centerinnov = pd.read_excel("studentsbycollege.xlsx",sheet_name="centerinnov").fillna(0)
media = pd.read_excel("studentsbycollege.xlsx",sheet_name="media").fillna(0)
genstudies = pd.read_excel("studentsbycollege.xlsx",sheet_name="genstudies").fillna(0)
education = pd.read_excel("studentsbycollege.xlsx",sheet_name="education").fillna(0)
finenapplied = pd.read_excel("studentsbycollege.xlsx",sheet_name="finenapplied").fillna(0)
gies = pd.read_excel("studentsbycollege.xlsx",sheet_name="gies").fillna(0)
grainger = pd.read_excel("studentsbycollege.xlsx",sheet_name="grainger").fillna(0)
aviation = pd.read_excel("studentsbycollege.xlsx",sheet_name="aviation").fillna(0)
las = pd.read_excel("studentsbycollege.xlsx",sheet_name="las").fillna(0)
provost = pd.read_excel("studentsbycollege.xlsx",sheet_name="provost").fillna(0)
ischool = pd.read_excel("studentsbycollege.xlsx",sheet_name="ischool").fillna(0)
socialwork = pd.read_excel("studentsbycollege.xlsx",sheet_name="socialwork").fillna(0)

# Grad
data2 = pd.read_excel("gradstudents.xlsx",sheet_name="data2").fillna(0)
campuswide2 = pd.read_excel("gradstudents.xlsx",sheet_name="campuswide2").fillna(0)
agr2 = pd.read_excel("gradstudents.xlsx",sheet_name="agr2").fillna(0)
ahs2 = pd.read_excel("gradstudents.xlsx",sheet_name="ahs2").fillna(0)
centerinnov2 = pd.read_excel("gradstudents.xlsx",sheet_name="centerinnov2").fillna(0)
media2 = pd.read_excel("gradstudents.xlsx",sheet_name="media2").fillna(0)
education2= pd.read_excel("gradstudents.xlsx",sheet_name="education2").fillna(0)
finenapplied2 = pd.read_excel("gradstudents.xlsx",sheet_name="finenapplied2").fillna(0)
gies2 = pd.read_excel("gradstudents.xlsx",sheet_name="gies2").fillna(0)
grad = pd.read_excel("gradstudents.xlsx",sheet_name="grad").fillna(0)
grainger2 = pd.read_excel("gradstudents.xlsx",sheet_name="grainger2").fillna(0)
aviation2 = pd.read_excel("gradstudents.xlsx",sheet_name="aviation2").fillna(0)
law = pd.read_excel("gradstudents.xlsx",sheet_name="law").fillna(0)
las2 = pd.read_excel("gradstudents.xlsx",sheet_name="las2").fillna(0)
ischool2 = pd.read_excel("gradstudents.xlsx",sheet_name="ischool2").fillna(0)
labor = pd.read_excel("gradstudents.xlsx",sheet_name="labor").fillna(0)
socialwork2 = pd.read_excel("gradstudents.xlsx",sheet_name="socialwork2").fillna(0)
vet = pd.read_excel("gradstudents.xlsx",sheet_name="vet").fillna(0)


data.index = data["Fall Term"]
college_dfs = [agr,ahs,centerinnov,media,genstudies,education,finenapplied,gies,grainger,aviation,las,provost,ischool,socialwork]
for df in college_dfs:
    df.index = df["Fall Term"]
college_names = ['Agr, Consumer, & Environmental Sciences', 'Applied Health Sciences', 'Center Innov in Tech Learn','College of Media','Division of General Studies',
                'Education','Fine & Applied Arts', 'Gies College of Business', 'Grainger Engineering', 'Institute of Aviation', 'Liberal Arts & Sciences',
                'Provost & VC Acad Affairs', 'School of Information Sciences','School of Social Work']
for college_df, college_name in zip(college_dfs, college_names):
    international_students = college_df['International']
    data[college_name + " International"] = international_students
data = data.fillna(0)

# Adding department feature to data
department_names = [
    ['Ag Ldrshp Educ Comm Program', 'Agr & Consumer Economics', 'Agr, Consumer, & Env Sci Admn',
     'Agr, Consumer, & Evn Sciences', 'Agricultural & Biological Engr', 'Agricultural Comm Pgm & Crse',
     'Animal Sciences', 'Crop Sciences', 'Food Science & Human Nutrition', 'Human Dvlpmt & Family Studies',
     'Natural Res & Env Sci'],
    ['Applied Health Sci Courses', 'Applied Health Sciences Admin', 'i-Health Program',
     'Kinesiology & Community Health', 'Recreation, Sport and Tourism', 'Speech & Hearing Science'],
    ['Center Innov in Teach Learn'],
    ['Advertising', 'College of Media Admin', 'College of Media Programs', 'Inst of Communications Rsch',
     'Journalism', 'Media and Cinema Studies'],
    ['Div General Studies Admin'],
    ['Curriculum and Instruction', 'Education Administration', 'Special Education'],
    ['Architecture', 'Art & Design', 'Dance', 'Fine & Applied Arts Admin', 'Fine & Applied Arts Courses',
     'Landscape Architecture', 'Music', 'Theatre', 'Urban & Regional Planning'],
    ['Accountancy', 'Business Administration', 'College of Business', 'Finance'],
    ['Aerospace Engineering', 'Bioengineering', 'Civil & Environmental Eng', 'Computer Science',
     'Electrical & Computer Eng', 'Engineering Administration', 'Engineering Courses', 'Industrial&Enterprise Sys Eng',
     'Materials Science & Engineering', 'Mechanical Sci & Engineering', 'Nuclear, Plasma, & Rad Engr', 'Physics'],
    ['Institute of Aviation'],
    ['African American Studies', 'Anthropology', 'Asian American Studies', 'Astronomy', 'Atmospheric Sciences',
     'Biochemistry', 'Chemical & Biomolecular Engr', 'Chemistry', 'Classics', 'Communication',
     'Comparative & World Literature', 'E. Asian Languages & Cultures', 'Earth Sci & Environmental Chng', 'Economics',
     'English', 'French and Italian', 'Gender and Women\'s Studies', 'Geography & GIS', 'Germanic Languages & Lit',
     'History', 'LAS Administration', 'Latin American & Carib Studies', 'Latina/Latino Studies', 'Life Sciences',
     'Linguistics', 'Mathematics', 'Philosophy', 'Political Science', 'Psychology', 'Religion',
     'Russian,E European,Eurasn Ctr', 'Sch Earth, Soc, Environ Admin', 'School of Integrative Biology',
     'School of Molecular & Cell Bio', 'Slavic Languages & Literature', 'Sociology', 'Spanish and Portuguese',
     'Statistics'],
    ['Undergraduate Admissions'],
    ['Information Sciences'],
    ['School of Social Work']
]

college_department_dict = dict(zip(college_names, department_names))

for college_df, college_name in zip(college_dfs, college_names):
    # international_students = college_df['International']    
    # Get the list of department names for the current college
    departments = college_department_dict[college_name]
    
    # Iterate over department names and add corresponding columns to the data dataframe
    for department_name in departments:
        # Check if the department name exists in the dataframe
        if department_name in college_df.columns:
            department_international_students = college_df[department_name]
            data[department_name + " International"] = department_international_students
        else:
            print(f"Department '{department_name}' does not exist in the dataframe for college '{college_name}'.")
data = data.fillna(0)

#Reset index for functions
data.reset_index(drop=True, inplace=True)

############################# GRAD

data2.index = data2["Fall Term"]
college_dfs_grad = [agr2,ahs2,centerinnov2,media2,education2,finenapplied2,gies2,grad,grainger2,aviation2,law,las2,ischool2,labor,socialwork2,vet]
for df in college_dfs_grad:
    df.index = df["Fall Term"]

college_names_grad = ['Agr, Consumer, & Environmental Sciences', 'Applied Health Sciences', 'Center Innov in Tech Learn','College of Media',
                'Education','Fine & Applied Arts', 'Gies College of Business', 'Gradute College','Grainger Engineering', 'Institute of Aviation',
                  'Law','Liberal Arts & Sciences',
                 'School of Information Sciences','School of Labor & Empl. Rel.','School of Social Work','Veterinary Medicine']
for college_df, college_name in zip(college_dfs_grad, college_names_grad):
    international_students = college_df['International']
    data2[college_name + " International"] = international_students
data2 = data2.fillna(0)

department_names_grad = [
    ['Agr & Consumer Economics','Agr, Consumer, & Evn Sciences', 'Agricultural & Biological Engr', 'Animal Sciences',
     'Crop Sciences', 'Food Science & Human Nutrition', 'Human Dvlpmt & Family Studies', 'Natural Res & Env Sci', 'Nutritional Sciences'],
    ['Applied Health Sci Courses', 'Kinesiology & Community Health', 'Recreation, Sport and Tourism',
     'Speech & Hearing Sciences'],
    ['Center Innov in Teach Learn'],
    ['Advertising', 'College of Media Admin', 'Inst of Communications Rsch','Journalism'],
    ['Curriculum and Instruction', 'Educ Policy, Orgzn & Leadership','Educational Psychology','Special Education'],
    ['Architecture', 'Art & Design', 'Dance', 'Landscape Architecture', 'Music', 'Theatre', 'Urban & Regional Planning'],
    ['Accountancy', 'Business Administration', 'College of Business', 'Executive MBA Program','Finance','MBA Program Administration'],
    ['Graduate Admin','Graduate College Programs'],
    ['Aerospace Engineering', 'Bioengineering', 'Civil & Environmental Eng', 'Computer Science',
     'Electrical & Computer Eng', 'Engineering Administration', 'Engineering Courses', 'Industrial&Enterprise Sys Eng',
     'Materials Science & Engineering', 'Mechanical Sci & Engineering', 'Nuclear, Plasma, & Rad Engr', 'Physics'],
    ['Institute of Aviation'],
    ['Law'],
    ['Anthropology', 'Astronomy', 'Atmospheric Sciences', 'Biochemistry', 'Cell & Developmental Biology',
     'Center for African Studies','Chemical & Biomolecular Engr', 'Chemistry', 'Classics', 'Communication',
     'Comparative & World Literature', 'Ctr S. Asian & MidEast Studies', 'E. Asian Languages & Cultures', 
     'Earth Sci & Environmental Chng', 'Economics', 'English', 'Entomology', 'Evolution Ecology Behavior',
     'French and Italian', 'Geography & GIS', 'Germanic Languages & Lit',
     'History', 'Latin American & Carib Studies', 'Liberal Arts & Sciences',
     'Linguistics', 'Mathematics', 'Microbiology', 'Molecular & Integrative Physl',
     'Neuroscience Program', 'Philosophy', 'Plant Biology', 'Political Science', 'Psychology', 'Religion',
     'Russian,E European,Eurasn Ctr', 'Sch Lit, Cultures, Ling Adm', 'School of Integrative Biology',
     'School of Molecular & Cell Bio', 'Slavic Languages & Literature', 'Sociology', 'Spanish and Portuguese',
     'Statistics','Translation & Interpreting St'],
    ['Information Sciences','Illinois Informatics Institute'],
    ['School of Labor & Empl. Rel.'],
    ['School of Social Work'],
    ['Vet Medicine Administration','Vet Clinical Medicine','Pathobiology',
     'Comparative Biosciences']
]

college_department_dict_grad = dict(zip(college_names_grad, department_names_grad))

for college_df, college_name in zip(college_dfs_grad, college_names_grad):
    # international_students = college_df['International']
    # # Get the list of department names for the current college
    departments = college_department_dict_grad[college_name]
    
    # Iterate over department names and add corresponding columns to the data dataframe
    for department_name in departments:
        # Check if the department name exists in the dataframe
        if department_name in college_df.columns:
            department_international_students = college_df[department_name]
            data2[department_name + " International"] = department_international_students
        else:
            print(f"Department '{department_name}' does not exist in the dataframe for college '{college_name}'.")
data2 = data2.fillna(0)

#Reset index for functions
data2.reset_index(drop=True, inplace=True)


#demo 1
def forecast_international_students1(data, college_name, forecast_steps):
    y = data[college_name + ' International']
    dp = DeterministicProcess(
    index=data["Fall Term"],
    constant=True,               
    order=1,                     
    drop=True,                   
    )
    X_insample = dp.in_sample()
    model = LinearRegression(fit_intercept=False)
    model.fit(X_insample, y)

    # Forecast future international students
    X_fore = dp.out_of_sample(steps=forecast_steps)
    y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

    last_year = 2022
    years_fore = pd.Index(range(last_year + 1, last_year + forecast_steps + 1))

    fig = go.Figure()


    historical_trace = go.Scatter(x=data["Fall Term"], y=data[college_name + ' International'], mode='lines+markers', name='Historical Data')
    historical_hover_text = [f'Year: {year}<br>International Students: {value}' for year, value in zip(data["Fall Term"], data[college_name + ' International'])]
    historical_trace.hoverinfo = 'text'
    historical_trace.hovertext = historical_hover_text
    fig.add_trace(historical_trace)

    # Add forecasted data with hover text
    forecast_trace = go.Scatter(x=years_fore, y=y_fore, mode='lines+markers', name='Forecasted Data')
    forecast_values_rounded = [round(value) for value in y_fore]  # Round forecasted values to integers
    forecast_hover_text = [f'Year: {year}<br>Forecasted Undergraduate International Students: {value}' for year, value in zip(years_fore, forecast_values_rounded)]
    # forecast_hover_text = [f'Year: {year}<br>Forecasted International Students: {value}' for year, value in zip(years_fore, y_fore)]
    forecast_trace.hoverinfo = 'text'
    forecast_trace.hovertext = forecast_hover_text
    fig.add_trace(forecast_trace)


    fig.update_layout(
        title='Forecast of Undergraduate International Students in {}'.format(college_name),
        xaxis_title='Year',
        yaxis_title='International Students',
        legend=dict(x=0, y=1, traceorder="normal"),
        hovermode='closest',
        template='plotly_white'
    )
    return fig
def forecast_interface(college_name, forecast_steps):
    plot_figure = forecast_international_students1(data, college_name, forecast_steps)
    return plot_figure
def description():
    return """
    <h2>Forecasting Undergraduate International Students (Colleges)</h2>
    <p>This interface allows you to forecast the number of Undergraduate International Students for a specific college using a linear regression model. 
    <br> Select the <b>college name</b> and the number of <b>forecast steps</b> to generate the forecasted data plotted over historical trends.
    </p>
"""
with gr.Blocks() as demo1:
    gr.HTML(description())
    college_name_input = gr.Dropdown(label="College Name", choices=list(college_department_dict.keys()))
    forecast_steps_input = gr.Slider(minimum=1, maximum=30, step=1, label="Forecast Steps",info="Choose how many years ahead to predict.")
    
    graph_btn = gr.Button("Forecast")
    graph_output = gr.Plot()
    graph_btn.click(fn=forecast_interface, inputs=[college_name_input,forecast_steps_input], outputs=graph_output)


# demo 2
def forecast_international_students2(data, college_name, department_name, forecast_steps):
    y = data[department_name + ' International']
    dp = DeterministicProcess(
        index=data["Fall Term"],
        constant=True,               
        order=1,                     
        drop=True,                   
    )
    X_insample = dp.in_sample()
    model = LinearRegression(fit_intercept=False)
    model.fit(X_insample, y)

    # Forecast future international students
    X_fore = dp.out_of_sample(steps=forecast_steps)
    y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

    last_year = 2022
    years_fore = pd.Index(range(last_year + 1, last_year + forecast_steps + 1))

    fig = go.Figure()

    # Add historical data with hover text
    historical_trace = go.Scatter(x=data["Fall Term"], y=data[department_name + ' International'], mode='lines+markers', name='Historical Data')
    historical_hover_text = [f'Year: {year}<br>International Students: {value}' for year, value in zip(data["Fall Term"], data[department_name + ' International'])]
    historical_trace.hoverinfo = 'text'
    historical_trace.hovertext = historical_hover_text
    fig.add_trace(historical_trace)

    # Add forecasted data with hover text
    forecast_trace = go.Scatter(x=years_fore, y=y_fore, mode='lines+markers', name='Forecasted Data')
    forecast_values_rounded = [round(value) for value in y_fore]  # Round forecasted values to integers
    forecast_hover_text = [f'Year: {year}<br>Forecasted Undergraduate International Students: {value}' for year, value in zip(years_fore, forecast_values_rounded)]
    forecast_trace.hoverinfo = 'text'
    forecast_trace.hovertext = forecast_hover_text
    fig.add_trace(forecast_trace)
    
    fig.update_layout(
        title=f'Forecast of Undergraduate International Students in {department_name}<br>at {college_name}<br>',
        xaxis_title='Year',
        yaxis_title='International Students',
        legend=dict(x=0, y=1, traceorder="normal"),
        hovermode='closest',
        template='plotly_white'
    )
    return fig
def update_department_choices(college_name):
        choices = college_department_dict[college_name]
        return gr.Dropdown(choices = choices,interactive = True)
def forecast_interface(college_name,department_name, forecast_steps):
        plot_figure = forecast_international_students2(data, college_name, department_name, forecast_steps)
        return plot_figure
def description():
    return """
    <h2>Forecasting Undergraduate International Students (Colleges and Departments)</h2>
    <p>This interface allows you to forecast the number of Undergraduate International Students for a specific department within a college using a linear regression model. 
    <br> Select the <b>college name</b>, <b>department name</b> and the number of <b>forecast steps</b> to generate the forecasted data plotted over historical trends.
    </p>
"""
with gr.Blocks() as demo2:
    gr.HTML(description())
    with gr.Row():
        college_name_input = gr.Dropdown(label="College Name", choices=list(college_department_dict.keys()))
        department_input = gr.Dropdown(label="Department", choices=[])
 
    forecast_steps_input = gr.Slider(minimum=1, maximum=30, step=1, label="Forecast Steps", info="Choose how many years ahead to predict.", interactive = True)
    college_name_input.change(update_department_choices, inputs= [college_name_input], outputs = [department_input])
    
    graph_btn = gr.Button("Forecast")
    graph_output = gr.Plot()
    graph_btn.click(fn=forecast_interface, inputs=[college_name_input,department_input,forecast_steps_input], outputs=graph_output)
    
# intro 1
def plot_college_distribution(college, plot_type):
    df = None
    # Map college to respective DataFrame
    college_df_mapping = {
        "Agr, Consumer, & Environmental Sciences": agr,
        "Applied Health Sciences": ahs,
        "Center Innov in Tech Learn": centerinnov,
        "College of Media": media,
        "Division of General Studies": genstudies,
        "Education": education,
        "Fine & Applied Arts": finenapplied,
        "Gies College of Business": gies,
        "Grainger Engineering": grainger,
        "Institute of Aviation": aviation,
        "Liberal Arts & Sciences": las,
        "Provost & VC Acad Affairs": provost,
        "School of Information Sciences": ischool,
        "School of Social Work": socialwork
    }

    if college in college_df_mapping:
        df = college_df_mapping[college]
    else:
        raise ValueError("Invalid college selection")

    fig = None
    if plot_type == "Area Plot":
        fig = go.Figure()
        for column in df.columns[1:9]:
            fig.add_trace(go.Scatter(x=df['Fall Term'], y=df[column], fill='tozeroy', mode='none', name=column))
    elif plot_type == "Bar Chart":
        fig = go.Figure()
        for column in df.columns[1:9]:
            fig.add_trace(go.Bar(x=df['Fall Term'], y=df[column], name=column))
    elif plot_type == "Pie Chart":
        fig = go.Figure()
        fig.add_trace(go.Pie(labels=df.columns[1:9], values=df.iloc[0, 1:9]))

    if fig:
        fig.update_layout(
            title={
                'text': f'Distribution of Undergraduate Students in {college}',
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Fall Term',
            yaxis_title='Number of Students',
            legend_title='Race/Ethnicity',
            hovermode='x',
            annotations=[
                dict(
                    text="Double click on a Race/Ethnicity to isolate the view.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.6,
                    y=1.1
                )
            ]
        )
        return fig
    else:
        raise ValueError("Invalid plot type")
def plot_campus():
    fig = go.Figure()
    for column in campuswide.columns[1:9]:
        fig.add_trace(go.Scatter(x=campuswide['Fall Term'], y=campuswide[column], fill='tozeroy', mode='none', name=column))
    fig.update_layout(
        title={
        'text': 'Distribution of Undergraduate Students in the Entire Campus',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        xaxis_title='Fall Term',
        yaxis_title='Number of Students',
        legend_title='Race/Ethnicity',
        hovermode='x',
    )
    return fig
def description():
    return """
    <h2>Undergraduate Student Enrollment at UIUC (Race/Ethnicity)</h2>
    <p>In this tab, we will be looking at undergraduate <b>student body distributions.</b></p>
"""
def description2():
    return """
    <p>If you're interested in looking at trends in the different colleges use the interface below! </b></p>

"""
with gr.Blocks() as intro1:
    gr.HTML(description())
    gr.Plot(plot_campus())
    gr.HTML(description2())
    
    with gr.Row():
        college_dropdown = gr.Dropdown(
            [
                'Agr, Consumer, & Environmental Sciences',
                'Applied Health Sciences',
                'Center Innov in Tech Learn',
                'College of Media',
                'Division of General Studies',
                'Education',
                'Fine & Applied Arts',
                'Gies College of Business',
                'Grainger Engineering',
                'Institute of Aviation',
                'Liberal Arts & Sciences',
                'Provost & VC Acad Affairs',
                'School of Information Sciences',
                'School of Social Work'
            ],
            label="Choose a college to visualize"
        )
        
        plot_type_dropdown = gr.Dropdown(
            ["Area Plot", "Bar Chart", "Pie Chart"],
            label="Choose a plot type"
        )
        
    graph_btn = gr.Button("Visualize")
    graph_output = gr.Plot()
    graph_btn.click(fn=plot_college_distribution, inputs=[college_dropdown, plot_type_dropdown], outputs=graph_output)

# intro 2
def plot_college_distribution2(college):
    df = None
    if college == "Agr, Consumer, & Environmental Sciences":
        df = agr
    elif college == "Applied Health Sciences":
        df = ahs
    elif college == "Center Innov in Tech Learn":
        df = centerinnov
    elif college == "College of Media":
        df = media
    elif college == "Division of General Studies":
        df = genstudies
    elif college == "Education":
        df = education
    elif college == "Fine & Applied Arts":
        df = finenapplied
    elif college == "Gies College of Business":
        df = gies
    elif college == "Grainger Engineering":
        df = grainger
    elif college == "Institute of Aviation":
        df = aviation
    elif college == "Liberal Arts & Sciences":
        df = las
    elif college == "Provost & VC Acad Affairs":
        df = provost
    elif college == "School of Information Sciences":
        df = ischool
    elif college == "School of Social Work":
        df = socialwork
    
    if df is None:
        raise ValueError("Invalid college selection")
    
    fig = go.Figure()
    # for column in df.columns[1:9]: # Selecting only the specified columns
    fig.add_trace(go.Scatter(x=df['Fall Term'], y=df["Grand Total"], mode='lines+markers'))

    # Update layout
    fig.update_layout(
        title={
        'text': f'Trend of Undergraduate Students in {college}',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        xaxis_title='Fall Term',
        yaxis_title='Number of Students',
        legend_title='Race/Ethnicity',
        hovermode='x'
    )

    return fig
def plot_campus():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=campuswide['Fall Term'], y=campuswide["Grand Total"],  mode='lines+markers'))
    fig.update_layout(
        title={
        'text': 'Trend of Undergraduate Students in the Entire Campus',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        xaxis_title='Fall Term',
        yaxis_title='Number of Students',
        # legend_title='Race/Ethnicity',
        hovermode='x',
    )
    return fig
def description():
    return """
    <h2>Undergraduate Student Enrollment at UIUC (Historical Trend)</h2>
    <p>In this tab, we will be looking at undergraduate <b>student body population trend over time.</b></p>
"""
def description2():
    return """
    <p>If you're interested in looking at trends in the different colleges use the interface below! </b></p>
"""
with gr.Blocks() as intro2:
    gr.HTML(description())

    with gr.Column(): 
        gr.Plot(plot_campus())

    gr.HTML(description2())
    with gr.Row():
        college_dropdown = gr.Dropdown([
    'Agr, Consumer, & Environmental Sciences',
    'Applied Health Sciences',
    'Center Innov in Tech Learn',
    'College of Media',
    'Division of General Studies',
    'Education',
    'Fine & Applied Arts',
    'Gies College of Business',
    'Grainger Engineering',
    'Institute of Aviation',
    'Liberal Arts & Sciences',
    'Provost & VC Acad Affairs',
    'School of Information Sciences',
    'School of Social Work'], 
    label="Choose a college to visualize")
    
    graph_btn = gr.Button("Visualize")
    graph_output = gr.Plot()
    graph_btn.click(fn=plot_college_distribution2, inputs=[college_dropdown], outputs=graph_output)

# Author's notes
def get_description():
    return """
    <h2>Forecasting Trends of International Students at UIUC</h2>

    <h3>Project Overview</h3>
    <p>This project aims to forecast the trends of international 
    students at the University of Illinois at Urbana-Champaign (UIUC) 
    using a deterministic linear regression model.</p>
    <p>Deterministic machine learning algorithms, such as linear regression and decision trees, aim to find a fixed relationship between inputs and outputs. </p>

    <h3>Why Linear Regression?</h3>
    <p>Linear regression is chosen for its suitability in modeling 
    relationships between features and the target variable. It assumes 
    a linear relationship, provides interpretable coefficients, is 
    simple to implement, computationally efficient, and serves as a 
    baseline model for comparison. Linear regression is particularly 
    useful when historical data exhibits linear trends and when the time 
    series data is approximately stationary.</p>

    <h3>Deterministic Process</h3>
    <p>Deterministic means the opposite of randomness, giving the same results every time. </p>
    <p>The deterministic process aims to model the relationship between 
    the time index and the variable being studied. It provides a structured 
    framework for understanding and modeling the behavior of a variable over 
    time in a deterministic, predictable manner. This process forms the basis for
      forecasting and making informed decisions based on historical data and underlying relationships.</p>

    

    <div style="display: flex; justify-content: space-between;">
        <div>
            <h4>Linear Regression</h4>
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/1200px-Linear_regression.svg.png" alt="Linear Regression" width="400"/>
        </div>
        <div>
            <h4>Deterministic Process</h4>
            <img src="https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcT5ThFuj9_0JdCUprmSn32cE6UD7rtw99JGHc9dh5SKYGjodhNI" alt="Deterministic Process" width="400"/>
        </div>
    </div>
    
    <h4>University of Illinois at Urbana Champaign Division of Management Information </h4>
    <p>Sources: https://www.dmi.illinois.edu/</p>

    """
with gr.Blocks() as p:
    output = gr.HTML(get_description)


# grad intro 1
def plot_college_distribution(college, plot_type):
    df = None
    # Map college to respective DataFrame
    college_df_mapping = {
        "Agr, Consumer, & Environmental Sciences": agr2,
        "Applied Health Sciences": ahs2,
        "Center Innov in Tech Learn": centerinnov2,
        "College of Media": media2,
        "Education": education2,
        "Fine & Applied Arts": finenapplied2,
        "Gies College of Business": gies2,
        "Graduate School": grad,
        "Grainger Engineering": grainger2,
        "Institute of Aviation": aviation2,
        "Law": law,
        "Liberal Arts & Sciences": las2,
        "School of Information Sciences": ischool2,
        "School of Labor & Empl. Rel.": labor,
        "School of Social Work": socialwork2,
        "Veterinary Medicine": vet
    }

    if college in college_df_mapping:
        df = college_df_mapping[college]
    else:
        raise ValueError("Invalid college selection")

    fig = None
    if plot_type == "Area Plot":
        fig = go.Figure()
        for column in df.columns[1:9]:
            fig.add_trace(go.Scatter(x=df['Fall Term'], y=df[column], fill='tozeroy', mode='none', name=column))
    elif plot_type == "Bar Chart":
        fig = go.Figure()
        for column in df.columns[1:9]:
            fig.add_trace(go.Bar(x=df['Fall Term'], y=df[column], name=column))
    elif plot_type == "Pie Chart":
        fig = go.Figure()
        fig.add_trace(go.Pie(labels=df.columns[1:9], values=df.iloc[0, 1:9]))

    if fig:
        fig.update_layout(
            title={
                'text': f'Distribution of Graduate Students in {college}',
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Fall Term',
            yaxis_title='Number of Students',
            legend_title='Race/Ethnicity',
            hovermode='x',
            annotations=[
                dict(
                    text="Double click on a Race/Ethnicity to isolate the view.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.6,
                    y=1.1
                )
            ]
        )
        return fig
    else:
        raise ValueError("Invalid plot type")
def plot_campus():
    fig = go.Figure()
    for column in campuswide2.columns[1:9]:
        fig.add_trace(go.Scatter(x=campuswide2['Fall Term'], y=campuswide2[column], fill='tozeroy', mode='none', name=column))
    fig.update_layout(
        title={
        'text': 'Distribution of Graduate Students in the Entire Campus',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        xaxis_title='Fall Term',
        yaxis_title='Number of Students',
        legend_title='Race/Ethnicity',
        hovermode='x',
    )
    return fig
def description():
    return """
    <h2>Graduate Student Enrollment at UIUC (Race/Ethnicity)</h2>
    <p>In this tab, we will be looking at Graduate <b>student body distributions.</b></p>
"""
def description2():
    return """
    <p>If you're interested in looking at trends in the different colleges use the interface below! </b></p>

"""
with gr.Blocks() as gradintro1:
    gr.HTML(description())
    gr.Plot(plot_campus())
    gr.HTML(description2())
    
    with gr.Row():
        college_dropdown = gr.Dropdown(
            college_names_grad,
            label="Choose a college to visualize"
        )
        
        plot_type_dropdown = gr.Dropdown(
            ["Area Plot", "Bar Chart", "Pie Chart"],
            label="Choose a plot type"
        )
        
    graph_btn = gr.Button("Visualize")
    graph_output = gr.Plot()
    graph_btn.click(fn=plot_college_distribution, inputs=[college_dropdown, plot_type_dropdown], outputs=graph_output)

# grad intro 2
def plot_college_distribution2(college):
    df = None
    if college == "Agr, Consumer, & Environmental Sciences":
        df = agr2
    elif college == "Applied Health Sciences":
        df = ahs2
    elif college == "Center Innov in Tech Learn":
        df = centerinnov2
    elif college == "College of Media":
        df = media2
    elif college == "Education":
        df = education2
    elif college == "Fine & Applied Arts":
        df = finenapplied2
    elif college == "Gies College of Business":
        df = gies2
    elif college == "Graduate School":
        df = grad
    elif college == "Grainger Engineering":
        df = grainger2
    elif college == "Institute of Aviation":
        df = aviation2
    elif college == "Law":
        df = law
    elif college == "Liberal Arts & Sciences":
        df = las2
    elif college == "School of Information Sciences":
        df = ischool2
    elif college == "School of Labor & Empl. Rel.":
        df = labor
    elif college == "School of Social Work":
        df = socialwork2
    elif college == "Veterinary Medicine":
        df = vet
    
    if df is None:
        raise ValueError("Invalid college selection")
    
    fig = go.Figure()
    # for column in df.columns[1:9]: # Selecting only the specified columns
    fig.add_trace(go.Scatter(x=df['Fall Term'], y=df["Grand Total"], mode='lines+markers'))

    # Update layout
    fig.update_layout(
        title={
        'text': f'Trend of Graduate Students in {college}',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        xaxis_title='Fall Term',
        yaxis_title='Number of Students',
        legend_title='Race/Ethnicity',
        hovermode='x'
    )

    return fig
def plot_campus():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=campuswide2['Fall Term'], y=campuswide2["Grand Total"],  mode='lines+markers'))
    fig.update_layout(
        title={
        'text': 'Trend of Graduate Students in the Entire Campus',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        xaxis_title='Fall Term',
        yaxis_title='Number of Students',
        # legend_title='Race/Ethnicity',
        hovermode='x',
    )
    return fig
def description():
    return """
    <h2>Gradiate Student Enrollment at UIUC (Historical Trend)</h2>
    <p>In this tab, we will be looking at undergraduate <b>student body population trend over time.</b></p>
"""
def description2():
    return """
    <p>If you're interested in looking at trends in the different colleges use the interface below! </b></p>
"""
with gr.Blocks() as gradintro2:
    gr.HTML(description())

    with gr.Column(): 
        gr.Plot(plot_campus())

    gr.HTML(description2())
    with gr.Row():
        college_dropdown = gr.Dropdown(college_names_grad, 
    label="Choose a college to visualize")
    
    graph_btn = gr.Button("Visualize")
    graph_output = gr.Plot()
    graph_btn.click(fn=plot_college_distribution2, inputs=[college_dropdown], outputs=graph_output)

# grad model 1
def forecast_international_students3(data, college_name, forecast_steps):
    y = data[college_name + ' International']
    dp = DeterministicProcess(
    index=data["Fall Term"],
    constant=True,               
    order=1,                     
    drop=True,                   
    )
    X_insample = dp.in_sample()
    model = LinearRegression(fit_intercept=False)
    model.fit(X_insample, y)

    # Forecast future international students
    X_fore = dp.out_of_sample(steps=forecast_steps)
    y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

    last_year = 2022
    years_fore = pd.Index(range(last_year + 1, last_year + forecast_steps + 1))
    
    fig = go.Figure()


    historical_trace = go.Scatter(x=data["Fall Term"], y=data[college_name + ' International'], mode='lines+markers', name='Historical Data')
    historical_hover_text = [f'Year: {year}<br>International Students: {value}' for year, value in zip(data["Fall Term"], data[college_name + ' International'])]
    historical_trace.hoverinfo = 'text'
    historical_trace.hovertext = historical_hover_text
    fig.add_trace(historical_trace)

    # Add forecasted data with hover text
    forecast_trace = go.Scatter(x=years_fore, y=y_fore, mode='lines+markers', name='Forecasted Data')
    forecast_values_rounded = [round(value) for value in y_fore]  # Round forecasted values to integers
    forecast_hover_text = [f'Year: {year}<br>Forecasted Graduate International Students: {value}' for year, value in zip(years_fore, forecast_values_rounded)]
    forecast_trace.hoverinfo = 'text'
    forecast_trace.hovertext = forecast_hover_text
    fig.add_trace(forecast_trace)

    fig.update_layout(
        title='Forecast of Graduate International Students in {}'.format(college_name),
        xaxis_title='Year',
        yaxis_title='International Students',
        legend=dict(x=0, y=1, traceorder="normal"),
        hovermode='closest',
        template='plotly_white'
    )
    return fig
def forecast_interface(college_name, forecast_steps):
    plot_figure = forecast_international_students3(data2, college_name, forecast_steps)
    return plot_figure
def description():
    return """
    <h2>Forecasting Graduate International Students (Colleges)</h2>
    <p>This interface allows you to forecast the number of Graduate International Students for a specific college using a linear regression model. 
    <br> Select the <b>college name</b> and the number of <b>forecast steps</b> to generate the forecasted data plotted over historical trends.
    </p>
"""
with gr.Blocks() as gradmodel1:
    gr.HTML(description())
    college_name_input = gr.Dropdown(label="College Name", choices=list(college_department_dict_grad.keys()))
    forecast_steps_input = gr.Slider(minimum=1, maximum=30, step=1, label="Forecast Steps",info="Choose how many years ahead to predict.")
    
    graph_btn = gr.Button("Forecast")
    graph_output = gr.Plot()
    graph_btn.click(fn=forecast_interface, inputs=[college_name_input,forecast_steps_input], outputs=graph_output)

# grad model 2
def forecast_international_students4(data, college_name, department_name, forecast_steps):
    y = data[department_name + ' International']
    dp = DeterministicProcess(
        index=data["Fall Term"],
        constant=True,               
        order=1,                     
        drop=True,                   
    )
    X_insample = dp.in_sample()
    model = LinearRegression(fit_intercept=False)
    model.fit(X_insample, y)

    # Forecast future international students
    X_fore = dp.out_of_sample(steps=forecast_steps)
    y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

    last_year = 2022
    years_fore = pd.Index(range(last_year + 1, last_year + forecast_steps + 1))

    fig = go.Figure()

    # Add historical data with hover text
    historical_trace = go.Scatter(x=data["Fall Term"], y=data[department_name + ' International'], mode='lines+markers', name='Historical Data')
    historical_hover_text = [f'Year: {year}<br>International Students: {value}' for year, value in zip(data["Fall Term"], data[department_name + ' International'])]
    historical_trace.hoverinfo = 'text'
    historical_trace.hovertext = historical_hover_text
    fig.add_trace(historical_trace)

    # Add forecasted data with hover text
    forecast_trace = go.Scatter(x=years_fore, y=y_fore, mode='lines+markers', name='Forecasted Data')
    forecast_values_rounded = [round(value) for value in y_fore]  # Round forecasted values to integers
    forecast_hover_text = [f'Year: {year}<br>Forecasted Graduate International Students: {value}' for year, value in zip(years_fore, forecast_values_rounded)]
    forecast_trace.hoverinfo = 'text'
    forecast_trace.hovertext = forecast_hover_text
    fig.add_trace(forecast_trace)

    fig.update_layout(
        title=f'Forecast of Graduate International Students in {department_name}<br>at {college_name}<br>',
        xaxis_title='Year',
        yaxis_title='International Students',
        legend=dict(x=0, y=1, traceorder="normal"),
        hovermode='closest',
        template='plotly_white'
    )
    return fig
def update_department_choices(college_name):
        choices = college_department_dict_grad[college_name]
        return gr.Dropdown(choices = choices,interactive = True)
def forecast_interface(college_name,department_name, forecast_steps):
        plot_figure = forecast_international_students4(data2, college_name, department_name, forecast_steps)
        return plot_figure
def description():
    return """
    <h2>Forecasting Graduate International Students (Colleges and Departments)</h2>
    <p>This interface allows you to forecast the number of Graduate International Students for a specific department within a college using a linear regression model. 
    <br> Select the <b>college name</b>, <b>department name</b> and the number of <b>forecast steps</b> to generate the forecasted data plotted over historical trends.
    </p>
"""
with gr.Blocks() as gradmodel2:
    gr.HTML(description())
    with gr.Row():
        college_name_input = gr.Dropdown(label="College Name", choices=list(college_department_dict_grad.keys()))
        department_input = gr.Dropdown(label="Department", choices=[])
 
    forecast_steps_input = gr.Slider(minimum=1, maximum=30, step=1, label="Forecast Steps", info="Choose how many years ahead to predict.", interactive = True)
    college_name_input.change(update_department_choices, inputs= [college_name_input], outputs = [department_input])
    
    graph_btn = gr.Button("Forecast")
    graph_output = gr.Plot()
    graph_btn.click(fn=forecast_interface, inputs=[college_name_input,department_input,forecast_steps_input], outputs=graph_output)
    

gr.TabbedInterface(
    [p, intro1, intro2, demo1, demo2, gradintro1, gradintro2, gradmodel1, gradmodel2], 
    ["Author's Notes","Undergrad Distribution: Race/Ethnicity","Undergrad Enrollment Trends", "Undergrad Model 1", "Undergrad Model 2", 
     "Grad Distribution: Race/Ethnicity", "Grad Enrollment Trends", "Grad Model 1", "Grad Model 2" ],
     title = "Sangjun Ko's Machine Learning Project: Forecasting International Students"
).launch()

