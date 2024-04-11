import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# Assuming 'analyze_and_plot_all' function is defined as before

# Function to load and prepare data, now capable of handling one or two files
def load_and_prepare_data(file1=None, file2=None):
    data_combined = pd.DataFrame()
    if file1:
        data1 = pd.read_excel(file1)
        data_combined = pd.concat([data_combined, data1])
    if file2:
        data2 = pd.read_excel(file2)
        data_combined = pd.concat([data_combined, data2])
        
    if not data_combined.empty:
        data_combined['Grade'] = data_combined['Jegyek'].str.split().str[0]
        data_combined['Numeric Grade'] = data_combined['Grade'].map({
            'Jeles': 5,  # Excellent
            'Jó': 4,     # Good
            'Közepes': 3,  # Satisfactory
            'Elégséges': 2,  # Pass
            'Elégtelen': 1,  # Fail
        })
        data_combined['Credit_Grade'] = data_combined['Kr.'] * data_combined['Numeric Grade']
    
    return data_combined

# Function to perform all analyses and generate plots
def analyze_and_plot_all(data_combined):
    # Preparing data for semester-wise and cumulative analyses
    data_combined_sorted = data_combined.sort_values(by='Félév')
    credit_grade_by_semester = data_combined_sorted.groupby('Félév').agg({
        'Credit_Grade': 'sum'
    }).reset_index()
    credit_grade_by_semester['CumulativeCreditGrade'] = credit_grade_by_semester['Credit_Grade'].cumsum()
    st.subheader('Cumulative Credit x Grade by Semester')
    st.markdown("""
    This plot shows how the product of credits and grades accumulates over semesters.
    A steadily increasing trend indicates consistent performance or improvement over time.
    """)
    # Plotting Cumulative Credit*Grade by Semester
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(credit_grade_by_semester['Félév'], credit_grade_by_semester['CumulativeCreditGrade'], marker='o', linestyle='-', color='blue', label='Cumulative Credit*Grade')
    plt.title('Cumulative Credit*Grade by Semester')
    plt.xlabel('Semester')
    plt.ylabel('Cumulative Credit*Grade')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)
    data_combined_sorted['Cumulative Credits'] = data_combined_sorted['Kr.'].cumsum()
    data_combined_sorted['Cumulative Average Grade'] = (data_combined_sorted['Numeric Grade'] * data_combined_sorted['Kr.']).cumsum() / data_combined_sorted['Cumulative Credits']

    # For plotting, we'll group by Semester to get the average grade per semester and the cumulative credits up to each semester
    grouped_data = data_combined_sorted.groupby('Félév').agg({
        'Numeric Grade': 'mean',
        'Cumulative Credits': 'last',
        'Cumulative Average Grade': 'last'
    }).reset_index()

    # Regression analysis
    st.subheader('Analysis of Grades and Credits by Semester')
    st.markdown("""
    The following plots provide insights into average grades, cumulative credits,
    and cumulative average grades across semesters, offering a glimpse into academic progress.
    """)
    grouped_data['Time Period'] = range(1, len(grouped_data) + 1)

    fig, ax1= plt.subplots(figsize=(10, 6))

    # Average grade by semester
    ax1.plot(grouped_data['Félév'], grouped_data['Numeric Grade'], marker='o', linestyle='-')
    ax1.set_title('Average Grade by Félév')
    ax1.set_xlabel('Semester')
    ax1.set_ylabel('Average Grade')
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    grouped_data['Season'] = grouped_data['Félév'].str[-1].apply(lambda x: 'Spring' if x == '1' else 'Fall')

    spring_avg_corrected = grouped_data[grouped_data['Season'] == 'Spring']['Numeric Grade'].mean()
    fall_avg_corrected = grouped_data[grouped_data['Season'] == 'Fall']['Numeric Grade'].mean()
    st.subheader('Analysis of Grades by Semesters Fall and Spring')
    st.markdown(f"""
    Your average grade during Spring is: {spring_avg_corrected}.
    While during Fall you managed to achieve: {fall_avg_corrected}
    """)

    # Cumulative credits by Félév
    fig, ax2= plt.subplots(figsize=(10, 6))
    ax2.plot(grouped_data['Félév'], grouped_data['Cumulative Credits'], marker='o', linestyle='-', color='r')
    ax2.set_title('Cumulative Credits by Félév')
    ax2.set_xlabel('Semester')
    ax2.set_ylabel('Cumulative Credits')
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
        
    grouped_data.rename(columns={'Cumulative Average Grade': 'CumulativeAverageGrade', 'Time Period': 'TimePeriod'}, inplace=True)

    # Fitting a linear regression model with the corrected column names
    model_corrected = ols('CumulativeAverageGrade ~ TimePeriod', data=grouped_data).fit()

    # Summary of the corrected model for p-values and other statistics
    model_summary_corrected = model_corrected.summary()

    # Extracting p-value for the slope (TimePeriod coefficient) to assess significance
    p_value_slope = model_corrected.pvalues['TimePeriod']

    # Summary of the model for p-values and other statistics
    # model_summary = model.summary()
    grouped_data['FittedValues'] = model_corrected.fittedvalues
    st.subheader('Regression Analysis on Cumulative Average Grade')
    st.markdown("""
    This regression analysis plot shows the trend in cumulative average grades over semesters,
    illustrating academic progression.
    """)
    # Re-plotting with corrected alignment
    fig, ax = plt.subplots(figsize=(10, 6))    
    ax.plot(grouped_data['Félév'], grouped_data['CumulativeAverageGrade'], marker='o', linestyle='-', label='Actual Data')
    # Ensuring we only plot as many fitted values as we have available
    ax.plot(grouped_data['Félév'], grouped_data['FittedValues'], linestyle='--', color='red', label='Regression Fit')
    plt.title('Cumulative Average Grade by Semester with Regression Fit')
    plt.xlabel('Semester')
    plt.ylabel('Cumulative Average Grade')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig)
    credit_grade_by_semester['TimePeriod'] = range(1, len(credit_grade_by_semester) + 1)
    model_cumulative = ols('CumulativeCreditGrade ~ TimePeriod', data=credit_grade_by_semester).fit()
    
    st.subheader('Understanding the Regression Analysis Summary')
    st.markdown("""
    The regression analysis summary:
    - **R-squared**: Represents the proportion of variance in the dependent variable that can be explained by the independent variables in the model.
    """)

    st.write(model_cumulative.summary())
    
    # Grouping by credits to calculate average grade based on credits
    avg_grade_by_credits = data_combined_sorted.groupby('Kr.').agg({
        'Numeric Grade': 'mean'
    }).reset_index()
    avg_grade_by_credits['Credits'] = avg_grade_by_credits['Kr.'].astype(str)

    st.subheader('Average Grade Based on Credits Analysis')
    st.markdown("""
    This visualization presents the relationship between the number of credits of a course and the average grade received. 
    Key observations to consider:
    - **Higher or Lower Credits**
    - **Variability**""")

    # Plotting the average grade based on credits
    fig, ax = plt.subplots(figsize=(10, 6))    
    ax.bar(avg_grade_by_credits['Credits'], avg_grade_by_credits['Numeric Grade'])
    plt.title('Average Grade Based on Credits')
    plt.xlabel('Credits')
    plt.ylabel('Average Grade')
    plt.xticks(avg_grade_by_credits['Credits'])
    st.pyplot(fig)

# Streamlit UI
st.title("Course Performance Analysis")

st.markdown("""
Upload two Excel files containing course data to analyze the relationship between semester progression, 
credit x grade product, and observe cumulative trends. Press the 'Analyze' button to start the analysis after uploading the files.
You can download it from Neptun by going to "Leckekönyv" and selecting all semesters and then clicking export to xlsx.
""")

file1 = st.file_uploader("Choose first Excel file (BSc)", type=['xlsx'])
file2 = st.file_uploader("Choose second Excel file (MSc)", type=['xlsx'])

# Button to trigger analysis or automatic execution upon startup with predefined data
if st.button('Analyze') or (not st.session_state.get('analyzed', False) and (file1 or file2)):
    data_combined = load_and_prepare_data(file1, file2)
    analyze_and_plot_all(data_combined)
    st.session_state['analyzed'] = True  # Mark as analyzed to prevent re-running automatically

# Optional: Automatically load and analyze predefined data on first run
if not st.session_state.get('initialized', False):
    # Load your predefined data here, e.g., from a default file path or URL
    data_combined = load_and_prepare_data('bsc.xlsx', 'msc.xlsx')
    analyze_and_plot_all(data_combined)
    st.session_state['initialized'] = True