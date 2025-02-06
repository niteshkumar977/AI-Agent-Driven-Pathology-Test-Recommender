# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import os
import mysql.connector
import sqlalchemy
import pymysql
import numpy as np
import calendar
import datetime
import random
import string
#from word_similarity import find_best_matches_from_dataframe
from word_similarity_updated_new import find_best_matches_from_dataframe

# # UPLOADING THE TEXTS (TEXT1,TEXT2....AND SO ON) FOR PATHOLOGY TEST NAME AND LABELS(TEST NAMES AS PER THE DIAGNOSTIC COMPANY)

# +
os.chdir(r'C:\Users\Admin\Test_Recommender_System\AI_Agent_type')

#df = pd.read_csv('input_data.csv', encoding='latin1')
df = pd.read_csv('horizontal_input_df.csv', encoding='latin1')
df.fillna('', inplace=True)
#df['text'].fillna('', inplace=True)
#df['label'].fillna('', inplace=True)
# -

# # CONVERT THE AI GENERATED RECOMMENDATIONS TO DATAFRAME OF TEST NAMES AND BLANK LABEL AND ALSO COMMA SEPRATED TEST NAMES

# +
def recommendation_converter(recommendations):
    test_names = []
    label_names = []

# Parse recommendations
    for rec in recommendations:
        if "Recommended tests: " in rec:
            rec = rec.replace("Recommended tests: ", "")
        test_names.append(rec.strip())
        label_names.append("Recommended")

    # Create DataFrame
    company_test_names = pd.DataFrame({'test_name': test_names, 'label': ""})
    company_test_names_comma_sep = company_test_names
    company_test_names_comma_sep['seq'] = 1

    delimiter = ','
    company_test_names_comma_sep = company_test_names_comma_sep.groupby('seq')['test_name'].apply(lambda x: delimiter.join(x)).reset_index()
    #company_test_names_comma_sep
    return company_test_names,company_test_names_comma_sep
    


# -

# # PACKAGE WISE DATA WITH GENDER, COMPONENTS NAME

# +
conn = mysql.connector.connect(
  user='nitesh.kr', password='yRpRtyJR0XSqVLibLT', host='gateway.echl.in', database='testenv_prod',port=10004)
  ##user='nitesh.kr', password='yRpRtyJR0XSqVLibLT', host='dms.echl.in', database='testenv_prod',port=3306)


cursor = conn.cursor()

query = '''SELECT Package_id,package_name,gender,test_id,NAME 'compts_name'
FROM(
SELECT ppd.package_id,pm.name AS package_name, pm1.id 'test_id',pm1.name,ppd.version,pm.gender
##,pp.mrp, pp.price
FROM package_master AS pm
LEFT JOIN package_profile_detail AS ppd ON pm.id = ppd.package_id
LEFT JOIN b2b_team_lead_package btlp ON pm.id = btlp.package_id
##LEFT JOIN product_price AS pp ON pm.id = pp.deal_type_id AND pp.deal_type = 'package'
LEFT JOIN profile_master AS pm1 ON pm1.id= ppd.profile_id AND pm1.isactive = 1
WHERE pm.isactive =1
AND ppd.active_status =1
AND team_lead_id IS NULL
GROUP BY pm.name, pm1.name,ppd.version
HAVING pm1.name IS NOT NULL
UNION ALL
SELECT ppd.package_id,pm.name AS package_name, pm1.id 'test_id',pm1.name,ppd.version,pm.gender
##,pp.mrp, pp.price
FROM package_master AS pm
LEFT JOIN package_parameter_detail AS ppd ON pm.id = ppd.package_id
LEFT JOIN b2b_team_lead_package btlp ON pm.id = btlp.package_id
##LEFT JOIN product_price AS pp ON pm.id = pp.deal_type_id AND pp.deal_type = 'package'
LEFT JOIN parameter_master AS pm1 ON pm1.id= ppd.parameter_id AND pm1.isactive = 1
WHERE pm.isactive =1
AND team_lead_id IS NULL
AND ppd.active_status =1
GROUP BY pm.name, pm1.name,ppd.version
HAVING pm1.name IS NOT NULL) z1
GROUP BY package_name,NAME,VERSION,gender
HAVING package_id IN (690,5015,7010,2313,404,1818,159,540,5175,498,439,5074,1037,
3961,5076,5075,671,5610,665,283,715,1103,836,714,399,
1737,7068,5499,1738,217,524,523,276,833,4423,221,711,181,5899,91)
ORDER BY package_name,NAME,VERSION
'''
myc = conn.cursor()
myc.execute(query)
table_rows = myc.fetchall()
df_b2c_package_total_split = pd.DataFrame(table_rows)
df_b2c_package_total_split.columns = [i[0] for i in myc.description]
#df_b2c_package_total_split.head()
##print(df)


delimiter = '_'
package_list_b2c = df_b2c_package_total_split.groupby(['package_name','gender'])['compts_name'].apply(lambda x: delimiter.join(x)).reset_index()
#package_list_b2c
# -

# # MATCHING THE RECOMMENDATION TESTS WITH THAT OF PACKAGES AS AVAILABLE WITH THE DIAGNOSTIC COMPANY TO GET THE BEST MATCHED PACKAGES AND INDIVIDUAL TESTS COMBINATION
#
# # MATCHING CRITERIA OF MIN 50% ON PACKAGES IS CONSIDERED ALONG WITH GENDER CONSIDERATIONS

# Split 'label' column in df2 into list
def final_outcome(company_test_names_concat,gender_input):
    label_list = company_test_names_concat['best_match_label'].iloc[0].split('_')

    def calculate_match_percentage(string1, string2):
        # Split strings by underscore delimiter
        tokens1 = string1.split("_")
        tokens2 = string2.split("_")
    
        # Get common tokens
        common_tokens = set(tokens1) & set(tokens2)
    
        # Calculate match percentage based on the number of common tokens
        match_percentage = len(common_tokens) / max(len(tokens1), len(tokens2)) * 100
        return match_percentage

    def match_percentage_between_columns(column1, column2):
        match_percentages = []
        for string1 in column1:
            for string2 in column2:
                match_percentage = calculate_match_percentage(string1, string2)
                match_percentages.append(match_percentage)
        return match_percentages



    # Function to check if any label is in compts_name
    def get_individual_tests(row):
        return [label for label in label_list if label not in row]




    df =pd.DataFrame()
    column1 = company_test_names_concat['best_match_label']
    column2 = package_list_b2c['compts_name']
    match_percentage = match_percentage_between_columns(column1, column2)

    # Apply function to each row in compts_name
    df['individual_tests'] = package_list_b2c['compts_name'].apply(get_individual_tests)

    df['match'] = match_percentage
    sorted_df = df.sort_values(by='match', ascending=False)

    company_test_names_concat['key'] = 0
    package_list_b2c['key'] = 0

    result = pd.merge(company_test_names_concat, package_list_b2c, on='key', how='outer').drop('key', axis=1)
    #result
    #result

    df_result_final = pd.concat([result,df],axis = 1)
    
    df_result_final = df_result_final.assign(Package=lambda x: ['package_{}'.format(i + 1) for i in range(len(x))])
    
    if gender_input == 'M':
        df_result_final = df_result_final[((df_result_final['match']>=50) & 
                                      ((df_result_final['gender'] == 'male') |(df_result_final['gender'] == 'both')))].head(3)
        return df_result_final
    else:
        df_result_final = df_result_final[((df_result_final['match']>=50) & 
                                      ((df_result_final['gender'] == 'female') |(df_result_final['gender'] == 'both')))].head(3)
        return df_result_final
    #return df_result_final['package_name'],df_result_final['individual_tests']


# +
import streamlit as st
import openai
from word_similarity_updated_new import find_best_matches_from_dataframe


# Load your trained model and tokenizer
#classifier.tokenizer = BertTokenizer.from_pretrained("model_name")
#classifier.model = BertModel.from_pretrained("model_name")
#classifier.classifier = LogisticRegression(max_iter=200)
#classifier.classifier.fit(classifier.train_embeddings, classifier.train_labels)  # Refit if required


# Set OpenAI API key
openai.api_key = 'openAIKey'

# Function to get recommendations from OpenAI
def get_recommendations(age, gender, symptoms, habits):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant of organization named HEALTHIANS IN INDIA specialized in recommending pathology tests only WITH NAMES  OF Healthians only separated by comma based on user inputs. Expected output is Recommended Basic Pathology tests only. Please avoid abbreviation of test names if complete names are available. No xray, urg and radiology tests to be recommened and no generic names to be recommended like basic metabolic panel"
        },
        {
            "role": "user",
            "content": f"Customer Details:\n\nAge: {age}\nGender: {gender}\nSymptoms: {symptoms}\nHabits: {habits}.\n\nRecommended Pathology Tests:"
        }
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100
    )
    
    return response.choices[0].message['content'].strip().split(',')

# Streamlit UI
st.title("AI-Agent Driven Pathology Test Recommender")

# Get user inputs
age = st.number_input("Age in years:", min_value=0, max_value=120, step=1)
gender = st.selectbox("Gender:", ('M', 'F'))
gender_input = gender
symptoms = st.text_area("Symptoms of any health condition/s:")
habits = st.text_area("Habits:")

if st.button("Get Recommendations"):
    if age and gender and symptoms and habits:
        recommendations = get_recommendations(age, gender, symptoms, habits)
        #st.write("Recommended Pathology Tests:")
        
        company_tests,company_tests_comma_sep = recommendation_converter(recommendations)
        #st.write('The following tests should be done by you based on your inputs : \n',company_tests_comma_sep)
        #st.write('The following tests should be done by you based on your inputs : \n')
        input_df = pd.DataFrame(company_tests['test_name'])
        
        results_df = find_best_matches_from_dataframe(df, input_df)
        
        results_df['seq'] = 1
        delimiter = '_'
        delimiter_comma = ','
        results_df_concat = results_df.groupby('seq')['best_match_label'].apply(lambda x: delimiter.join(x)).reset_index()
        results_df_concat_comma_sep = results_df.groupby('seq')['best_match_label'].apply(lambda x: delimiter_comma.join(x)).reset_index()
        #company_test_names_concat
        #results_df_concat['best_match_label']
        package_tests_combo = final_outcome(results_df_concat,gender_input)
        #st.write(company_test_names_concat)
        st.write('Thanks for giving the details!!',
                 '\nBased on the inputs , we recommend, you should get the following diagnostics tests done :\n\n',
                 ', '.join(results_df_concat_comma_sep['best_match_label'].tolist()))
        
        st.write('Below are best value packages & individual tests combinations suggested for you , based on the recommendations given above :',
                 package_tests_combo[['Package','individual_tests']],index = False)
    else:
        st.write("Please fill in all the fields")
# -






