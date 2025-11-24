import flask
import pickle
import numpy as np
import pandas as pd

# Use pickle to load in the trained model
with open('webapp/model/credit_approval_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        # Retrieve submitted inputs
        gender = flask.request.form.get('gender')
        own_car = flask.request.form.get('own_car')
        own_realty = flask.request.form.get('own_realty')
        income = flask.request.form.get('income')
        age = flask.request.form.get('age')
        total_worked = flask.request.form.get('total_worked')

        basic_inputs_list = [gender, own_car, own_realty, income, age, total_worked]

        employment_status = flask.request.form.get('employment_status')
        education = flask.request.form.get('education')
        family_status = flask.request.form.get('family_status')
        housing = flask.request.form.get('housing')

        dummy_inputs_list = []

        # Convert inputs for model
        employment_status_list = ['Government employee', 'Hourly/commission employee', 'Pensioner', 'Salaried employee', 'Student']
        # Loop through all employment statuses
        for element in employment_status_list:
            # If current element equals employment status in list, mark element as 1 and others as 0
            if element == employment_status:
                dummy_inputs_list.append(1)
            else:
                dummy_inputs_list.append(0)

        education_list = ["Bachelor's degree", "High school", "Master's degree or higher", "Some college", "Some high school"]
        education_results = []
        # Loop through all employment statuses
        for element in education_list:
            # If current element equals education in list, mark element as 1 and others as 0
            if element == education:
                dummy_inputs_list.append(1)
            else:
                dummy_inputs_list.append(0)

        family_status_list = ["Married", "Separated", "Single", "Widow"]
        # Loop through all employment statuses
        for element in family_status_list:
            # If current element equals family status in list, mark element as 1 and others as 0
            if element == family_status:
                dummy_inputs_list.append(1)
            else:
                dummy_inputs_list.append(0)

        housing_list = ["Homeowner", "Public housing", "Renting", "With parents"]
        # Loop through all employment statuses
        for element in housing_list:
            # If current element equals housing in list, mark element as 1 and others as 0
            if element == housing:
                dummy_inputs_list.append(1)
            else:
                dummy_inputs_list.append(0)

        inputs_list = basic_inputs_list + dummy_inputs_list
        to_np_arr = np.array(inputs_list)
        predict_inputs = to_np_arr.reshape(1,-1)
        predict_df = pd.DataFrame(predict_inputs, columns = ["Applicant_Gender", "Owned_Car", "Owned_Realty", "Total_Income", "Applicant_Age", "Years_of_Working",
                                                             "Government employee", "Hourly/commission employee", "Pensioner", "Salaried employee", "Student",
                                                             "Married", "Separated", "Single", "Widow", "Bachelor's degree", "High school", "Master's degree or higher", "Some college", "Some high school",
                                                             "Homeowner", "Public housing", "Renting", "With parents"])
        prediction = model.predict(predict_df)
        
        # Convert prediction to text
        if prediction == 0:
            prediction_str = "Not Approved"
        elif prediction == 1:
            prediction_str = "Approved"
        return(flask.render_template('main.html', result=prediction_str))

if __name__ == '__main__':
    app.run(port=1234,debug=True)