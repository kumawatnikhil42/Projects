from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

df=pd.read_csv("salary_prediction_data.csv")


lb_ed=LabelEncoder()
df["Education"]=lb_ed.fit_transform(df["Education"])

lb_l=LabelEncoder()
df["Location"]=lb_l.fit_transform(df["Location"])

lb_jt=LabelEncoder()
df["Job_Title"]=lb_jt.fit_transform(df["Job_Title"])

lb_g=LabelEncoder()
df["Gender"]=lb_g.fit_transform(df["Gender"])

x=df.drop(columns=["Salary"],axis=1)
y=df["Salary"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
rf=RandomForestRegressor(n_estimators=100,random_state=42)

rf.fit(x_train,y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
       
        education = request.form['education'] 
        location = request.form['location'] 
        job_title = request.form['job_title']
        gender = request.form['gender'] 
        age = request.form['age']
        experience = float(request.form['experience']) 
        
        
        education_encoded = lb_ed.transform([education])[0]
        location_encoded = lb_l.transform([location])[0]
        job_title_encoded = lb_jt.transform([job_title])[0]
        gender_encoded = lb_g.transform([gender])[0]
        
        
        input_data = np.array([[education_encoded, location_encoded, job_title_encoded, gender_encoded, age,experience]])
        
        
        predicted_salary = rf.predict(input_data)[0]
        
        return render_template('index.html', predicted_salary=predicted_salary)
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)


