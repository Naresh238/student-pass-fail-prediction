
from flask import Flask, render_template, request
import joblib
import pandas as pd
app = Flask(__name__)
model = joblib.load("student_model.pkl")
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    if request.method == "POST":
        try:
            attendance = float(request.form["attendance"])
            python_marks = float(request.form["python_marks"])
            ml_marks = float(request.form["ml_marks"])
            values = [attendance, python_marks, ml_marks]
            if not all(0 <= value <= 100 for value in values):
                raise ValueError("Enter values between 0 and 100.")
            data = pd.DataFrame([{
                "Attendance": attendance,
                "Python_Marks": python_marks,
                "ML_Marks": ml_marks
            }])
            prediction = model.predict(data)[0]
            result = "Pass" if prediction == 1 else "Fail"

        except ValueError as e:
            error = str(e) or "Please enter valid numbers."
        except (KeyError, TypeError):
            error = "Please enter valid numbers."
    return render_template(
        "index.html", result=result, error=error
    )
if __name__ == "__main__":
    app.run(debug=True)
