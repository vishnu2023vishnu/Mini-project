from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("placement_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/predict", methods=["POST"])
def predict():

    cgpa = float(request.form["cgpa"])
    internships = int(request.form["internships"])
    projects = int(request.form["projects"])
    workshops = int(request.form["workshops"])
    aptitude = float(request.form["aptitude"])
    softskills = float(request.form["softskills"])
    ssc = float(request.form["ssc"])
    hsc = float(request.form["hsc"])

    extracurricular = 1
    training = 1

    features = [[
        cgpa,
        internships,
        projects,
        workshops,
        aptitude,
        softskills,
        extracurricular,
        training,
        ssc,
        hsc
    ]]

    probability = model.predict_proba(features)
    score = round(max(probability[0]) * 100, 2)

    if score >= 80:
        status = "Excellent Chance 🚀"
    elif score >= 60:
        status = "Good Chance 👍"
    else:
        status = "Needs Improvement 📚"

    prediction = model.predict(features)

    result = "PLACED ✅" if prediction[0] == 1 else "NOT PLACED ❌"

    if cgpa < 7:
        suggestion = "Improve CGPA"
    elif aptitude < 60:
        suggestion = "Practice Aptitude"
    else:
        suggestion = "Excellent Profile"

    return render_template(
        "result.html",
        result=result,
        score=score,
        suggestion=suggestion,
        status=status
    )


if __name__ == "__main__":
    app.run(debug=True)
