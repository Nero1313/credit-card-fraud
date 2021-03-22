from flask import Flask, request,abort, send_from_directory, render_template, redirect
import pandas as pd
from werkzeug.utils import secure_filename
import os

images_folder = os.path.join('static', 'images')
file_folder = os.path.join('data')
csv_folder = os.path.join('output_data')

def predict_fraud(file):
    df = pd.read_csv(file, engine='python')
    df = df.drop(['Time'], axis=1)
    X = df.iloc[:, :].values
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    import joblib
    filename = 'finalized_model.sav'
    model = joblib.load(filename)
    result = model.predict(X)
    return result

# create rest server
app = Flask(__name__)

# static routing
app.config['images_folder'] = images_folder
app.config['UPLOAD_FOLDER'] = file_folder
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True


# route: mapping of the http method and the path
@app.route("/", methods=["GET"])
def root():
    image_1 = os.path.join(app.config['images_folder'], 'one.png')
    image_2 = os.path.join(app.config['images_folder'], 'two.png')
    image_3 = os.path.join(app.config['images_folder'], 'three.png')
    image_4 = os.path.join(app.config['images_folder'], 'four.png')
    return render_template("index.html", image_1=image_1, image_2=image_2, image_3=image_3, image_4=image_4)
# @app.route("/data", methods=["POST"])
# def post_data():
#     # get the test file from user
#     csvfile = request.form.get('csvfile')
#
#     result = predict_fraud(csvfile)
#     data = pd.DataFrame(result)
#     data.to_csv("./output_data/result.csv")
#
#     return render_template("result.html", result=result)

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file1']
        f.filename='test.csv'
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

        file = os.path.join(app.config['UPLOAD_FOLDER'], 'test.csv')
        result = predict_fraud(file)
        data = pd.DataFrame(result)
        data.to_csv("./output_data/result.csv")

        return render_template('result.html', result=result)

app.config["CSV_FILE"] = csv_folder

@app.route("/get-csv/<filename>")
def get_csv(filename):
    try:
        return send_from_directory(
            app.config["CSV_FILE"], filename=filename, as_attachment=True
        )
    except FileNotFoundError:
        abort(404)

# start the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000, debug=True)

