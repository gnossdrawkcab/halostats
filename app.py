from flask import Flask, render_template, redirect, url_for
import subprocess

app = Flask(__name__)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to run the Python script when the button is pressed
@app.route('/run-script', methods=['POST'])
def run_script():
    # Replace 'main.py' with the path to the Python script you want to run
    try:
        subprocess.run(["python", "main.py"], check=True)
        return redirect(url_for('index'))  # Redirect back to the homepage after running the script
    except Exception as e:
        return "An error occurred: {}".format(e)

if __name__ == '__main__':
    app.run(debug=True)
