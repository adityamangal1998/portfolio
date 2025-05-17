from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/portfolio-details')
def portfolio_details():
    return render_template('portfolio-details.html')

@app.route('/service-details')
def service_details():
    return render_template('service-details.html')

@app.route('/starter-page')
def starter_page():
    return render_template('starter-page.html')

@app.route('/resume')
def resume():
    return render_template('resume.html')

@app.route('/contact', methods=['POST'])
def contact():
    # Handle the form submission here
    name = request.form.get('name')
    email = request.form.get('email')
    subject = request.form.get('subject')
    message = request.form.get('message')
    # You can add logic to process the form data, such as sending an email
    return "Form submitted successfully!"

if __name__ == '__main__':
    app.run(debug=True)
