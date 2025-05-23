from flask import Flask, render_template, request

app = Flask(__name__)

# Portfolio project data
projects = {
    "tiny-llama": {
        "title": "Fine-Tuning TinyLlama on WhatsApp Chats",
        "description": "Fine-tuned TinyLlama-1.1B on personal WhatsApp chat data to build a custom AI chatbot. Implemented LoRA & PEFT for optimization and deployed for real-time interaction.",
        "technologies": ["TinyLlama", "LoRA", "PEFT", "Python", "HuggingFace"],
        "category": "AI & NLP",
        "date": "2023",
        "medium_link": "https://medium.com/@adityamangal98/fine-tuning-tinyllama-on-whatsapp-chats-build-your-own-personal-ai-chatbot-d7ae3144d2aa?sk=fc8f3da47e13097ff360a017088eb434",
        "images": ["tiny-llama-1.png", "tiny-llama-2.png"]
    },
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/portfolio/<project>')
def portfolio_details(project):
    # Get project data or return 404 if not found
    project_data = projects.get(project)
    if not project_data:
        return "Project not found", 404
        
    return render_template('portfolio-details.html', project_data=project_data)

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
