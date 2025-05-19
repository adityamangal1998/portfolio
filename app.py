from flask import Flask, render_template, request

app = Flask(__name__)

# Portfolio project data
projects = {
    "tiny-llama": {
        "title": "Fine-Tuning TinyLlama on WhatsApp Chats",
        "description": "Fine-tuned TinyLlama-1.1B on personal WhatsApp chat data to build a custom AI chatbot. Implemented LoRA & PEFT for optimization and deployed for real-time interaction.",
        "implementation": "The implementation involved preprocessing WhatsApp chat exports, tokenizing messages, and fine-tuning the TinyLlama-1.1B model using Low-Rank Adaptation (LoRA) techniques. PEFT (Parameter-Efficient Fine-Tuning) was used to optimize the training process while maintaining model quality.",
        "challenges": "Handling conversational context efficiently was a key challenge. This was addressed by implementing a sliding window approach to maintain sufficient context without exceeding token limits. Another challenge was optimizing the model for deployment on consumer hardware, which was solved using quantization techniques.",
        "outcomes": "The resulting model accurately mimics the writing style and response patterns of the target individual. It achieves over 90% coherent responses in blind tests.",
        "technologies": ["TinyLlama", "LoRA", "PEFT", "Python", "HuggingFace"],
        "category": "AI & NLP",
        "date": "2023",
        "github_link": "https://github.com/adityamangal1998",
        "images": ["app-1.jpg", "project-1.jpg", "project-2.jpg"]
    },
    "car-parking-solution": {
        "title": "Smart Car Parking Detector",
        "description": "Designed a lightweight computer vision tool to automatically identify available parking spots in real-time using classic image processing. No deep learning required, making it fast and resource-efficient for edge devices.",
        "implementation": "The system uses traditional computer vision techniques like background subtraction, contour detection, and perspective transformation to identify vacant parking spots. A reference frame establishes the baseline empty parking lot, and then real-time frames are compared to detect changes.",
        "challenges": "Handling varying lighting conditions was a significant challenge. The solution implemented adaptive thresholding and histogram equalization to normalize lighting. Accurate spot detection despite partial occlusions was addressed using robust contour analysis.",
        "outcomes": "The system achieves 95% accuracy in detecting available spots while maintaining real-time performance (25+ FPS) on resource-constrained hardware like Raspberry Pi.",
        "technologies": ["OpenCV", "Python", "Image Processing", "Edge Computing"],
        "category": "Computer Vision",
        "date": "2022",
        "github_link": "https://github.com/adityamangal1998/Car-Parking-Solution",
        "images": ["app-4.jpg", "project-1.jpg", "project-2.jpg"]
    },
    "image-data-augmentor": {
        "title": "Image-Data-Augmentor (PyPI Package)",
        "description": "Published a Python package offering advanced and customizable image augmentation pipelines. Used by ML practitioners to boost model robustness with diverse, realistic image transformations.",
        "implementation": "The PyPI package implements a comprehensive suite of image augmentation techniques with a pipeline pattern where transformations can be chained and applied with configurable probability.",
        "challenges": "Ensuring consistency across different image formats and maintaining performance with large datasets were key challenges. Optimized processing using NumPy vectorization and a caching system were implemented as solutions.",
        "outcomes": "The package has been downloaded over 1,000 times from PyPI and is being used in several computer vision projects, with users reporting 5-15% model accuracy improvements.",
        "technologies": ["Python", "Pillow", "NumPy", "PyPI", "Computer Vision"],
        "category": "Machine Learning Tools",
        "date": "2021",
        "github_link": "https://github.com/adityamangal1998/Image-Data-Augmentor",
        "images": ["app-5.jpg", "project-1.jpg", "project-2.jpg"]
    },
    "concurrent-image-read": {
        "title": "Concurrent-Image-Read (PyPI Library)",
        "description": "Developed and published a Python library for high-speed, multi-threaded image loading. Accelerates data pipelines for deep learning by minimizing I/O bottlenecks during training.",
        "implementation": "This library uses a thread pool architecture to prefetch and process images concurrently. It implements a producer-consumer pattern with a priority queue to optimize memory usage while maintaining high throughput.",
        "challenges": "Balancing memory usage with performance was the primary challenge. The implementation uses a dynamic buffer sizing algorithm that adjusts based on available system resources and observed loading patterns.",
        "outcomes": "Benchmarks show a 3-5x speedup in image loading pipelines compared to sequential loading, resulting in significantly faster training times for computer vision models.",
        "technologies": ["Python", "Threading", "Concurrency", "PyPI", "Deep Learning"],
        "category": "Developer Tools",
        "date": "2021",
        "github_link": "https://github.com/adityamangal1998/Concurrent-Image-Read",
        "images": ["app-6.jpg", "project-1.jpg", "project-2.jpg"]
    },
    "prescription-scanning": {
        "title": "Prescription Scanning with LLM and RAG",
        "description": "Developed a system to extract structured data from handwritten prescriptions using RAG and Bedrock-powered LLMs integrated with a medicine vector DB.",
        "implementation": "The system uses a multi-stage pipeline: first, OCR extracts text from prescription images; then, a specialized RAG system retrieves relevant medical information from a vector database of medicines; finally, LLMs structure and validate the extracted information.",
        "challenges": "Handwritten prescriptions vary widely in style and clarity, requiring robust preprocessing and error correction. Integration with the medicine database also required careful entity resolution and fuzzy matching.",
        "outcomes": "The system achieves 92% accuracy in extracting medication names, dosages, and instructions from handwritten prescriptions, enabling automated processing for pharmacy systems.",
        "technologies": ["LLMs", "RAG", "FAISS", "Bedrock API", "OCR", "Vector Databases"],
        "category": "Healthcare AI",
        "date": "2023",
        "github_link": "",
        "images": ["app-3.jpg", "project-1.jpg", "project-2.jpg"]
    },
    "sql-queries": {
        "title": "Improving SQL Queries with Fine-Tuning",
        "description": "Fine-tuned LLaMA-2-7B to generate and optimize SQL queries with RAG-enhanced contextual retrieval. Focused on performance and real-time query improvements.",
        "implementation": "Used a hybrid approach combining fine-tuning on SQL datasets and RAG to retrieve schema information. The model was optimized to understand database schemas and generate efficient, error-free SQL queries.",
        "challenges": "Ensuring the generated queries were not just syntactically correct but also efficient was challenging. Implemented a validation layer that analyzes query execution plans and suggests optimizations.",
        "outcomes": "The system reduces query writing time by 70% while generating queries that are 35% more efficient than those written by junior developers. Now integrated into several database management tools.",
        "technologies": ["LLaMA-2", "RAG", "SQL", "Hugging Face", "Database Optimization"],
        "category": "Developer Tools",
        "date": "2023",
        "github_link": "",
        "images": ["app-2.jpg", "project-1.jpg", "project-2.jpg"]
    },
    "insa-engine": {
        "title": "InSa-Engine",
        "description": "Built a robust service for extracting key information from insurance policy PDFs using RAG and LLMs. Integrated Polifyx and CAMS APIs.",
        "implementation": "Created a high-throughput document processing pipeline using Kafka for message queuing and FastAPI for the service layer. Documents are processed through OCR, then key information is extracted using RAG-enhanced LLMs.",
        "challenges": "Insurance documents have complex, non-standard formats and often contain tables and structured data. Developed specialized document segmentation and table extraction components to handle these challenges.",
        "outcomes": "The engine processes over 10,000 insurance policies daily with 94% extraction accuracy, saving hundreds of hours of manual data entry and enabling automated claim processing.",
        "technologies": ["FastAPI", "LLMs", "RAG", "Kafka", "OCR", "AWS"],
        "category": "FinTech",
        "date": "2024",
        "github_link": "",
        "images": ["product-1.jpg", "project-1.jpg", "project-2.jpg"]
    },
    "table-detection": {
        "title": "Table Detection using Vision-Language Models",
        "description": "Fine-tuned Qwen2.5-VL-3B on PubTables-1M to detect and extract structured tables from scanned documents. Focused on layout and text accuracy.",
        "implementation": "Used a vision-language model approach that combines visual understanding of document layout with text comprehension. Fine-tuned on the PubTables-1M dataset with additional custom annotation to improve performance on complex table structures.",
        "challenges": "Tables in scanned documents often have inconsistent formatting, merged cells, and poor image quality. Developed specialized data augmentation techniques to improve robustness to these variations.",
        "outcomes": "The system achieves state-of-the-art 96% accuracy on public benchmarks and successfully extracts even complex nested tables from low-quality scans.",
        "technologies": ["Qwen2.5-VL", "Vision-Language Models", "PubTables", "OCR", "PyTorch"],
        "category": "Document Intelligence",
        "date": "2023",
        "github_link": "",
        "images": ["product-2.jpg", "project-1.jpg", "project-2.jpg"]
    },
    "image-super-resolution": {
        "title": "Image Super-Resolution with ViT",
        "description": "Used SwinIR (a ViT-based model) to upscale low-res images while preserving fine details. Optimized for real-world document clarity.",
        "implementation": "Implemented and fine-tuned SwinIR models with a focus on document images. Added custom training objectives that prioritize text readability and line preservation in documents.",
        "challenges": "Balancing general image quality with specific text legibility requirements was challenging. Developed a hybrid loss function combining perceptual and text-specific components.",
        "outcomes": "The system can upscale document images by up to 4x while maintaining text legibility, significantly improving OCR accuracy on low-resolution documents.",
        "technologies": ["SwinIR", "PyTorch", "OpenCV", "ViT", "Image Processing"],
        "category": "Computer Vision",
        "date": "2022",
        "github_link": "",
        "images": ["product-3.jpg", "project-1.jpg", "project-2.jpg"]
    },
    "vit-image-captioning": {
        "title": "ViT-based Image Captioning",
        "description": "Combined ViT and GPT via BLIP for context-aware image captioning. Fine-tuned on custom datasets for AI storytelling.",
        "implementation": "Used BLIP (Bootstrapping Language-Image Pre-training) to connect vision and language models. Fine-tuned the system on a custom dataset of images with detailed, contextually rich captions.",
        "challenges": "Creating captions that are not just accurate but also contextually relevant and engaging required significant innovation in the training process and evaluation metrics.",
        "outcomes": "The system generates detailed, creative captions that outperform standard image captioning models in human evaluation tests, particularly for storytelling applications.",
        "technologies": ["ViT", "GPT", "BLIP", "Transformers", "Multimodal AI"],
        "category": "Computer Vision & NLP",
        "date": "2023",
        "github_link": "",
        "images": ["branding-1.jpg", "project-1.jpg", "project-2.jpg"]
    },
    "document-extraction": {
        "title": "Document Extraction using LayoutLM V3",
        "description": "Built a key-value extractor using LayoutLM V3, trained on SROIE, for accurate structured data retrieval from complex documents.",
        "implementation": "Fine-tuned LayoutLM V3 on the SROIE dataset and additional proprietary data. Developed a post-processing pipeline to structure extracted information into consistent key-value pairs.",
        "challenges": "Document layouts vary greatly across different sources and formats. Created a robust preprocessing pipeline that normalizes document layouts while preserving spatial relationships.",
        "outcomes": "The system extracts key information from invoices, receipts, and forms with 97% accuracy, enabling automated data entry for accounting and ERP systems.",
        "technologies": ["LayoutLM V3", "SROIE", "Transformers", "Document Understanding"],
        "category": "Document Intelligence",
        "date": "2023",
        "github_link": "",
        "images": ["branding-2.jpg", "project-1.jpg", "project-2.jpg"]
    },
    "kyc-extraction": {
        "title": "KYC Docs Extraction",
        "description": "Used YOLOv5 and Google OCR to extract information from PAN, Aadhaar, Passport, and Voter ID documents.",
        "implementation": "Created a two-stage pipeline: first, YOLOv5 detects and classifies document types and locates key information fields; then, targeted OCR extracts the specific data from those regions.",
        "challenges": "The wide variety of document formats, security features, and image quality issues made consistent extraction difficult. Developed document-specific preprocessing and validation rules.",
        "outcomes": "The system processes over 5,000 KYC documents daily with 98% accuracy, dramatically reducing manual verification time for financial institutions.",
        "technologies": ["YOLOv5", "OCR", "Rule-based NLP", "Computer Vision"],
        "category": "FinTech",
        "date": "2022",
        "github_link": "",
        "images": ["branding-3.jpg", "project-1.jpg", "project-2.jpg"]
    },
    "qa-insurance": {
        "title": "Question-Answering System for Insurance Docs",
        "description": "Fine-tuned BERT and used K-Means clustering to enable question answering in insurance policies within Polifyx.",
        "implementation": "Used BERT for extractive question answering and enhanced it with K-Means clustering to categorize policy sections and improve answer retrieval accuracy.",
        "challenges": "Insurance policies contain complex legal language and domain-specific terminology. Created a specialized insurance vocabulary and fine-tuned on a curated dataset of policy Q&A pairs.",
        "outcomes": "The system correctly answers 87% of policy-related queries, reducing customer support workload and improving client self-service capabilities.",
        "technologies": ["BERT", "KMeans", "NLP", "Question Answering"],
        "category": "FinTech",
        "date": "2022",
        "github_link": "",
        "images": ["books-1.jpg", "project-1.jpg", "project-2.jpg"]
    },
    "robodoctor": {
        "title": "RoboDoctor - Healthcare Prediction Web App",
        "description": "Deployed ML models in a Flask app for predicting diseases like liver, cancer, heart ailments, and COVID-19 using symptoms and image data.",
        "implementation": "Created an integrated platform combining multiple prediction models: symptom-based classifiers for general diagnosis and specialized CNNs for analyzing medical images like X-rays and CT scans.",
        "challenges": "Balancing model accuracy with responsible healthcare applications was crucial. Implemented comprehensive uncertainty quantification and 'refer to doctor' recommendations for edge cases.",
        "outcomes": "The app has been used by over 5,000 users as a preliminary screening tool, with particularly strong performance in COVID-19 detection from chest X-rays (95% accuracy).",
        "technologies": ["Flask", "ML", "X-Ray Analysis", "CT Scan Processing", "Python"],
        "category": "Healthcare AI",
        "date": "2021",
        "github_link": "https://github.com/adityamangal1998/RoboDoc_Hack36",
        "images": ["books-2.jpg", "project-1.jpg", "project-2.jpg"]
    },
    "d3-driver-drowsiness": {
        "title": "D3 - Driver Drowsiness Detection",
        "description": "Developed a drowsiness detector using 468 facial landmarks from Google MediaPipe to track signs of sleepiness and yawning.",
        "implementation": "Used MediaPipe to track precise facial landmarks in real-time video. Implemented specialized algorithms to detect eye closure duration, blink frequency, yawning, and head position changes.",
        "challenges": "Ensuring reliability across different lighting conditions and with users wearing glasses was difficult. Created adaptive thresholding and specialized detection paths for these scenarios.",
        "outcomes": "The system detects drowsiness with 94% accuracy in real-world driving conditions and can issue timely alerts before safety becomes compromised.",
        "technologies": ["MediaPipe", "CV2", "Python", "Computer Vision"],
        "category": "Safety Systems",
        "date": "2022",
        "github_link": "https://github.com/adityamangal1998/D3_HackStack",
        "images": ["books-3.jpg", "project-1.jpg", "project-2.jpg"]
    }
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
