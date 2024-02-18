import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from embeddingModels.model import *
from splitter.chunk import *
from langchain_community.vectorstores import FAISS

UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'pdf', 'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def computation():
    selected_model = request.form.get('model_name')
    if not selected_model:
        return render_template('index.html', error="Please select a model")
    embeddings=load_selected_embeddings(selected_model)
       
    print("Embedding Done")

    chunk_size = int(request.form.get('chunk_size'))
    if not chunk_size:
        return render_template('index.html', error="Please select a chunk size")
    text_chunks=chunk(chunk_size)
    
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    vector_store.save_local("vectorDB")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Only handle individual file uploads and validate extensions
        uploaded_files = request.files.getlist('file')
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                flash('File uploaded successfully!')
            else:
                flash('Invalid file type. Only PDF and CSV files are allowed.')
        computation()
        return redirect(url_for('upload_file'))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
