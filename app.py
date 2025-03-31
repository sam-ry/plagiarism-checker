import os
import pandas as pd
import tempfile
from flask import Flask, request, render_template, jsonify, send_file, session, redirect, url_for
from compute import extract_text, read_pdfs, cosine_similarity_value
from visualization import generate_heatmap
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend to avoid Tkinter errors
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Secret key for session management
UPLOAD_FOLDER = 'user_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Hardcoded credentials (for now)
USERNAME = 'admin'
PASSWORD = 'password'
@app.route('/')
def home():
    """Redirect to login page if not logged in"""
    if session.get('logged_in'):
        return redirect(url_for('upload_files'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if session.get('logged_in'):
        return redirect(url_for('upload_files'))  # If already logged in, go to upload page

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('upload_files'))
        else:
            return 'Invalid credentials. Try again.'
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logs the user out and redirects to login page"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('files')  #retrives the list of files from the form with input field named files
        folder_path = request.form.get('folder_path') #retrives folderpath

        texts, filenames = {}, []
        for file in uploaded_files:
            #  if the files are uploaded manually
            if file.filename.endswith('.pdf'):
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                texts[file.filename] = extract_text(file_path)  

            # if the folderpath is provided
        if folder_path and os.path.exists(folder_path):
            texts.update(read_pdfs(folder_path))

        if len(texts) < 2:
            return 'Upload atleast two PDFs to compare'
            
        similarity_results, similarity_matrix, filenames = cosine_similarity_value(list(texts.values()), list(texts.keys()))
        # Generate heatmap
        heatmap_path = generate_heatmap(filenames, similarity_matrix)
        # tempdir for the manually uploaded files
        df = pd.DataFrame(similarity_results)
        reportfile_path = os.path.join(tempfile.gettempdir(), 'similarity_report.csv')
        # heatmap_path = os.path.join(tempfile.gettempdir(), 'heatmap_report.csv')
        df.to_csv(reportfile_path, index=True)
        return render_template('result.html', results = similarity_results, reportfile_path = reportfile_path, heatmap_path=heatmap_path)
    return render_template('upload.html')



@app.route('/download_report')
def download_report():
    reportfile_path = os.path.join(tempfile.gettempdir(), 'similarity_report.csv')
    return send_file(reportfile_path, as_attachment = True)

@app.route('/download_heatmap')
def download_heatmap():
    heatmap_path = os.path.join(tempfile.gettempdir(), 'similarity_heatmap.png')
    return send_file(heatmap_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug = True)