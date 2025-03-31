import os
import fitz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_text(pdf_path):
    text=''
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def read_pdfs(folder_path):
    files = {} #dict of filename and contents
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            files[filename] = extract_text(pdf_path)
    return files

def cosine_similarity_value(filetexts, filenames):
    # filenames = list(read_pdfs.keys())
    vectors = TfidfVectorizer()
    tfidf_matrix = vectors.fit_transform(filetexts)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    results = []
    num_files = len(filenames)

    for i in range(num_files):
        for j in range(i+1, num_files):
            similarity = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
            results.append({
                'file1': filenames[i],
                'file2': filenames[j],
                'similarity': round(similarity*100,3)
            })
    return results, similarity_matrix, filenames