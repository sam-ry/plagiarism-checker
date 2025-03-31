import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tempfile

def generate_heatmap(filenames, similarity_matrix):
    df = pd.DataFrame(similarity_matrix, index=filenames, columns=filenames)

    plt.figure(figsize=(10,8))

    custom_palette = sns.color_palette('Greys', as_cmap = True)
    sns.heatmap(df, annot=True, cmap=custom_palette, fmt = '.2f', linewidths=0.5)
    plt.title('Document Similarity heatmap')

    # save as image
    heatmap_path = os.path.join(tempfile.gettempdir(), 'similarity_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches = 'tight')
    plt.close()

    return heatmap_path