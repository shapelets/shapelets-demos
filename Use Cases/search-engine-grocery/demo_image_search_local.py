from shapelets.apps import dataApp
from towhee import ops, pipe
import numpy as np
from pathlib import Path
import shutil
from shapelets.storage import Shelf, Transformer
from shapelets.indices import EmbeddingIndex

def textChanged(text:str):
    query = text_encoder(text).get_dict()['vec']
    result = index.search(query, num_images)

    images = []
    labels = []
    for r in result:
        if data[r.oid][0] and data[r.oid][1]:
            labels.append(data[r.oid][0] + ' ' + str(r.distance))
            images.append(data[r.oid][1])
    for i in range(num_images):
        try:
            list_images[i].src = images[i]        
        except:
            list_images[i].src = "https://upload.wikimedia.org/wikipedia/commons/2/2b/No-Photo-Available-240x300.jpg"  
        list_images[i].width = 150
        list_images[i].height = 150
        yield list_images[i]

print("Loading vector data...")

data = np.load('data.npy', allow_pickle=True)
vectors = np.load('vectors.npy', allow_pickle=True)

basePath = Path('./')
shelfPath = basePath / 'shelf'
indexPath = basePath / 'index'

if shelfPath.exists():
    shutil.rmtree(shelfPath)

if indexPath.exists():
    shutil.rmtree(indexPath)

shelfPath.mkdir(parents=True, exist_ok=True)
indexPath.mkdir(parents=True, exist_ok=True)

archive = Shelf.local([Transformer.compressor()], shelfPath)
index = EmbeddingIndex.create(dimensions=vectors.shape[1], shelf=archive, base_directory=indexPath)

builder = index.create_builder()

docId = 0
for vector in vectors:
    builder.add(vector, docId)
    docId += 1

builder.upsert()

print("Vector data loaded")

text_encoder = (
        pipe.input('text')
        .map('text', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='text'))
        .map('vec', 'vec', lambda x: x / np.linalg.norm(x))
        .output('text', 'vec')
    )

app = dataApp()
txt = app.title('Image search engine')
text = app.input_text_area(title="Text title", on_change=textChanged)

num_images = 20
list_images = []
for i in range(num_images):
    list_images.append(app.image(preview=False))

grid = app.grid_layout()
for i in range(0, num_images, 5):
    grid.place([list_images[i], list_images[i+1], list_images[i+2], list_images[i+3], list_images[i+4]])
