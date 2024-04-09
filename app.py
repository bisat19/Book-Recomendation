import pickle
import streamlit as st
import numpy as np

PAGE_TITLE = "Book Recomendation | bisat19"
PAGE_ICON = "📖"
st.set_page_config(page_title=PAGE_TITLE,page_icon=PAGE_ICON)

st.header('Sistem Rekomendasi Buku dengan Machine Learning')
model = pickle.load(open('artifacts/model.pkl','rb'))
nama_buku = pickle.load(open('artifacts/nama_buku.pkl','rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl','rb'))
buku_pivot = pickle.load(open('artifacts/buku_pivot.pkl','rb'))

def fetch_poster(suggestion):
    nama_buku = []
    ids_index = []
    poster_buku = []

    for book_id in suggestion:
        nama_buku.append(buku_pivot.index[book_id])

    for i in nama_buku[0]:
        ids = np.where(final_rating['title'] == i)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_buku.append(url)

    return poster_buku

def recommend_books(nama_buku):
    list_buku = []
    buku_id = np.where(buku_pivot.index == nama_buku)[0][0]
    distance, suggestion = model.kneighbors(buku_pivot.iloc[buku_id,:].values.reshape(1,-1),n_neighbors = 6)
    poster_url = fetch_poster(suggestion)
    for i in range(len(suggestion)):
        buku = buku_pivot.index[suggestion[i]]
        for j in buku: 
            list_buku.append(j)
    return list_buku,poster_url

buku_terpilih = st.selectbox(
    'Type or select a book',
    nama_buku
)

if st.button('Show Recommendation'):
    buku_rekomendasi, poster_url = recommend_books(buku_terpilih)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(buku_rekomendasi[1])
        st.image(poster_url[1])
    with col2:
        st.text(buku_rekomendasi[2])
        st.image(poster_url[2])
    with col3:
        st.text(buku_rekomendasi[3])
        st.image(poster_url[3]) 
    with col4:
        st.text(buku_rekomendasi[4])
        st.image(poster_url[4])
    with col5:
        st.text(buku_rekomendasi[5])
        st.image(poster_url[5])