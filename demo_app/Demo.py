import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

data = pd.read_csv("words_preprocessed.csv")
model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr') #TODO add different models

def get_embeddings(embedding_file_path):
    return np.load(embedding_file_path)

def setup_vectorDB(embeddings,method):
    vector_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(vector_dimension) #TODO Add different methods
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    return index

# This function will return a sorted (closest to most far) dataframe of words to the center word 
def getSortedCenterFlatL2(center_word,index):
    # encode the guessed word
    encoded_search = model.encode(center_word)
    search_vector = np.array([encoded_search])
    faiss.normalize_L2(search_vector)

    k = index.ntotal
    distances, ann = index.search(search_vector, k=k)

    results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
    merged_df = pd.merge(results,data,left_on='ann',right_index=True)
    return merged_df.iloc[1:]

def findGuessDistance(df,word):
    place_of_word = df[df["Words"] == word].index
    return place_of_word

#TODO try implementing binary search by sort the word list by first character at the beginning, to improve search time efficiency.
def isWordInList(word_list, guessed_word):
    if guessed_word == st.session_state.center_word:
        return True
    if guessed_word in word_list:
        return True
    return False

def checkIfGuessed(guessed_word):
    for obj in st.session_state.guessed_distances:
        if obj["word"] == guessed_word:
            return True
    return False






if "game_started" not in st.session_state:
    st.session_state.game_started = False

def start_game():
    st.session_state.game_started = True
def reset_game():
    st.session_state.game_started = False
    st.session_state.center_word = ""
    st.session_state.guessed_distances=[]



if not st.session_state.game_started:
    st.session_state
    if st.button("Play", type="primary",use_container_width = True, on_click = start_game):
        st.session_state.game_started = True
    st.write("")
    st.write("")
    st.write("")

    is_random = st.checkbox("Random Word",value = True)
    if is_random:
        st.divider()
        rand_array = np.random.randint(len(data["Words"]),size = 1)
        rand_word = data["Words"].tolist()[rand_array[0]]
        st.session_state.center_word = rand_word

    if not is_random:
        word_to_found = st.text_input(":red[Word to be found]")
        if st.button("Set Word", type="primary"):
            st.session_state.center_word = word_to_found
        st.divider()

    st.selectbox(
        ":red[Embedding Model]",
        ("bert-base-turkish-cased-mean-nli-stsb-tr")
    )
    st.divider()

    st.selectbox(
        ":red[VectorDB]",
        ("FAISS","Chroma")
    )
    st.divider()

    st.selectbox(
        ":red[Search Method]",
        ("HNSW","Euclidean distance (L2)","Cosine Similarity")
    )
    st.divider()
else:
    if "guessed_distances" not in st.session_state:
        st.session_state.guessed_distances = []
    if "game_over" not in st.session_state:
        st.session_state.game_over = False
    st.session_state.game_over = False    
    st.button("Reset", type = "primary",on_click = reset_game)
    #load statics
    embeddings = get_embeddings("statics/Embeddings/embeddings_bert_v02.npy")

    #setup vectorDB
    index = setup_vectorDB(embeddings,"L2")

    sorted_df = getSortedCenterFlatL2(st.session_state.center_word,index)

    #THIS IS A SHORTCUT FOR REMOVING THE 2 WORD PHRASES IN DF PLEASE FIX THIS LATER
    # one_word_df = sorted_df[~sorted_df['Words'].str.contains(" ", regex=False,na=False)]
    # one_word_df = one_word_df.reset_index(drop=True)


    if st.checkbox("Show words"):
        st.write("Center word:{}".format(st.session_state.center_word))
        st.dataframe(sorted_df,use_container_width = True)
    st.write("")
    st.write("")
    st.write("")

    guessed_word = st.text_input(":red[Guess a word]")
    guessed_word = guessed_word.lower()
    if st.button("Submit Guess"):
        if checkIfGuessed(guessed_word):
            st.warning('You allready guessed that word', icon="⚠️")
        elif isWordInList(word_list=list(sorted_df["Words"]), guessed_word=guessed_word):
            if guessed_word == st.session_state.center_word:
                st.balloons()
                st.session_state.game_over = True
            else:
                distance = findGuessDistance(sorted_df,guessed_word)
                distance_dict = {"word":guessed_word,"distance":distance[0]+1}
                # if distance not in st.session_state.guessed_distances:
                st.session_state.guessed_distances.append(distance_dict)
                st.session_state.guessed_distances = sorted(st.session_state.guessed_distances, key=lambda x: x['distance'])
        else:
            st.warning('Sorry We dont know this word', icon="⚠️")

    for distance_dict in st.session_state.guessed_distances:
        st.progress((sorted_df.shape[0] - distance_dict["distance"]) / sorted_df.shape[0],text = ":red[{}]: {}".format(distance_dict["word"],distance_dict["distance"]))    
    # st.write("Seni çok seviyorum buse")
    if st.session_state.game_over:
        st.success("You found the word! Congratulations")


    