import csv
import os
import pickle
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Smart Voting System", page_icon="🗳️", layout="wide")

DATA_DIR = "data"
NAMES_PATH = os.path.join(DATA_DIR, "names.pkl")
FACES_PATH = os.path.join(DATA_DIR, "faces_data.pkl")
VOTES_PATH = "Votes.csv"
COL_NAMES = ["NAME", "VOTE", "DATE", "TIME"]
PARTIES = ["BJP", "CONGRESS", "AAP", "NOTA"]
SAMPLES_PER_VOTER = 20

os.makedirs(DATA_DIR, exist_ok=True)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def load_dataset():
    if os.path.exists(NAMES_PATH) and os.path.exists(FACES_PATH):
        with open(NAMES_PATH, "rb") as f:
            names = pickle.load(f)
        with open(FACES_PATH, "rb") as f:
            faces = pickle.load(f)
        faces = np.asarray(faces)
        if len(faces) > 0 and len(names) == len(faces):
            return list(names), faces.reshape(len(faces), -1)
    return [], np.empty((0, 7500), dtype=np.uint8)


def save_dataset(names, faces):
    with open(NAMES_PATH, "wb") as f:
        pickle.dump(names, f)
    with open(FACES_PATH, "wb") as f:
        pickle.dump(faces, f)


def extract_face_features(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    preview = img.copy()
    if len(faces) == 0:
        return None, cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    cv2.rectangle(preview, (x, y), (x + w, y + h), (50, 50, 255), 2)

    crop_img = img[y : y + h, x : x + w]
    resized_img = cv2.resize(crop_img, (50, 50))
    flattened = resized_img.flatten().reshape(1, -1)
    return flattened, cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def has_already_voted(voter_id):
    if not os.path.exists(VOTES_PATH):
        return False
    try:
        with open(VOTES_PATH, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0].strip().lower() == str(voter_id).strip().lower():
                    return True
    except FileNotFoundError:
        return False
    return False


def record_vote(voter_id, party):
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")
    file_exists = os.path.isfile(VOTES_PATH) and os.path.getsize(VOTES_PATH) > 0

    with open(VOTES_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(COL_NAMES)
        writer.writerow([voter_id, party, date_str, time_str])


st.title("🗳️ Smart Voting System")

mode = st.sidebar.radio("Select Mode", ["Cast Vote", "Register Voter", "Vote Records"])
names_data, faces_data = load_dataset()

if mode == "Register Voter":
    st.header("Register New Voter Face")
    voter_id = st.text_input("Enter Voter Card Number").strip()

    if "captured_samples" not in st.session_state:
        st.session_state.captured_samples = []

    st.info(
        f"Capture up to **{SAMPLES_PER_VOTER}** clear face shots using your camera, then click **Save Voter Data**."
    )
    photo = st.camera_input("Capture Face Sample", key="register_cam")

    if photo is not None and st.button("Add Sample"):
        if not voter_id:
            st.error("Please enter a Voter Card Number first.")
        else:
            features, preview_rgb = extract_face_features(photo)
            if features is None:
                st.warning("No face detected. Ensure good lighting and face the camera.")
            else:
                st.session_state.captured_samples.append(features)
                st.image(preview_rgb, caption="Detected Face", width=320)
                st.success(
                    f"Added sample {len(st.session_state.captured_samples)}/{SAMPLES_PER_VOTER}"
                )

    if st.session_state.captured_samples:
        st.write(f"Samples ready: **{len(st.session_state.captured_samples)}**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Voter Data", type="primary"):
                if not voter_id:
                    st.error("Voter Card Number is required.")
                else:
                    new_faces = np.vstack(st.session_state.captured_samples)
                    count = len(new_faces)
                    updated_names = names_data + [voter_id] * count
                    if len(faces_data) > 0 and faces_data.shape[1] == new_faces.shape[1]:
                        updated_faces = np.vstack([faces_data, new_faces])
                    else:
                        updated_faces = new_faces
                    save_dataset(updated_names, updated_faces)
                    st.session_state.captured_samples = []
                    st.success(f"Saved {count} samples for Voter ID: {voter_id}")
        with col2:
            if st.button("Clear Samples"):
                st.session_state.captured_samples = []
                st.rerun()

elif mode == "Cast Vote":
    st.header("Face Recognition & Vote Casting")

    if len(names_data) == 0 or len(faces_data) == 0:
        st.warning("No registered voters found yet. Switch to **Register Voter** first.")
    else:
        n_neighbors = min(5, len(names_data))
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(faces_data, names_data)

        photo = st.camera_input("Scan Your Face to Verify", key="vote_cam")

        if photo is not None:
            features, preview_rgb = extract_face_features(photo)
            if preview_rgb is not None:
                st.image(preview_rgb, caption="Camera Preview", width=320)

            if features is None:
                st.warning("No face detected in the captured image.")
            else:
                predicted_voter = str(knn.predict(features)[0])
                st.subheader(f"Detected Voter ID: `{predicted_voter}`")

                if has_already_voted(predicted_voter):
                    st.error("YOU HAVE ALREADY VOTED.")
                else:
                    selected_party = st.selectbox("Select Party", PARTIES)
                    if st.button("Submit Vote", type="primary"):
                        record_vote(predicted_voter, selected_party)
                        st.success(
                            f"YOUR VOTE HAS BEEN RECORDED for **{selected_party}**. THANK YOU FOR PARTICIPATING IN THE ELECTIONS."
                        )

elif mode == "Vote Records":
    st.header("Recorded Votes")
    if os.path.exists(VOTES_PATH) and os.path.getsize(VOTES_PATH) > 0:
        df = pd.read_csv(VOTES_PATH)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No votes recorded yet.")
