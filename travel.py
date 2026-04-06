import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="🌍 AI Travel App", layout="wide")
st.title("🌍 AI Travel Recommendation System")

# 🌍 Full Dataset
places = [
    ["Paris","France","High","City","https://upload.wikimedia.org/wikipedia/commons/e/e6/Paris_Night.jpg",
     ["Versailles","Lyon"],["Hotel Paris Centre"],["Eiffel Tower","Louvre"]],
    
    ["Dubai","UAE","High","City","https://upload.wikimedia.org/wikipedia/commons/9/93/Dubai_skyline.jpg",
     ["Abu Dhabi"],["Burj Hotel"],["Burj Khalifa","Desert Safari"]],
    
    ["Tokyo","Japan","Medium","City","https://upload.wikimedia.org/wikipedia/commons/1/1e/Tokyo_Tower.jpg",
     ["Kyoto"],["Tokyo Inn"],["Tokyo Tower"]],
    
    ["New York","USA","High","City","https://upload.wikimedia.org/wikipedia/commons/a/a1/Manhattan_skyline.jpg",
     ["Boston"],["NY Grand"],["Statue of Liberty"]],
    
    ["London","UK","High","City","https://upload.wikimedia.org/wikipedia/commons/c/cd/London_Montage.jpg",
     ["Oxford"],["Royal Inn"],["Big Ben"]],
    
    ["Rome","Italy","Medium","City","https://upload.wikimedia.org/wikipedia/commons/5/5c/Colosseum.jpg",
     ["Florence"],["Rome Palace"],["Colosseum"]],
    
    ["Sydney","Australia","High","City","https://upload.wikimedia.org/wikipedia/commons/4/4e/Sydney_Opera_House.jpg",
     ["Melbourne"],["Harbour Hotel"],["Opera House"]],
    
    ["Singapore","Singapore","Medium","City","https://upload.wikimedia.org/wikipedia/commons/9/9e/Singapore_Skyline.jpg",
     ["Sentosa"],["Marina Hotel"],["Gardens by Bay"]],
    
    ["Bangkok","Thailand","Low","City","https://upload.wikimedia.org/wikipedia/commons/6/6d/Bangkok_skytrain.jpg",
     ["Pattaya"],["Bangkok Inn"],["Grand Palace"]],
    
    ["Bali","Indonesia","Medium","Beach","https://upload.wikimedia.org/wikipedia/commons/0/0c/Bali_beach.jpg",
     ["Lombok"],["Beach Villa"],["Ubud"]],
    
    ["Goa","India","Medium","Beach","https://upload.wikimedia.org/wikipedia/commons/0/0f/Goa_beach.jpg",
     ["Gokarna"],["Goa Stay"],["Baga Beach"]],
    
    ["Kerala","India","Medium","Beach","https://upload.wikimedia.org/wikipedia/commons/6/6e/Kerala_backwaters.jpg",
     ["Munnar"],["Backwater Resort"],["Backwaters"]],
    
    ["Manali","India","Low","Hill","https://upload.wikimedia.org/wikipedia/commons/9/9e/Manali.jpg",
     ["Shimla"],["Hill View"],["Solang Valley"]],
    
    ["Ooty","India","Low","Hill","https://upload.wikimedia.org/wikipedia/commons/3/3e/Ooty.jpg",
     ["Coimbatore"],["Ooty Inn"],["Tea Gardens"]],
    
    ["Cape Town","South Africa","Medium","City","https://upload.wikimedia.org/wikipedia/commons/a/ae/Cape_Town.jpg",
     ["Durban"],["Ocean View"],["Table Mountain"]],
]

df = pd.DataFrame(places, columns=[
    "Place","Country","Budget","Category","Image","Nearby","Hotels","Attractions"
])

# 🔢 Convert to numeric for ML
budget_map = {"Low":0,"Medium":1,"High":2}
category_map = {"Beach":0,"Hill":1,"City":2}

df["B"] = df["Budget"].map(budget_map)
df["C"] = df["Category"].map(category_map)

X = df[["B","C"]]

# 🤖 Train KNN Model
model = NearestNeighbors(n_neighbors=1)
model.fit(X)

# 🎯 User Input
col1, col2 = st.columns(2)
with col1:
    budget = st.selectbox("Select Budget", ["Low","Medium","High"])
with col2:
    category = st.selectbox("Select Category", ["Beach","Hill","City"])

user_input = [[budget_map[budget], category_map[category]]]

# 🔍 Recommend
if st.button("Get Recommendation"):
    _, index = model.kneighbors(user_input)
    place = df.iloc[index[0][0]]

    st.header(f"👉 Visit {place['Place']} ({place['Country']})")

    # 🖼️ Image
    st.image(place["Image"], use_column_width=True)

    # 💰 Budget
    st.write(f"💰 Budget: {place['Budget']}")

    # ⭐ Attractions
    st.subheader("⭐ Attractions")
    for a in place["Attractions"]:
        st.write("•", a)

    # 📍 Nearby
    st.subheader("📍 Nearby Places")
    for n in place["Nearby"]:
        st.write("•", n)

    # 🏨 Hotels
    st.subheader("🏨 Hotels")
    for h in place["Hotels"]:
        st.write("•", h)

    # 📍 Google Maps
    maps_url = f"https://www.google.com/maps/search/{place['Place']}"
    st.markdown(f"[📍 View on Google Maps]({maps_url})") 
st.markdown("""
<style>

/* 🔥 Apply background to full app */
[data-testid="stAppViewContainer"] {
    background: 
        linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.7)),
        url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
    
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* 🧊 Main container transparent */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stSidebar"],
[data-testid="stDecoration"] {
    background: transparent !important;
}

/* 🧊 Glass Cards */
.card {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(15px);
    padding: 20px;
    border-radius: 15px;
    color: white;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
    transition: 0.3s;
}

.card:hover {
    transform: scale(1.03);
}

/* 🎯 Sidebar Glass */
section[data-testid="stSidebar"] > div {
    background: rgba(0,0,0,0.5);
    backdrop-filter: blur(10px);
}

/* 📝 Text color fix */
h1, h2, h3, h4, h5, h6, p, label, span {
    color: white !important;
}

/* 🔘 Button */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
}

</style>
""", unsafe_allow_html=True)