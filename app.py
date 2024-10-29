import streamlit as st
from streamlit_lottie import st_lottie
import json
from st_pages import get_nav_from_toml, add_page_title

@st.cache_data  # Add caching decorator
def load_lottie_file():
    with open('miscellaneous/lottiefiles.json', "r") as f:
        return json.load(f)

def main():
    st.set_page_config(layout="wide")
    
    nav = get_nav_from_toml(".streamlit/pages_sections.toml")
    pg = st.navigation(nav)
    add_page_title(pg)
    pg.run()

    # Load and display lottie animation using cached function
    lottie_data = load_lottie_file()
    st_lottie(lottie_data, height=300)

if __name__ == "__main__":
    main()
