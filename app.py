import streamlit as st
from lessons import lesson_1, lesson_2  # Import other lesson files as needed

st.set_page_config(page_title="Raspberry PI AI", page_icon="images/logo.png")

st.title("Raspberry PI AI Course")
st.sidebar.title("Select a Lesson")

lesson_selected = st.sidebar.radio("Lessons",
                                   ["Lesson 1: TFLite Model Image Classification",
                                    "Lesson 2: EfficientNet Image Classification",
                                    "Lesson 3"])

if lesson_selected == "Lesson 1: TFLite Model Image Classification":
    lesson_1.lesson_1()
elif lesson_selected == "Lesson 2: EfficientNet Image Classification":
    lesson_2.lesson_2()
elif lesson_selected == "Lesson 3":
    st.subheader("Lesson 3")
    st.write("This is content for Lesson 3.")
