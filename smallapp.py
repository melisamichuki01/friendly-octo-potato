import streamlit as st
import time

st.set_page_config(
    page_title="Small App",   # text shown in the browser tab
    page_icon="📘",                    # emoji or image shown in the tab
    layout="centered",                     # "wide" uses the full browser width;
                                       # "centered" (the default) is narrower
)

st.title("Big heading")
st.header("Section heading")
st.subheader("Smaller heading")
st.markdown("**Bold**, *italic*, and a [docs](https://streamlit.io)")
st.divider()
st.write("This is a small app that does nothing. It is used to test the Streamlit app runner.")

# Free text input. Returns a string. placeholder is grey hint text.
name = st.text_input("Your name", placeholder="Type here")

# Number box with limits. Returns an int or float.
age = st.number_input("Your age", min_value=0, max_value=120, value=25)

# Slider from 0 to 10, starting at 7. Returns an int.
rating = st.slider("How much do you like Python?", 0, 10, 7)

# Tick box. Returns True when ticked and False when not.
agree = st.checkbox("I have read the terms")

# Dropdown. Returns the one item the user picked.
language = st.selectbox("Favourite data tool", ["","Python", "R", "SQL", "Excel"])

# Multiple choice. Returns a LIST of every item picked (possibly empty).
topics = st.multiselect("Topics you have studied", ["ML", "SQL", "Stats", "Viz"])

# Calendar picker. Returns a datetime.date object.
day = st.date_input("Pick a date")

# Radio buttons. horizontal=True lays them out in a row.
color = st.radio("Pick a colour", ["Red", "Green", "Blue"], horizontal=False)

if st.button("Click me"):
        # "name or 'there'" means: use the name if it is not empty, otherwise
        # use the word "there". Empty strings are "falsy" in Python.
        st.success(f"Hi {name or 'there'}, the button was clicked.")
        st.balloons()  # Fun! Balloons fly up the screen.

st.info(f"Name: {name}, Age: {age}, Rating: {rating}, Agree: {agree}, Language: {language}, Topics: {topics}, Day: {day}, Color: {color}")
st.error("This is an error message. It is red and has a cross icon.")
st.warning("This is a warning message. It is yellow and has an exclamation icon.")
st.exception("This is an exception message. It is red and has a cross icon. It also shows a traceback.")

st.progress(0.5)  # 50% progress bar
st.snow()  # Fun! Snow falls down the screen.

st.subheader("Columns")
# Three equal columns side by side.
col1, col2, col3 = st.columns(3)

# st.metric shows a big number. Arguments: label, value, and an optional
# change (delta). A positive delta shows green, a negative one shows red.
with col1:
    st.metric("Accuracy", "0.91", "0.02")
with col2:
    st.metric("Precision", "0.87", "-0.01")
with col3:
    st.metric("Recall", "0.84", "0.03")

st.subheader("Tabs")
# st.tabs creates clickable tabs. Each "with" block fills one tab.
tab_one, tab_two = st.tabs(["Overview", "Details"])
with tab_one:
    st.write("Put the summary here.")
with tab_two:
    st.write("Put the technical detail here.")

st.subheader("Expander")
# st.expander is a collapsible box. Hidden until the user clicks it open.
with st.expander("Show how this works"):
    st.write("Content in an expander stays hidden until the user opens it.")

st.subheader("Empty placeholder")
# st.empty() reserves a spot on the page that you can overwrite later.
slot = st.empty()
slot.write("Waiting for a result...")
# time.sleep pauses the script for 0.5 seconds, simulating slow work.
time.sleep(0.5)
# Writing to the same slot replaces the earlier content in place.
slot.write("Result ready. The placeholder was replaced in place.")

# st.form groups the widgets inside it. Changing any of them does nothing
# until the user presses the submit button. The string "profile_form" is
# a unique name for this form on the page.
with st.form("profile_form"):
    name = st.text_input("Name")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    plan = st.selectbox("Plan", ["Basic", "Standard", "Premium"])

    # The only widget that triggers a rerun. It returns True on the click
    # rerun and False otherwise, just like st.button.
    submitted = st.form_submit_button("Save")

    # Use the submitted flag to decide what to show.
    if submitted:
        st.success(f"Saved {name or 'profile'}: {tenure} months on {plan}.")
    else:
        st.write("Fill in the form and press Save.")

