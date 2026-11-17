"""
Streamlit Tutorial: a runnable, lesson-by-lesson guide to Streamlit.

HOW TO RUN
    streamlit run streamlit_tutorial.py

    Streamlit starts a local web server and opens the page in your browser
    (usually at http://localhost:8501). Pick a lesson from the sidebar.

HOW IT WORKS (read this first)
    Streamlit runs this file from TOP TO BOTTOM every time the user clicks,
    types, or moves a widget. There are no event handlers. The page is simply
    redrawn from the new results each time. Almost every decision in this
    file follows from that single fact.

REQUIREMENTS
    pip install streamlit pandas numpy matplotlib scikit-learn
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# time: used to simulate slow work (sleep) and to measure how long things take.
import time

# matplotlib.pyplot: a plotting library. We use it in lesson 4 for full control
# over a chart. Streamlit can display matplotlib figures with st.pyplot().
import matplotlib.pyplot as plt

# numpy: fast numerical arrays and random number generation.
import numpy as np

# pandas: tables of data (DataFrames). Streamlit displays them natively.
import pandas as pd

# streamlit: the library this whole file is about. By convention it is
# imported as "st" so every call reads as st.something(...).
import streamlit as st

# scikit-learn: a machine learning library. We use a simple logistic
# regression as a stand-in for "a model you have already trained".
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
# set_page_config controls the browser tab title, icon and layout.
# RULE: it must be the FIRST Streamlit command in the script. Streamlit will
# raise an error if any other st.* call happens before it.
st.set_page_config(
    page_title="Streamlit Tutorial",   # text shown in the browser tab
    page_icon="📘",                    # emoji or image shown in the tab
    layout="wide",                     # "wide" uses the full browser width;
                                       # "centered" (the default) is narrower
)


# ---------------------------------------------------------------------------
# Helper functions shared by several lessons
# ---------------------------------------------------------------------------

# @st.cache_data tells Streamlit: "this function returns DATA (a table, a
# number, a list). If you call it again with the same arguments, return the
# saved result instead of running the function again."
# Because every rerun reruns the whole script, without this decorator the
# table would be rebuilt on every click.
@st.cache_data
def make_sample_data(n_rows=300, seed=42):
    """Create a small synthetic customer table.

    Args:
        n_rows: how many customers to create.
        seed:   a fixed number that makes the random data repeatable. The same
                seed always gives the same table, so results are predictable.

    Returns:
        A pandas DataFrame with three features and a 0/1 "churned" label.
    """
    # Create a random number generator with a fixed seed.
    # Using a generator (rather than np.random.seed) is the modern approach.
    rng = np.random.default_rng(seed)

    # Build the feature columns. Each column is a list of n_rows values.
    df = pd.DataFrame({
        # Whole months a customer has stayed, from 0 to 72 inclusive.
        "tenure": rng.integers(0, 73, n_rows),
        # Monthly bill, a decimal number between 20 and 150, rounded to cents.
        "monthly_charge": rng.uniform(20, 150, n_rows).round(2),
        # Number of support calls, drawn from a Poisson distribution (count data).
        "support_calls": rng.poisson(1.5, n_rows),
    })

    # Build a "risk score". Higher means more likely to churn.
    # The weights are chosen so that: long tenure lowers risk, high charges
    # raise risk, and more support calls raise risk. The random noise makes
    # the problem realistic, so no model is perfect.
    risk = (
        -0.08 * df["tenure"]
        + 0.02 * df["monthly_charge"]
        + 0.4 * df["support_calls"]
        + rng.normal(0, 1, n_rows)
    )

    # Turn the risk score into a 0/1 label. Customers above the median risk
    # are labelled 1 ("churned"). .astype(int) converts True/False to 1/0.
    df["churned"] = (risk > np.median(risk)).astype(int)

    return df


# @st.cache_resource is for OBJECTS you create once and reuse, such as a
# trained model or a database connection. Unlike cache_data, it does not copy
# the result each time; every rerun gets the same object back. Use it for
# things you call methods on (model.predict, conn.execute, and so on).
@st.cache_resource
def train_demo_model(df):
    """Train a small logistic regression model and report its accuracy.

    Args:
        df: the DataFrame from make_sample_data().

    Returns:
        A tuple (model, accuracy): the fitted model and its score on test data.
    """
    # The input columns the model learns from. These MUST be the same columns,
    # with the same names, that the prediction form supplies later.
    features = ["tenure", "monthly_charge", "support_calls"]

    # Split the data into a training set (to learn from) and a test set (to
    # check the model on data it has never seen). test_size=0.25 keeps 25% for
    # testing. random_state=0 makes the split the same every time.
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["churned"], test_size=0.25, random_state=0
    )

    # Create an untrained logistic regression. max_iter raises the number of
    # optimisation steps allowed, so the solver has room to converge.
    model = LogisticRegression(max_iter=1000)

    # Learn the relationship between the features and the churn label.
    model.fit(X_train, y_train)

    # .score() returns accuracy: the fraction of test predictions that are right.
    accuracy = model.score(X_test, y_test)

    # Return both, so the caller can show the accuracy as well as use the model.
    return model, accuracy


# ---------------------------------------------------------------------------
# Lessons
# Each lesson_* function draws one page. The sidebar picks which one runs.
# ---------------------------------------------------------------------------

def lesson_welcome():
    """Lesson 0: the welcome page and the one idea behind Streamlit."""
    # st.title draws the largest heading on the page.
    st.title("Streamlit Tutorial")

    # st.write is the general-purpose display call. It accepts text, numbers,
    # dataframes, dicts and charts, and formats them sensibly.
    st.write(
        "Work through the lessons in the sidebar. Each one explains an idea, "
        "then gives you working code to try."
    )

    # st.subheader draws a smaller heading.
    st.subheader("The one idea that explains everything")

    # st.info draws a blue information box. Other boxes: st.success (green),
    # st.warning (yellow), st.error (red).
    st.info(
        "Streamlit runs this whole script from top to bottom every time "
        "anything changes. Clicking a button or moving a slider reruns the file. "
        "The page is redrawn from the new results."
    )

    st.subheader("Lessons")

    # st.markdown renders Markdown: **bold**, *italic*, lists, links, and so on.
    # The triple quotes let the Markdown span several lines.
    st.markdown(
        """
1. Text and layout
2. Widgets: getting input
3. Columns, tabs and expanders
4. Showing data and charts
5. Caching: avoiding repeated work
6. Session state: remembering values
7. Forms: submit many inputs at once
8. Wrapping a machine learning model
9. Messages, spinners and progress
        """
    )


def lesson_text():
    """Lesson 1: text elements and the structure of a page."""
    st.header("1. Text and layout")

    st.write("Use these calls to build the structure of a page.")

    # st.code shows a block of code with syntax highlighting. It does NOT run
    # the code; it is only displayed. The \n inside the string makes new lines.
    st.code(
        'st.title("Big heading")\n'
        'st.header("Section heading")\n'
        'st.subheader("Smaller heading")\n'
        'st.markdown("**Bold**, *italic*, and a [link](https://streamlit.io)")',
        language="python",
    )

    # The real text elements, in order. Streamlit draws them top to bottom.
    st.title("Big heading")          # largest heading
    st.header("Section heading")     # section heading
    st.subheader("Smaller heading")  # sub-section heading
    st.markdown("**Bold**, *italic*, and a [link](https://streamlit.io).")
    st.caption("Small grey text for notes.")   # de-emphasised text
    st.write("`st.write` accepts text, numbers, tables and charts.")  # backticks = inline code

    # st.divider draws a horizontal line to separate sections.
    st.divider()
    st.write("Everything above this line was drawn in order, top to bottom.")


def lesson_widgets():
    """Lesson 2: widgets, the controls that collect input from the user."""
    st.header("2. Widgets: getting input")
    st.write(
        "A widget draws a control and returns the value the user chose. "
        "Store the return value in a variable and use it below."
    )

    # st.columns(2) splits the page into two side-by-side areas and returns
    # them as a tuple. Everything inside a "with col:" block goes into that column.
    col_a, col_b = st.columns(2)

    # Each widget below returns a value. The VARIABLE on the left keeps that
    # value. Streamlit rebuilds the variable on every rerun from the current
    # state of the control.
    with col_a:
        # Free text input. Returns a string. placeholder is grey hint text.
        name = st.text_input("Your name", placeholder="Type here")

        # Number box with limits. Returns an int or float.
        age = st.number_input("Your age", min_value=0, max_value=120, value=25)

        # Slider from 0 to 10, starting at 7. Returns an int.
        rating = st.slider("How much do you like Python?", 0, 10, 7)

        # Tick box. Returns True when ticked and False when not.
        agree = st.checkbox("I have read the terms")

    with col_b:
        # Dropdown. Returns the one item the user picked.
        language = st.selectbox("Favourite data tool", ["Python", "R", "SQL", "Excel"])

        # Multiple choice. Returns a LIST of every item picked (possibly empty).
        topics = st.multiselect("Topics you have studied", ["ML", "SQL", "Stats", "Viz"])

        # Calendar picker. Returns a datetime.date object.
        day = st.date_input("Pick a date")

        # Radio buttons. horizontal=True lays them out in a row.
        color = st.radio("Pick a colour", ["Red", "Green", "Blue"], horizontal=True)

    # Show every value the widgets returned, so you can see the data flow.
    st.subheader("What the widgets returned")
    st.write({
        "name": name,
        "age": age,
        "rating": rating,
        "agree": agree,
        "language": language,
        "topics": topics,
        "day": str(day),      # convert the date to text so it displays cleanly
        "colour": color,
    })

    st.subheader("The button trap")
    st.write(
        "`st.button` returns True only on the rerun caused by the click. "
        "Click the button, then move a slider. The message disappears, "
        "because the button is False on that rerun."
    )

    # st.button returns True ONLY during the rerun caused by this click.
    # On every other rerun it returns False, so the block below is skipped.
    # The trap: anything inside this block vanishes as soon as the user
    # interacts with something else. Lesson 6 shows how to fix that.
    if st.button("Click me"):
        # "name or 'there'" means: use the name if it is not empty, otherwise
        # use the word "there". Empty strings are "falsy" in Python.
        st.success(f"Hi {name or 'there'}, the button was clicked.")


def lesson_layout():
    """Lesson 3: columns, tabs, expanders and placeholders for arranging a page."""
    st.header("3. Columns, tabs and expanders")

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


def lesson_data():
    """Lesson 4: showing tables and charts."""
    st.header("4. Showing data and charts")

    # Get the sample table. Because make_sample_data is cached, this is fast
    # on every rerun after the first.
    df = make_sample_data()

    st.subheader("Interactive table")
    st.write("`st.dataframe` lets users sort and search. Click a column header.")

    # df.head(50) takes the first 50 rows only, to keep the page light.
    # use_container_width=True stretches the table to the full column width.
    # NOTE: newer Streamlit versions prefer width="stretch" instead; if you see
    # a deprecation warning, switch to that.
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Built-in charts")
    st.write("Pass a dataframe and Streamlit draws the chart.")

    # value_counts() counts how many customers have each number of support calls.
    # sort_index() puts the groups in order 0, 1, 2, ...
    # st.bar_chart draws one bar per group.
    st.bar_chart(df["support_calls"].value_counts().sort_index())

    st.subheader("Matplotlib for full control")

    # Create a figure (the whole picture) and one set of axes (the plotting area).
    fig, ax = plt.subplots(figsize=(7, 3.5))  # width and height in inches

    # Split the table into two groups (churned and not churned) and draw a
    # histogram for each. alpha=0.6 makes the bars slightly see-through so the
    # overlapping histograms stay readable.
    for label, group in df.groupby("churned"):
        ax.hist(group["monthly_charge"], bins=20, alpha=0.6,
                label=f"churned={label}")

    ax.set_xlabel("Monthly charge")   # label for the horizontal axis
    ax.set_ylabel("Customers")        # label for the vertical axis
    ax.legend()                       # show the key that explains the colours

    # st.pyplot displays the matplotlib figure on the page.
    st.pyplot(fig)


def lesson_caching():
    """Lesson 5: caching, to skip work that has already been done."""
    st.header("5. Caching: avoiding repeated work")

    st.write(
        "Each rerun runs the whole script. Caching stores the result of a "
        "slow function so later reruns skip the work."
    )

    st.subheader("cache_data: for data results")
    # Show the code pattern as text. st.code does not run it.
    st.code(
        "@st.cache_data\n"
        "def load_data(path):\n"
        "    return pd.read_csv(path)",
        language="python",
    )

    st.subheader("cache_resource: for objects you call methods on")
    st.code(
        "@st.cache_resource\n"
        "def load_model():\n"
        "    return joblib.load('model.joblib')",
        language="python",
    )

    st.subheader("See the difference")

    # A plain function that pretends to be slow. Each call sleeps for one second,
    # then squares the number. Nothing is saved, so the work repeats every time.
    def slow_square(x):
        time.sleep(1)  # pause for one second to simulate expensive work
        return x * x

    # The same function, but decorated. The result is saved for each input
    # value. The first call with a given x takes one second; repeats are instant.
    @st.cache_data
    def slow_square_cached(x):
        time.sleep(1)
        return x * x

    # A slider that gives a number from 1 to 10. Moving it reruns the script.
    value = st.slider("Pick a number", 1, 10, 3)

    # Time the uncached call. time.time() returns the current time in seconds,
    # so the difference between two readings is how long the call took.
    start = time.time()
    result = slow_square(value)
    uncached_time = time.time() - start

    # Time the cached call the same way.
    start = time.time()
    cached_result = slow_square_cached(value)
    cached_time = time.time() - start

    # :.2f formats the number with two decimal places.
    st.write(f"Uncached: {result} in {uncached_time:.2f} s")
    st.write(f"Cached: {cached_result} in {cached_time:.2f} s")

    st.caption("Move the slider to a new value, then back. The cached call is fast on repeats.")


def lesson_session_state():
    """Lesson 6: session state, which remembers values across reruns."""
    st.header("6. Session state: remembering values")

    st.write(
        "Normal variables reset on every rerun. `st.session_state` is a "
        "dictionary that survives reruns for one browser session."
    )

    # st.session_state works like a dictionary that belongs to this browser tab.
    # It is EMPTY on the first run, so we set a starting value once. The check
    # "not in" makes sure we do not reset it on later reruns.
    if "count" not in st.session_state:
        st.session_state.count = 0

    col_a, col_b = st.columns(2)
    with col_a:
        # When clicked, add one to the saved count. The new value is stored in
        # session state, so it is still there on the next rerun.
        if st.button("Add one"):
            st.session_state.count += 1

        # Reset the saved count back to zero.
        if st.button("Reset"):
            st.session_state.count = 0
    with col_b:
        # Show the saved count. Because it lives in session state, the number
        # is correct even though the script reran from the top.
        st.metric("Count", st.session_state.count)

    st.subheader("Keeping a result on screen")

    # This is the fix for the button trap from lesson 2. The button is True only
    # on the click rerun, so we save the result into session state at that moment.
    if st.button("Run analysis"):
        st.session_state.analysis = "Analysis complete: 42 rows processed."

    # On every later rerun, the saved result is still in session state, so we can
    # keep showing it even though the button is now False.
    if "analysis" in st.session_state:
        st.success(st.session_state.analysis)

    st.caption("The result stays on screen after you change other widgets.")


def lesson_forms():
    """Lesson 7: forms, which batch several inputs into one submit."""
    st.header("7. Forms: submit many inputs at once")

    st.write(
        "Inside a form, changing an input does not rerun the app. "
        "Only the submit button does."
    )

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


def lesson_model():
    """Lesson 8: wrapping a trained model in an interactive prediction form."""
    st.header("8. Wrapping a machine learning model")

    st.write(
        "This lesson trains a small model on synthetic data the first time it "
        "loads, then wraps it in a prediction form. The same pattern works for "
        "any model you have saved with joblib."
    )

    # Build the data (cached) and train the model (cached as a resource).
    # The first load does the work. Later reruns reuse the saved model, which is
    # the whole reason to use cache_resource for models.
    df = make_sample_data()
    model, accuracy = train_demo_model(df)

    # Show how well the model did on the held-out test data.
    st.caption(f"Model trained once. Test accuracy: {accuracy:.2f}")

    # The inputs go in the SIDEBAR so the page stays clean. The variable names
    # and order must match the columns the model was trained on.
    st.sidebar.subheader("Customer details")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly = st.sidebar.number_input("Monthly charge", 0.0, 200.0, 70.0)
    calls = st.sidebar.slider("Support calls", 0, 10, 1)

    # The model expects a table, not loose numbers. Wrap the three inputs in a
    # one-row DataFrame. The column names MUST match the training features exactly.
    row = pd.DataFrame([{
        "tenure": tenure,
        "monthly_charge": monthly,
        "support_calls": calls,
    }])

    # Only predict when the user clicks the button, not on every slider move.
    if st.button("Predict churn"):
        # model.predict returns an array of labels, one per row. We have one row,
        # so [0] takes the first (and only) label: 0 or 1.
        pred = model.predict(row)[0]

        # model.predict_proba returns the probability of each class for each row.
        # [0] takes the first row, and [1] takes the probability of class 1 (churn).
        proba = model.predict_proba(row)[0][1]

        st.metric("Prediction", "Will churn" if pred == 1 else "Will stay")

        # st.progress draws a bar from 0.0 to 1.0. text= adds a label. :.0% formats
        # the decimal as a percentage with no decimal places (0.73 becomes 73%).
        st.progress(float(proba), text=f"Churn probability: {proba:.0%}")

    # An expander that shows the key code, so students can see how it is built.
    with st.expander("Show the code for this lesson"):
        st.code(
            "@st.cache_resource\n"
            "def train_demo_model(df):\n"
            "    model = LogisticRegression(max_iter=1000)\n"
            "    model.fit(X_train, y_train)\n"
            "    return model, accuracy\n",
            language="python",
        )


def lesson_messages():
    """Lesson 9: status messages, spinners, toasts and progress bars."""
    st.header("9. Messages, spinners and progress")

    # Four message boxes, each with its own colour and meaning.
    st.success("Model loaded.")
    st.info("Predictions are estimates.")
    st.warning("Some values were missing and were filled in.")
    st.error("File not found.")

    st.subheader("Spinner")
    if st.button("Run a slow task"):
        # st.spinner shows a loading animation while the code inside runs.
        with st.spinner("Working..."):
            time.sleep(2)  # simulate a slow job

        # st.toast shows a short pop-up notification in the corner.
        st.toast("Finished!")

    st.subheader("Progress bar")
    if st.button("Start progress"):
        # Create the bar at 0 with a starting label. We keep a reference to it
        # so we can update it in the loop below.
        bar = st.progress(0, text="Starting...")

        # range(1, 101, 10) gives 1, 11, 21, ..., 91. Each step updates the bar.
        for i in range(1, 101, 10):
            time.sleep(0.1)  # pause so the change is visible to the user
            bar.progress(i, text=f"{i}% done")

        # Finish at exactly 100 so the bar is full.
        bar.progress(100, text="Complete")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
# A dictionary mapping each label shown in the sidebar to the function that
# draws that lesson. The order here is the order shown in the menu.
LESSONS = {
    "Welcome": lesson_welcome,
    "1. Text and layout": lesson_text,
    "2. Widgets": lesson_widgets,
    "3. Columns, tabs, expanders": lesson_layout,
    "4. Data and charts": lesson_data,
    "5. Caching": lesson_caching,
    "6. Session state": lesson_session_state,
    "7. Forms": lesson_forms,
    "8. Wrapping a model": lesson_model,
    "9. Messages and progress": lesson_messages,
}

# A title for the sidebar section.
st.sidebar.title("Lessons")

# A radio menu in the sidebar. list(LESSONS.keys()) gives the labels in order.
# The chosen label is returned as a string. label_visibility="collapsed" hides
# the "Go to" label so the menu looks cleaner.
choice = st.sidebar.radio("Go to", list(LESSONS.keys()), label_visibility="collapsed")

# Look up the function for the chosen label and call it. This draws one page.
# Only the selected lesson runs, so the other lessons cost nothing.
LESSONS[choice]()
