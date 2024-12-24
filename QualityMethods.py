import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import streamlit as st


def display_data_info(df):
    """Displays the DataFrame's info in the Streamlit app."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

def describe_data(df):
    """Generates descriptive statistics for the DataFrame."""
    return df.describe()

def data_types_analysis(df):
    """Displays data type information and allows conversion."""
    # Display data types of columns
    df_original = df.copy()
    st.header("Data Types Analysis")
    st.write("### Data Types of Columns:")
    st.write(df.dtypes)

    # Display the number of rows before handling
    rows_before = df.shape[0]
    st.write(f"**Number of rows before handling data type conversion:** {rows_before}")

    # Flag for data type conversion success
    is_conversion_successful = False

    # Data type conversion interface
    st.subheader("Convert Data Types:")
    if 'data_type_analysis_clicked' in st.session_state and st.session_state['data_type_analysis_clicked']:
        selected_column = st.selectbox("Select a column to convert", df.columns, key="convert_col")
        new_type = st.selectbox("Select the new data type", ["int", "float", "str", "datetime"], key="new_type")
        convert_button = st.button("Convert Data Type", key='convert_btn')

        if convert_button:
            try:
                if new_type == "datetime":
                    df[selected_column] = pd.to_datetime(df[selected_column], errors='coerce')
                else:
                    df[selected_column] = df[selected_column].astype(new_type)

                # Store the updated dataframe in session state
                st.session_state['data'] = df
                st.session_state['type_converted'] = True  # Flag to indicate conversion was successful
                reset_all_flags()  # Reset all flags
                st.success(f"Column '{selected_column}' converted to {new_type} successfully!")
                is_conversion_successful = True

            except Exception as e:
                st.warning(f"Error converting column '{selected_column}': {e}")

    # If conversion was successful, display the number of rows after conversion
    if is_conversion_successful:
        rows_after = df.shape[0]
        st.write(f"**Number of rows after handling data type conversion:** {rows_after}")

        # Show the original and updated DataFrames side by side
        st.write("### DataFrames Before and After Data Type Conversion:")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Before Conversion:**")
            st.dataframe(df_original)

        with col2:
            st.write("**After Conversion:**")
            st.dataframe(df)

    return df


def visualize_data(df, column):
    """Generates visualizations (Histogram with KDE and Box Plot) for the selected column."""
    if column not in df.columns:
        st.warning(f"Column '{column}' not found in the dataset.")
        return None, None

    # Histogram with KDE
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[column], ax=ax, kde=True)
    ax.set_title(f"Histogram of {column} with KDE")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    # Box Plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df[column], ax=ax2)
    ax2.set_title(f"Box Plot of {column}")

    return fig, fig2

def visualize_categorical_data(df, column):
    """Generates visualizations (Count Plot and Pie Chart) for the selected categorical column."""
    if column not in df.columns:
        st.warning(f"Column '{column}' not found in the dataset.")
        return None, None

    # Count Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=df[column], ax=ax)
    ax.set_title(f"Count Plot of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")

    # Pie Chart
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    category_counts = df[column].value_counts()
    ax2.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f"Pie Chart of {column}")
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    return fig, fig2


def correlation_matrix(df):
    """Generates a correlation matrix for the DataFrame."""
    # Filter only numeric columns (both int and float types)
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])
    
    if numeric_cols.empty:
        st.warning("No numeric columns found for correlation analysis.")
        return None
    
    # Create the correlation matrix plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_cols.corr(), annot=True, ax=ax, cmap='coolwarm')
    plt.title("Correlation Matrix (Numeric Columns)")

    return fig

def missing_value_analysis(df):
    """Displays the number of missing values per column and a heatmap of missing values."""
    
    # Calculate missing values per column
    missing_values = df.isnull().sum()
    st.write("### Missing Values per Column:")
    st.write(missing_values)
    
    # Update the DataFrame in session state
    st.session_state['data'] = df
    
    # Plotting the heatmap
    st.subheader("Missing Values Heatmap:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=True, ax=ax)
    plt.title("Missing Values Heatmap")
    st.pyplot(fig)
    
    return df


def handle_missing_values(df, method="mean", column=None):
    """Handles missing values based on the selected method and column."""
    # Create a copy of the original DataFrame for display
    df_original = df.copy()

    # Calculate and display the number of rows before handling
    rows_before = df.shape[0]
    st.write(f"**Number of rows before handling missing values:** {rows_before}")

    # Determine if the specified column is categorical or numeric
    if column:
        if column not in df.columns:
            st.warning(f"Column '{column}' does not exist in the DataFrame.")
            return df
        is_numeric = pd.api.types.is_numeric_dtype(df[column])
    else:
        is_numeric = False

    # Flag to indicate whether processing is successful
    is_handling_successful = False

    # Handle missing values based on the specified method
    if method == "mean":
        if column:
            if not is_numeric:
                st.warning(f"Cannot use mean to fill missing values in non-numeric column '{column}'.")
            else:
                df[column].fillna(df[column].mean(), inplace=True)
                is_handling_successful = True
        else:
            numeric_cols = df.select_dtypes(include=['number'])
            if numeric_cols.empty:
                st.warning("Cannot use mean to fill missing values in non-numeric columns.")
            else:
                df.fillna(df.mean(), inplace=True)
                is_handling_successful = True

    elif method == "median":
        if column:
            if not is_numeric:
                st.warning(f"Cannot use median to fill missing values in non-numeric column '{column}'.")
            else:
                df[column].fillna(df[column].median(), inplace=True)
                is_handling_successful = True
        else:
            numeric_cols = df.select_dtypes(include=['number'])
            if numeric_cols.empty:
                st.warning("Cannot use median to fill missing values in non-numeric columns.")
            else:
                df.fillna(df.median(), inplace=True)
                is_handling_successful = True

    elif method == "mode":
        if column:
            if df[column].isnull().all():
                st.warning(f"Cannot compute mode for column '{column}' with all values missing.")
            else:
                df[column].fillna(df[column].mode()[0], inplace=True)
                is_handling_successful = True
        else:
            if df.isnull().all().all():
                st.warning("Cannot compute mode for columns with all values missing.")
            else:
                df.fillna(df.mode().iloc[0], inplace=True)
                is_handling_successful = True

    elif method == "drop":
        if column:
            df.dropna(subset=[column], inplace=True)
            is_handling_successful = True
        else:
            df.dropna(inplace=True)
            is_handling_successful = True
    else:
        st.error("Invalid method for handling missing values.")
        return df

    if is_handling_successful:
        # Calculate and display the number of rows after handling
        rows_after = df.shape[0]
        st.write(f"**Number of rows after handling missing values:** {rows_after}")

        # Update the session state with the modified DataFrame
        st.session_state.data = df
        st.success(f"Missing values handled using '{method}' method.")

        # Display the original and updated DataFrames side by side
        st.write("### DataFrames Before and After Handling Missing Values:")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Before:**")
            st.dataframe(df_original)

        with col2:
            st.write("**After:**")
            st.dataframe(df)

    return df


def handle_duplicates(df):
    """Handles duplicate rows in the DataFrame directly without requiring a button."""
    # Create a copy of the original DataFrame for display purposes
    df_original = df.copy()

    # Calculate the number of duplicate rows
    num_duplicates = df.duplicated().sum()
    st.write(f"**Number of duplicate rows:** {num_duplicates}")

    # Display the original DataFrame
    st.write("### Original DataFrame:")
    st.dataframe(df_original)

    if num_duplicates > 0:
        # Remove duplicate rows
        df_cleaned = df.drop_duplicates()

        # Update the session state with the cleaned DataFrame
        st.session_state['data'] = df_cleaned

        # Display success message
        st.success(f"Duplicate rows removed. The dataset now has {df_cleaned.shape[0]} rows.")

        # Display the original and updated DataFrames side by side
        st.write("### DataFrames Before and After Removing Duplicates:")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Before:**")
            st.dataframe(df_original)

        with col2:
            st.write("**After:**")
            st.dataframe(df_cleaned)
    else:
        st.info("No duplicate rows found.")

    # Return the cleaned DataFrame
    return st.session_state['data']



def rename_columns(df, column_to_rename, new_column_name):
    """Renames a specified column in the DataFrame."""
    df_original = df.copy()

    # Check if the column exists
    if column_to_rename in df.columns:
        # Rename the column
        df.rename(columns={column_to_rename: new_column_name}, inplace=True)
        st.session_state['data'] = df
        st.success(f"Column '{column_to_rename}' has been renamed to '{new_column_name}'.")

    else:
        st.warning(f"Column '{column_to_rename}' does not exist in the DataFrame.")

    return df

def drop_column(df, column_to_drop):
    """Drops a specified column from the DataFrame."""
    # Create a copy of the original DataFrame for display purposes
    df_original = df.copy()

    # Check if the column exists
    if column_to_drop in df.columns:
        # Drop the column
        df = df.drop(columns=[column_to_drop])
        st.session_state['data'] = df  # Update the DataFrame in session state
        st.success(f"Column '{column_to_drop}' has been dropped.")
    else:
        st.warning(f"Column '{column_to_drop}' does not exist in the DataFrame.")

    return df


def outlier_analysis(df, column):
    """Identifies and displays outliers using the IQR method, and allows for dropping outliers."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    st.write(f"Number of outliers in {column}: {len(outliers)}")
    
    return lower_bound, upper_bound



def handle_outliers(df, column, lower_bound, upper_bound, method='clip'):
    """Handles outliers in the specified column based on the selected method."""
    df_copy = df.copy()

    # Handle outliers based on the selected method
    if method == 'clip':
        df_copy[column] = df_copy[column].clip(lower=lower_bound, upper=upper_bound)
        st.success(f"Outliers in '{column}' have been clipped to the defined bounds.")
    elif method == 'drop':
        df_copy = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)]
        st.success(f"Outliers in '{column}' have been removed.")
    else:
        st.error("Invalid method for handling outliers.")
        return df

    # Update the DataFrame in session state
    st.session_state['data'] = df_copy

    return df_copy

def download_dataset(df):
    """Downloads the DataFrame as a CSV file."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="downloaded_data.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)



def ok_button(functionality_flags=None):
    """Creates an 'OK' button that deactivates functionality when clicked."""
    right_column, _ = st.columns([5, 1])  # Create two columns, one for the right sidebar
    with right_column:
        if st.button("OK", key=f'ok_btn_{functionality_flags}'):
            deactivate_functionality(functionality_flags)

def deactivate_functionality(flags_to_deactivate=None):
    """Deactivates the specified functionality by resetting flags to False."""
    if flags_to_deactivate is None:
        # If no flags are specified, reset all session state flags
        reset_all_flags()
    else:
        # Reset only the specified flags
        for flag in flags_to_deactivate:
            if flag in st.session_state:
                st.session_state[flag] = False

def reset_all_flags():
    """Resets all session state flags."""
    keys_to_reset = [
        'show_data', 'describe_data', 'missing_analysis_run',
        'missing_values_handled', 'duplicates_handled',  # Keep this to reset duplicates_handled flag
        'outlier_analysis_run', 'outliers_handled',
        'visualize_data_run', 'correlation_run',
        'rename_column_trigger',  # Reset flag for renaming columns
        'drop_column_trigger'   # Reset flag for dropping columns
    ]
    for key in keys_to_reset:
        if key in st.session_state:  # Ensure that the key exists before resetting
            st.session_state[key] = False
# import langchain_community

# from langchain.chat_models import ChatOpenAI
# from langchain.schema import SystemMessage
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import ConversationChain
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.document_loaders import TextLoader
# from langchain.llms import OpenAI
# from langchain.agents import initialize_agent, Tool, AgentType
# import os
# from google.cloud import language_v1
# from langchain.ollama import ChatOllama

# Load a small knowledge base or documents (optional)
# def load_documents_from_directory(directory_path):
#     loader = TextLoader(directory_path)
#     documents = loader.load()
#     return documents

# # Initialize the RAG model
# def initialize_rag_model():
#     # Load documents (could be a directory, or load some example text files here)
#     documents = load_documents_from_directory("path_to_your_documents")  # Adjust path to documents
    
#     # Create embeddings
#     embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
#     vector_store = FAISS.from_documents(documents, embeddings)
    
#     # Initialize the chat model
#     chat_model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv('OPENAI_API_KEY'))
    
#     # Initialize conversation chain with the retriever (RAG setup)
#     conversation_chain = ConversationChain(llm=chat_model, retriever=vector_store.as_retriever())
    
#     return conversation_chain

# # Create a chat input system using RAG
# def rag_chat():
#     st.header("Chat with RAG (Retrieval-Augmented Generation)")
    
#     conversation_chain = initialize_rag_model()
    
#     if 'messages' not in st.session_state:
#         st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]
    
#     # Display previous conversation
#     for message in st.session_state.messages:
#         st.chat_message(message["role"]).write(message["content"])
    
#     # User input
#     user_input = st.chat_input("Ask something:")
#     if user_input:
#         # Append user message
#         st.session_state.messages.append({"role": "user", "content": user_input})
        
#         # Get response from the RAG model
#         response = conversation_chain.run(input=user_input)
        
#         # Append assistant's response
#         st.session_state.messages.append({"role": "assistant", "content": response})
        
#         # Display assistant response
#         st.chat_message("assistant").write(response)


# # Function to initialize session states if not already set
# def initialize_session():
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = [{"role": "assistant", "content": "Hi there! How can I assist you today?"}]
#     if "history" not in st.session_state:
#         st.session_state["history"] = [{"system": "You are a helpful assistant that answers user's questions"}]

# # Function to handle message input and response generation
# def handle_message_input():
#     if prompt := st.chat_input():
#         st.session_state["messages"].append({"role": "user", "content": prompt})
#         st.session_state["history"].append(("human", prompt))
        
#         # Initialize the Llama model for generating responses
#         llm = ChatOllama(model="llama3.1", temperature=0)
        
#         # Get the assistant's response by streaming
#         with st.chat_message("assistant"):
#             stream = llm.stream(st.session_state["history"])
#             response = st.write_stream(stream)
            
#             # Add response to session state for message history
#             st.session_state["messages"].append({"role": "assistant", "content": response})
#             st.session_state["history"].append(("assistant", response))

# # Function to display conversation history
# def display_messages():
#     for msg in st.session_state["messages"]:
#         st.chat_message(msg["role"]).write(msg["content"])


# def reset_flags(except_flags=None):
#     """Resets session state flags, except for the ones listed in except_flags."""
#     keys_to_reset = [
#         'show_data', 'describe_data', 'missing_analysis_run',
#         'missing_values_handled', 'duplicates_handled',  # Keep this to reset duplicates_handled flag
#         'outlier_analysis_run', 'outliers_handled',
#         'visualize_data_run', 'correlation_run'
#     ]
#     for key in keys_to_reset:
#         if key in st.session_state and (except_flags is None or key not in except_flags):
#             st.session_state[key] = False.

#RAAAAAAAAAAAAAAAAAAG

# import requests

# import subprocess
# import json
# import os
# import streamlit as st

# def query_ollama(ollama_path, prompt):
#     # Ensure the ollama_path is valid
#     if not os.path.exists(ollama_path):
#         st.error(f"Error: The provided Ollama executable path does not exist: {ollama_path}")
#         return None
    
#     # Prepare the command to run the Ollama executable
#     command = [ollama_path, 'chat', '--model', 'llama3.2', '--prompt', prompt]
    
#     try:
#         # Run the Ollama process and capture the output
#         result = subprocess.run(command, capture_output=True, text=True, check=True)
        
#         # Process the output (assuming Ollama returns a JSON string)
#         response_data = json.loads(result.stdout)
        
#         # Extract and return the content from the response
#         return response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
    
#     except subprocess.CalledProcessError as e:
#         st.error(f"Error running Ollama: {e}")
#         return None
#     except json.JSONDecodeError as e:
#         st.error(f"Error decoding Ollama response: {e}")
#         return None
#     except Exception as e:
#         st.error(f"Unexpected error: {e}")
#         return None
