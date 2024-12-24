import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import StringIO
from QualityMethod import *


def reset_flags(except_flags=None):
    """Resets all flags except the specified ones."""
    all_flags = [
        'show_data', 'info_data', 'describe_data', 'missing_analysis_run',
        'missing_values_handled', 'data_type_analysis_clicked', 'duplicates_handled',
        'outlier_analysis_run', 'visualize_data_triggered', 'visualize_categorical_data_triggered',
        'correlation_matrix_run', 'rename_column_trigger', 'download_trigger','type_converted' 
    ]
    if except_flags is None:
        except_flags = []

    for flag in all_flags:
        if flag not in except_flags:
            if flag in st.session_state:
                del st.session_state[flag]

def main():
    
    st.set_page_config(layout="wide")
    st.sidebar.title("Data Quality Analysis")
    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"], key='file_uploader')

    if uploaded_file is not None:
        if 'data' not in st.session_state:
            try:
                if uploaded_file.name.endswith(".csv"):
                    csv_file = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    df = pd.read_csv(csv_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file)
                st.session_state['data'] = df
                st.sidebar.success("Dataset uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
        else:
            df = st.session_state['data'].copy()

        # Button to show data
        if st.sidebar.button("Show Data", key='show_data_btn'):
            reset_flags(except_flags=['show_data'])  # Reset all except 'show_data' flag
            st.session_state['show_data'] = True

        if 'show_data' in st.session_state and st.session_state['show_data']:
            st.header("Data")
            st.write(df)
            st.session_state['show_data'] = False  # Reset the flag after displaying data
            ok_button(['show_data'])  # OK button to deactivate Show Data functionality

        # Button to show info data
        if st.sidebar.button("Info Data", key='info_data_btn'):
            reset_flags(except_flags=['info_data'])  # Reset all except 'info_data' flag
            st.session_state['info_data'] = True

        if 'info_data' in st.session_state and st.session_state['info_data']:
            st.header("Data Info")
            display_data_info(df)
            st.session_state['info_data'] = False  # Reset the flag after displaying info
            ok_button(['info_data'])  # OK button to deactivate Info Data functionality

        # Button to describe data
        if st.sidebar.button("Describe Data", key='describe_data_btn'):
            reset_flags(except_flags=['describe_data'])  # Reset all except 'describe_data' flag
            st.session_state['describe_data'] = True

        if 'describe_data' in st.session_state and st.session_state['describe_data']:
            st.header("Data Description")
            st.table(describe_data(df))
            st.session_state['describe_data'] = False  # Reset the flag after describing data
            ok_button(['describe_data'])  # OK button to deactivate Describe Data functionality


        # Sidebar button to trigger data type analysis
        if st.sidebar.button("Data Type Analysis", key='data_type_btn'):
            reset_flags(except_flags=['data_type_analysis_clicked'])  # Reset all except 'data_type_analysis_clicked' flag
            st.session_state['data_type_analysis_clicked'] = True  # Set flag to trigger the data type analysis
            
        # Check if data type analysis was triggered and display the analysis
        if 'data_type_analysis_clicked' in st.session_state and st.session_state['data_type_analysis_clicked']:
            if 'data' in st.session_state and not st.session_state['data'].empty:
                df = st.session_state['data']  # Get the DataFrame from session state
                df = data_types_analysis(df)  # Call the function to display the analysis and handle conversion
            else:
                st.write("Please upload or load a dataset to start the analysis.")

        # Display the dataframe after conversion if successful
        if 'type_converted' in st.session_state and st.session_state['type_converted']:
            reset_flags(except_flags=['type_converted'])  # Reset all except 'data_type_analysis_clicked' flag
            st.session_state['type_converted'] = True 
           # st.write(st.session_state['data'])  # Display the converted dataframe
            ok_button(['data_type_analysis_clicked', 'type_converted'])

                        # Button to perform missing value analysis
        if st.sidebar.button("Missing Value Analysis", key='missing_val_btn'):
            reset_flags(except_flags=['missing_analysis_run'])  # Reset all except 'missing_analysis_run' flag
            st.session_state['missing_analysis_run'] = True

        if 'missing_analysis_run' in st.session_state and st.session_state['missing_analysis_run']:
            st.header("Missing Values Analysis")
            missing_value_analysis(df)
            st.session_state['missing_analysis_run'] = False  # Reset the flag after analysis
            ok_button(['missing_analysis_run'])  # OK button to deactivate Missing Value Analysis functionality

        # Method for handling missing values
        method = st.sidebar.selectbox("Select Method", ["mean", "median", "mode", "drop"], key="missing_method")
        column = st.sidebar.selectbox("Select Column (optional)", df.columns, key="missing_col")

        if st.sidebar.button("Handle Missing Values", key='handle_missing_btn'):
            reset_flags(except_flags=['missing_values_handled'])  # Reset all except 'missing_values_handled' flag
            df = handle_missing_values(df, method, column)
            st.session_state['data'] = df
            st.session_state['missing_values_handled'] = True

        if 'missing_values_handled' in st.session_state and st.session_state['missing_values_handled']:
            missing_value_analysis(df)
            st.session_state['missing_values_handled'] = False
            ok_button(['missing_values_handled'])  # OK button to deactivate Handle Missing Values functionality

            
        #duplicate
      # Sidebar option for handling duplicates
        if st.sidebar.button("Remove Duplicates", key='handle_duplicates_btn'):
            # Reset flags except for 'duplicates_handled' to ensure proper state management
            reset_flags(except_flags=['duplicates_handled'])
            # Set 'duplicates_handled' to True to trigger the handling of duplicates
            st.session_state['duplicates_handled'] = True

        if 'duplicates_handled' in st.session_state and st.session_state['duplicates_handled']:
            st.header("Duplicates Analysis")

            # Call handle_duplicates to process the DataFrame and update it in session state
            df = handle_duplicates(st.session_state['data'])
            st.session_state['duplicates_handled'] = False

    # Add the OK button to deactivate the "Handle Duplicates" functionality
            ok_button(['duplicates_handled']) # OK button to deactivate Duplicate Handling functionality

            # Display the updated DataFrame after handling duplicates
            # st.write("### Updated DataFrame:")
            # st.dataframe(df)

            # # Add the OK button to deactivate the "Handle Duplicates" functionality
            # if st.button("OK", key='ok_duplicates_handling'):
            #     st.session_state['duplicates_handled'] = False


        # Outlier analysis section
        column_for_outlier = st.sidebar.selectbox("Select Column for Outlier Analysis", df.select_dtypes(include=['float64', 'int64']).columns, key="outlier_col")

        if st.sidebar.button("Outlier Analysis", key='outlier_analysis_btn'):
            reset_flags(except_flags=['outlier_analysis_run'])  # Reset all except 'outlier_analysis_run' flag
            st.session_state['outlier_analysis_run'] = True

        if 'outlier_analysis_run' in st.session_state and st.session_state['outlier_analysis_run']:
            st.header("Outlier Analysis")
            lower_bound, upper_bound = outlier_analysis(df, column_for_outlier)
            if lower_bound is not None and upper_bound is not None:
                outlier_method = st.sidebar.selectbox("Select Outlier Handling Method", ['clip', 'drop'], key="outlier_method")
                if st.sidebar.button("Handle Outliers", key='handle_outliers_btn'):
                    df = handle_outliers(df, column_for_outlier, lower_bound, upper_bound, outlier_method)
                    st.session_state['data'] = df
                    st.session_state['outliers_handled'] = True
                    
        if 'outliers_handled' in st.session_state and st.session_state['outliers_handled']:
            st.header("Data after Handling Outliers")
            st.dataframe(st.session_state['data'])
            st.session_state['outliers_handled'] = False
            ok_button(['outlier_analysis_run'])  # OK button to deactivate Outlier Analysis functionality

        # Button for data visualization
        column_for_visualization = st.sidebar.selectbox("Select Numerical Column for Visualization", df.select_dtypes(include=['float64', 'int64']).columns, key="visualize_column")

        if st.sidebar.button("Visualize Data", key='visualize_data_btn'):
            reset_flags(except_flags=['visualize_data_triggered'])  # Reset all except 'visualize_data_triggered' flag
            st.session_state['visualize_data_triggered'] = True

        if 'visualize_data_triggered' in st.session_state and st.session_state['visualize_data_triggered']:
            st.header(f"Visualizations for {column_for_visualization}")
            fig, fig2 = visualize_data(df, column_for_visualization)
            if fig and fig2:
                st.pyplot(fig)
                st.pyplot(fig2)
            st.session_state['visualize_data_triggered'] = False
            ok_button(['visualize_data_triggered'])  # OK button to deactivate Visualize Data functionality

        # Categorical column visualization
        column_for_categorical_visualization = st.sidebar.selectbox("Select Categorical Column for Visualization", df.select_dtypes(include=['object', 'category']).columns, key="visualize_categorical_column")

        if st.sidebar.button("Visualize Categorical Data", key='visualize_categorical_data_btn'):
            reset_flags(except_flags=['visualize_categorical_data_triggered'])  # Reset all except 'visualize_categorical_data_triggered' flag
            st.session_state['visualize_categorical_data_triggered'] = True

        if 'visualize_categorical_data_triggered' in st.session_state and st.session_state['visualize_categorical_data_triggered']:
            st.header(f"Visualizations for {column_for_categorical_visualization}")
            fig, fig2 = visualize_categorical_data(df, column_for_categorical_visualization)
            if fig and fig2:
                st.pyplot(fig)
                st.pyplot(fig2)
            st.session_state['visualize_categorical_data_triggered'] = False
            ok_button(['visualize_categorical_data_triggered'])  # OK button to deactivate Visualize Categorical Data functionality

        # Correlation matrix button
        if st.sidebar.button("Correlation Matrix", key='correlation_matrix_btn'):
            reset_flags(except_flags=['correlation_matrix_run'])  # Reset all except 'correlation_matrix_run' flag
            st.session_state['correlation_matrix_run'] = True

        if 'correlation_matrix_run' in st.session_state and st.session_state['correlation_matrix_run']:
            st.header("Correlation Matrix")
            fig = correlation_matrix(st.session_state.get('data', None))
            if fig is not None:
                st.pyplot(fig)
            st.session_state['correlation_matrix_run'] = False
            ok_button(['correlation_matrix_run'])  # OK button to deactivate Correlation Matrix functionality

        # Rename Column button
        if st.sidebar.button("Rename Column", key='rename_column_btn'):
            reset_flags(except_flags=['rename_column_trigger'])  # Reset all except 'rename_column_trigger' flag
            st.session_state['rename_column_trigger'] = True

        if st.session_state.get('rename_column_trigger'):
            if not df.empty:
                st.header("Rename Columns")
                st.write("### Original DataFrame:")
                st.dataframe(df)

                column_to_rename = st.selectbox("Select a column to rename", df.columns, key="rename_column_select")
                new_column_name = st.text_input("Enter the new column name", key="new_column_name_input")

                if st.button("Apply Renaming", key="apply_rename_btn"):
                    df_original = df.copy()
                    rename_columns(df, column_to_rename, new_column_name)
                    st.session_state['data'] = df
                    st.session_state['rename_column_trigger'] = False
                    st.write("### DataFrames Before and After Renaming:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Before Renaming:**")
                        st.dataframe(df_original)
                    with col2:
                        st.write("**After Renaming:**")
                        st.dataframe(df)

                ok_button(['rename_column_trigger'])  # OK button to deactivate Rename Column functionality
        
        if st.sidebar.button("Drop Column", key='drop_column_btn'):
            reset_flags(except_flags=['drop_column_trigger'])  # Reset all except 'drop_column_trigger' flag
            st.session_state['drop_column_trigger'] = True

        if st.session_state.get('drop_column_trigger'):
            if not df.empty:
                st.header("Drop Columns")
                st.write("### Original DataFrame:")
                st.dataframe(df)

                # Select the column to drop
                column_to_drop = st.selectbox("Select a column to drop", df.columns, key="drop_column_select")

                # Button to apply the column drop
                if st.button("Apply Drop", key="apply_drop_btn"):
                    df_original = df.copy()  # Copy the original DataFrame for comparison
                    df = drop_column(df, column_to_drop)  # Call the drop_column function
                    st.session_state['data'] = df  # Update the session state with the modified DataFrame
                    st.session_state['drop_column_trigger'] = False  # Reset the trigger flag

                    # Display the original and updated DataFrames side by side
                    st.write("### DataFrames Before and After Dropping the Column:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Before Dropping:**")
                        st.dataframe(df_original)
                    with col2:
                        st.write("**After Dropping:**")
                        st.dataframe(df)

                ok_button(['drop_column_trigger'])  # OK button to deactivate the functionality
        

        if st.sidebar.button("Download dataset", key='download_btn'):
            download_dataset(df)
            ok_button(['download_trigger'])  # Assuming you may want to reset this flag after download
        
if __name__ == "__main__":
    main()

    # def handle_action_reset(action_name, reset_flags_list=None):
#     """Helper function to reset flags and set the session state for specific actions."""
#     reset_flags(except_flags=reset_flags_list)
#     st.session_state[action_name] = True

# st.title("Chat with Llama2.3 locally")
        
        # # Initialize session states
        # initialize_session()
        
        # # Display the chat history
        # display_messages()
        
        # # Handle user input and response generation
        # handle_message_input()
        # #RAAAAAAAAAAAAAAAAAAAAAAAG
        # if st.sidebar.button("Start RAG Chat", key='start_rag_chat'):
        #     vector_store = build_vector_store(df)
        #     start_rag_chat(vector_store)

        # # Chat interface
        # if 'vector_store' in st.session_state:
        #     st.header("RAG Chat")
        #     handle_chat()
    # if st.sidebar.button("Start Chat with Ollama", key='start_chat_btn'):
    #     st.session_state['chat_started'] = True  # Flag to start chat session

    # if 'chat_started' in st.session_state and st.session_state['chat_started']:
    #     st.header("Chat with Ollama 3.2")
        
    #     # Input box for user prompt
    #     user_input = st.text_input("Ask a question", key='user_input')
        
    #     if user_input:
    #         # Get response from Ollama
    #         response = query_ollama(user_input)
    #         if response:
    #             st.write(f"**Ollama's Response:** {response}")
    #         else:
    #             st.write("Sorry, something went wrong with the chat.")
        
    #     # Button to end chat session
    #     if st.button("End Chat", key='end_chat_btn'):
    #         st.session_state['chat_started'] = False
    #         st.write("Chat session ended.")
