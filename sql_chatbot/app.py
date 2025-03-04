import streamlit as st
import pandas as pd
import os
import re
import pyarrow as pa
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import mysql.connector
from sqlalchemy import create_engine, inspect,text
from urllib.parse import quote_plus

load_dotenv()

# Page configuration with custom theme and expanded layout
st.set_page_config(
    page_title="SQL Query Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #f8f9fa;
        color: #343a40;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #3949ab;
        font-weight: 600;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        position: relative;
        max-width: 80%;
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        border-top-right-radius: 0;
    }
    
    .assistant-message {
        background-color: #f1f3f4;
        margin-right: auto;
        border-top-left-radius: 0;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #3949ab;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #303f9f;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Input field styling */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f1f3f4;
    }
    
    /* File uploader styling */
    .stFileUploader > div > button {
        background-color: #e3f2fd;
        color: #3949ab;
    }
    
    /* Code blocks */
    code {
        border-radius: 5px;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Data frame styling */
    .dataframe-container {
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        border-radius: 10px;
        padding: 1rem;
        background-color: white;
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 500;
    }
    .status-success {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .status-pending {
        background-color: #fff8e1;
        color: #f57c00;
    }
    .status-error {
        background-color: #fce4ec;
        color: #c2185b;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 1px;
        background-color: #e0e0e0;
        margin: 1.5rem 0;
    }
    
    /* SQL highlighting */
    .sql-highlight {
        background-color: #f5f7ff;
        border-left: 4px solid #3949ab;
        padding: 1rem;
        border-radius: 4px;
        font-family: 'Roboto Mono', monospace;
        overflow-x: auto;
        margin: 1rem 0;
    }
    
    /* Data source tabs */
    .data-source-tab {
        padding: 8px 16px;
        background-color: #f1f3f4;
        border-radius: 8px 8px 0 0;
        cursor: pointer;
        border: 1px solid #ddd;
        border-bottom: none;
    }
    
    .data-source-tab.active {
        background-color: #3949ab;
        color: white;
    }
    
    /* Database connection form */
    .connection-form {
        background-color: white;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Check if API key is in environment
api_key_exists = "GOOGLE_API_KEY" in os.environ

# Initialize session state variables
if 'messages' not in st.session_state:
    welcome_message = "👋 Hello! I'm your SQL Query Assistant. You can upload CSV files or connect to a MySQL database, and I'll help you generate SQL queries from natural language questions."
    if api_key_exists:
        welcome_message += "\n\n✅ Google API Key detected in environment variables."
    st.session_state.messages = [
        {"role": "assistant", "content": welcome_message}
    ]
if 'table_schemas' not in st.session_state:
    st.session_state.table_schemas = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = api_key_exists
if 'file_upload_state' not in st.session_state:
    st.session_state.file_upload_state = "not_started"  # Options: not_started, in_progress, completed
if 'last_sql' not in st.session_state:
    st.session_state.last_sql = ""
if 'model_name' not in st.session_state:
    st.session_state.model_name = "gemini-1.5-pro"  # Updated model name
if 'data_source' not in st.session_state:
    st.session_state.data_source = "csv"  # Options: csv, mysql
if 'db_connection' not in st.session_state:
    st.session_state.db_connection = None
if 'db_engine' not in st.session_state:
    st.session_state.db_engine = None

# Function to clean SQL language markers
def clean_sql_output(text):
    # Remove ```sql and ``` markers
    text = re.sub(r'```sql\s*', '', text)
    text = re.sub(r'```', '', text)
    return text
def handle_duplicate_columns(df):
    """Rename duplicate columns by appending incrementing numbers"""
    cols = pd.Series(df.columns)
    duplicates = cols[cols.duplicated()].unique()
    
    for dup in duplicates:
        cnt = 1
        indices = cols[cols == dup].index
        for idx in indices[1:]:  # Keep first occurrence as is
            cols[idx] = f"{dup}_{cnt}"
            cnt += 1
            
    return df.set_axis(cols, axis=1)

def prepare_dataframe_for_display(df):
    """Convert DataFrame to display-safe version with unique columns"""
    # Create a copy to avoid modifying the original
    safe_df = df.copy()
    
    # Handle duplicate columns first
    safe_df = handle_duplicate_columns(safe_df)
    
    # Convert pandas extension types to standard types
    for col in safe_df.columns:
        dtype_name = str(safe_df[col].dtype)
        if 'Int' in dtype_name or 'Float' in dtype_name:
            safe_df[col] = safe_df[col].astype('object')
    
    return safe_df

# Function to get status badge HTML
def get_status_badge(status, text):
    return f'<span class="status-badge status-{status}">{text}</span>'

# First install: pip install pymysql

def test_mysql_connection(host, port, user, password, database, use_ssl=False):
    try:
        password_escaped = quote_plus(password)
        connection_string = f"mysql+pymysql://{user}:{password_escaped}@{host}:{port}/{database}"
        
        if use_ssl:
            connection_string += "?ssl_ca=/path/to/ca-cert.pem&ssl_cert=/path/to/client-cert.pem&ssl_key=/path/to/client-key.pem"
        
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()  # Actually consume the result
            
        return True, f"Connected to {database}@{host}", engine

    except Exception as e:
        return False, str(e), None


# Function to get MySQL table schemas
def get_mysql_schemas(engine):
    try:
        # Get all table names
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        if not table_names:
            return [], {}, "No tables found in the database."
        
        # Get schemas for each table
        table_schemas = []
        file_details = {}
        
        for table_name in table_names:
            # Get column details
            columns = [col['name'] for col in inspector.get_columns(table_name)]
            schema = f"Table: {table_name} ({', '.join(columns)})"
            table_schemas.append(schema)
            
            # Get sample data
            query = f"SELECT * FROM {table_name} LIMIT 1000"
            df = pd.read_sql(query, engine)
            file_details[table_name] = df
        
        return table_schemas, file_details, None
    except Exception as e:
        return [], {}, f"Error getting database schemas: {str(e)}"

# Function to create embeddings and vectorstore
def create_embeddings(table_schemas):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.create_documents(table_schemas)
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore, None
    except Exception as primary_error:
        try:
            # Try alternative model
            embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.create_documents(table_schemas)
            vectorstore = FAISS.from_documents(docs, embeddings)
            return vectorstore, None
        except Exception as alt_error:
            return None, f"Could not create embeddings with any available model: {str(alt_error)}"

# Function to generate response
def generate_response(prompt):
    if not st.session_state.api_key_set:
        return "⚠️ Google API Key not found. Please check your environment variables or set it manually in the sidebar."
    elif not st.session_state.table_schemas:
        return "⚠️ Please upload CSV files or connect to a database in the sidebar first."
    else:
        try:
            # Retrieve relevant tables
            docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
            relevant_tables = "\n".join([doc.page_content for doc in docs])
            
            # Generate SQL query using the specified model
            try:
                llm = GoogleGenerativeAI(model=st.session_state.model_name)
                
                # Use the LLM directly instead of deprecated LLMChain
                sql_prompt_template = """
                Given the following table schemas:
                {tables}

                Generate a clean SQL query to answer the question. Do not include ```sql or ``` markers in your response:
                {query}

                SQL Query:
                """
                
                formatted_prompt = sql_prompt_template.format(
                    tables=relevant_tables,
                    query=prompt
                )
                
                response = llm.invoke(formatted_prompt)
                sql_query = clean_sql_output(response)
                
                # Save SQL for download
                st.session_state.last_sql = sql_query
                
                # Format the response with modern styling
                response = f"Based on your question, I've identified these relevant tables:\n\n"
                response += f"{relevant_tables}\n\n"
                response += f"Here's the SQL query I generated:\n\n"
                response += f"```sql\n{sql_query}\n```\n\n"
                response += "Is there anything you'd like me to modify about this query?"
                
                return response
                
            except Exception as model_error:
                # Try alternative model if first one fails
                st.warning(f"Error with primary model: {str(model_error)}. Trying alternative model...")
                
                # Try with a different model
                alternative_models = ["gemini-pro", "gemini-1.0-pro", "gemini-1.0-pro-latest"]
                
                for model in alternative_models:
                    try:
                        st.session_state.model_name = model
                        llm = GoogleGenerativeAI(model=model)
                        
                        # Same prompt template as above
                        sql_prompt_template = """
                        Given the following table schemas:
                        {tables}

                        Generate a clean SQL query to answer the question. Do not include ```sql or ``` markers in your response:
                        {query}

                        SQL Query:
                        """
                        
                        formatted_prompt = sql_prompt_template.format(
                            tables=relevant_tables,
                            query=prompt
                        )
                        
                        response = llm.invoke(formatted_prompt)
                        sql_query = clean_sql_output(response)
                        
                        # Save SQL for download
                        st.session_state.last_sql = sql_query
                        
                        # Format the response with modern styling
                        response = f"Based on your question, I've identified these relevant tables:\n\n"
                        response += f"{relevant_tables}\n\n"
                        response += f"Here's the SQL query I generated:\n\n"
                        response += f"```sql\n{sql_query}\n```\n\n"
                        response += "Is there anything you'd like me to modify about this query?"
                        
                        return response
                    
                    except Exception as alt_error:
                        continue
                
                # If all models fail
                return f"❌ Unable to generate a response with any available models. Please check your API key and permissions. Error: {str(model_error)}"
                
        except Exception as e:
            return f"❌ An error occurred: {str(e)}"

# Function to execute SQL query on MySQL database
def execute_query_on_mysql(query):
    if not st.session_state.db_engine:
        return None, "No database connection available"
    
    try:
        df = pd.read_sql(query, st.session_state.db_engine)
        return df, None
    except Exception as e:
        return None, f"Error executing query: {str(e)}"

# Sidebar for configuration with modern styling
with st.sidebar:
    st.title("🛠️ Configuration")
    
    # API Key status with modern indicator
    st.markdown("### API Authentication")
    if api_key_exists:
        st.markdown(f"Google API Key: {get_status_badge('success', 'Connected')}", unsafe_allow_html=True)
    else:
        st.markdown(f"Google API Key: {get_status_badge('error', 'Missing')}", unsafe_allow_html=True)
        api_key = st.text_input("Enter Google API Key", type="password", 
                               help="Your API key will only be stored for this session")
        col1, col2 = st.columns([1, 1])
        with col1:
            if api_key and st.button("Connect", use_container_width=True):
                os.environ["GOOGLE_API_KEY"] = api_key
                st.session_state.api_key_set = True
                st.success("API Key set successfully!")
    
    # Model selection
    st.markdown("### Model Settings")
    model_options = ["gemini-1.5-pro", "gemini-pro", "gemini-1.0-pro", "gemini-1.0-pro-latest"]
    selected_model = st.selectbox(
        "Select Google AI Model", 
        options=model_options,
        index=model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0
    )
    
    if selected_model != st.session_state.model_name:
        st.session_state.model_name = selected_model
        st.success(f"Model changed to {selected_model}")
    
    st.markdown("---")
    
    # Data source selection
    st.markdown("### Data Sources")
    data_source = st.radio("Select data source:", ["CSV Files", "MySQL Database"], 
                          index=0 if st.session_state.data_source == "csv" else 1)
    
    # Update data source in session state
    st.session_state.data_source = "csv" if data_source == "CSV Files" else "mysql"
    
    # Show appropriate data source interface
    if st.session_state.data_source == "csv":
        # CSV file uploader with modern styling
        upload_status = "success" if st.session_state.file_upload_state == "completed" else "pending"
        status_text = "Ready" if upload_status == "success" else "Waiting"
        
        st.markdown(f"CSV Files: {get_status_badge(upload_status, status_text)}", unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)
        
        process_button = st.button("Process Files", use_container_width=True, disabled=not uploaded_files)
        
        if process_button:
            if not st.session_state.api_key_set:
                st.error("Please set your Google API Key first!")
            else:
                st.session_state.file_upload_state = "in_progress"
                
                with st.spinner("Processing files..."):
                    try:
                        table_schemas = []
                        file_details = {}
                        
                        for uploaded_file in uploaded_files:
                            df = pd.read_csv(uploaded_file)
                            table_name = uploaded_file.name.split(".")[0]
                            schema = f"Table: {table_name} ({', '.join(df.columns)})"
                            table_schemas.append(schema)
                            file_details[table_name] = df
                        
                        # Store schemas in FAISS
                        vectorstore, error = create_embeddings(table_schemas)
                        
                        if error:
                            st.session_state.file_upload_state = "not_started"
                            st.error(error)
                        else:
                            # Save to session state
                            st.session_state.table_schemas = table_schemas
                            st.session_state.vectorstore = vectorstore
                            st.session_state.uploaded_files = file_details
                            st.session_state.file_upload_state = "completed"
                            
                            # Add message to chat
                            files_message = f"✅ Successfully processed {len(table_schemas)} file(s)!\n\nAvailable tables:\n"
                            for schema in table_schemas:
                                files_message += f"- {schema}\n"
                            
                            st.session_state.messages.append({"role": "assistant", "content": files_message})
                            st.success(f"Successfully processed {len(table_schemas)} file(s)!")
                        
                    except Exception as e:
                        st.session_state.file_upload_state = "not_started"
                        st.error(f"An error occurred: {str(e)}")
    
    else:
        # MySQL connection form
        connection_status = "success" if st.session_state.db_connection else "pending"
        status_text = "Connected" if connection_status == "success" else "Not Connected"
        
        st.markdown(f"MySQL Connection: {get_status_badge(connection_status, status_text)}", unsafe_allow_html=True)
        
        with st.form("mysql_connection_form"):
            st.markdown("#### MySQL Connection Details")
            mysql_host = st.text_input("Host", "localhost")
            mysql_port = st.text_input("Port", "3306")
            mysql_user = st.text_input("Username")
            mysql_password = st.text_input("Password", type="password")
            mysql_database = st.text_input("Database Name")
            mysql_ssl = st.checkbox("Use SSL")
            
            col1, col2 = st.columns(2)
            with col1:
                test_button = st.form_submit_button("Test Connection", use_container_width=True)
            with col2:
                connect_button = st.form_submit_button("Connect", use_container_width=True)
                
        if test_button:
            if not mysql_host or not mysql_user or not mysql_database:
                st.error("Please fill in all required fields.")
            else:
                with st.spinner("Testing connection..."):
                    success, message, _ = test_mysql_connection(
                        mysql_host, mysql_port, mysql_user, mysql_password, mysql_database, mysql_ssl
                    )
                    
                    if success:
                        st.success("Connection test successful!")
                    else:
                        st.error(f"Connection test failed: {message}")
        
        if connect_button:
            if not st.session_state.api_key_set:
                st.error("Please set your Google API Key first!")
            elif not mysql_host or not mysql_user or not mysql_database:
                st.error("Please fill in all required fields.")
            else:
                with st.spinner("Connecting to database..."):
                    success, message, engine = test_mysql_connection(
                        mysql_host, mysql_port, mysql_user, mysql_password, mysql_database, mysql_ssl
                    )
                    
                    if not success:
                        st.error(f"Connection failed: {message}")
                    else:
                        # Get table schemas
                        table_schemas, file_details, schema_error = get_mysql_schemas(engine)
                        
                        if schema_error:
                            st.error(schema_error)
                        else:
                            # Create embeddings
                            vectorstore, embed_error = create_embeddings(table_schemas)
                            
                            if embed_error:
                                st.error(embed_error)
                            else:
                                # Save to session state
                                st.session_state.table_schemas = table_schemas
                                st.session_state.vectorstore = vectorstore
                                st.session_state.uploaded_files = file_details
                                st.session_state.file_upload_state = "completed"
                                st.session_state.db_connection = message
                                st.session_state.db_engine = engine
                                
                                # Add message to chat
                                db_message = f"✅ Successfully connected to MySQL database!\n\nAvailable tables:\n"
                                for schema in table_schemas:
                                    db_message += f"- {schema}\n"
                                
                                st.session_state.messages.append({"role": "assistant", "content": db_message})
                                st.success(f"Successfully connected to database with {len(table_schemas)} table(s)!")
    
    # Display schema preview with modern styling
    if st.session_state.table_schemas:
        st.markdown("### Available Tables")
        for i, schema in enumerate(st.session_state.table_schemas):
            with st.expander(f"Table {i+1}"):
                st.code(schema)
    
    # Display MySQL connection info if connected
    if st.session_state.db_connection:
        st.markdown("### Database Connection")
        st.info(f"Connected to: {st.session_state.db_connection.split('@')[1].split('/')[0]}")
        
        if st.button("Disconnect", use_container_width=True):
            # Close the engine connection if it exists
            if st.session_state.db_engine:
                try:
                    st.session_state.db_engine.dispose()
                except:
                    pass
            
            # Reset connection-related session state
            st.session_state.db_connection = None
            st.session_state.db_engine = None
            
            # Optionally also clear the data if we want to fully reset
            if st.session_state.data_source == "mysql":
                st.session_state.table_schemas = []
                st.session_state.vectorstore = None
                st.session_state.uploaded_files = {}
                st.session_state.file_upload_state = "not_started"
                
                # Add message to chat
                st.session_state.messages.append({"role": "assistant", "content": "Database connection closed."})
            
            st.success("Database disconnected.")
            st.rerun()
    
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Choose a data source (CSV files or MySQL database)
    2. Upload files or connect to your database
    3. Chat with the bot by asking questions about your data
    4. Get SQL queries generated from your questions
    """)

# Main layout with tabs for better organization
tab1, tab2, tab3 = st.tabs(["💬 Query Generator", "📊 Data Explorer", "🔍 Query Execution"])

# Tab 1: Modern Chat Interface
with tab1:
    st.title("SQL Query Assistant")
    st.markdown("Ask natural language questions and get SQL queries for your data")
    
    # Modern chat container with custom styling
    chat_container = st.container(height=400)
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                message_html = f"""
                <div class="chat-message user-message">
                    <p>{message["content"]}</p>
                </div>
                """
            else:
                # Replace SQL code blocks with styled ones
                content = message["content"]
                if "```sql" in content or "```" in content:
                    content = re.sub(r'```sql\n(.*?)\n```', r'<div class="sql-highlight">\1</div>', content, flags=re.DOTALL)
                    content = re.sub(r'```\n(.*?)\n```', r'<div class="sql-highlight">\1</div>', content, flags=re.DOTALL)
                
                message_html = f"""
                <div class="chat-message assistant-message">
                    <p>{content}</p>
                </div>
                """
            
            st.markdown(message_html, unsafe_allow_html=True)
    
    # SQL download section with modern styling
    if st.session_state.last_sql:
        st.markdown("### Generated SQL")
        st.code(st.session_state.last_sql, language="sql")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button(
                label="Download SQL",
                data=st.session_state.last_sql,
                file_name="generated_query.sql",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            if st.session_state.db_connection and st.button("Execute Query", use_container_width=True):
                # Switch to the Query Execution tab
                st.session_state.current_tab = "Query Execution"
                # Store the current query for execution
                st.session_state.query_to_execute = st.session_state.last_sql
                st.rerun()
    
    # Modern input section
    st.markdown("### Ask a Question")
    
    # Two-column layout for input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.text_area(
            "Type your question here:",
            height=100,
            placeholder="e.g., 'Show me employees with salary above average but below their department average'",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if st.button("Generate SQL", type="primary", use_container_width=True):
            if prompt:
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Generate assistant response with spinner
                with st.spinner("Generating SQL query..."):
                    response = generate_response(prompt)
                
                # Add assistant message to chat
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Force a rerun to update the chat display
                st.rerun()

# Tab 2: Data Explorer with modern styling
with tab2:
    st.title("Data Explorer")
    
    if st.session_state.uploaded_files:
        col1, col2 = st.columns([1, 3])
        
                # Tab 2: Data Explorer with modern styling (continued)
        with col1:
            selected_table = st.selectbox("Choose a table:", 
                                          options=list(st.session_state.uploaded_files.keys()))
            st.subheader("Basic Statistics")
            try:
                stats_df = st.session_state.uploaded_files[selected_table].describe().transpose()
                safe_stats = prepare_dataframe_for_display(stats_df)
                st.dataframe(safe_stats, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating statistics: {str(e)}")

        with col2:
            st.subheader("Data Types")
            try:
                # Create a safer representation of the dtypes
                dtype_dict = {col: str(dtype) for col, dtype in 
                     zip(st.session_state.uploaded_files[selected_table].dtypes.index,
                         st.session_state.uploaded_files[selected_table].dtypes.values)}
                dtype_df = pd.DataFrame.from_dict(dtype_dict, orient='index', columns=['Data Type'])
                st.dataframe(dtype_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying data types: {str(e)}")

    # Tab 3: Query Execution with modern styling
    with tab3:
        st.title("Query Execution")
    
        if st.session_state.data_source == "mysql" and st.session_state.db_connection:
            st.markdown("### Execute SQL Queries")
        
            # Text area for custom SQL input
            custom_sql = st.text_area(
            "Enter SQL Query:",
                height=200,
                value=st.session_state.get("query_to_execute", ""),
                help="Write or paste your SQL query here"
            )
        
            col1, col2 = st.columns([1, 3])
            with col1:
                execute_button = st.button("Execute Query", type="primary", use_container_width=True)
        
            if execute_button and custom_sql:
                with st.spinner("Executing query..."):
                    df, error = execute_query_on_mysql(custom_sql)
                
                    if error:
                        st.error(f"Execution error: {error}")
                    else:
                        st.success("Query executed successfully!")
                        st.markdown(f"**Results:** {len(df)} rows returned")
                    
                        # Display the result with modern styling
                        with st.container():
                            st.markdown("### Query Results")
                            safe_df = prepare_dataframe_for_display(df)
                            st.dataframe(
                                safe_df,
                                use_container_width=True,
                                hide_index=True,
                                height=400
                            )
                    
                        # Download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download as CSV",
                                data=csv,
                            file_name="query_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        # In the Query Execution section, modify the JSON download button code:
                        with col2:
    # Use the safe_df that already has unique column names
                            st.download_button(
        label="Download as JSON",
        data=safe_df.to_json(indent=2),  # Use safe_df instead of df
        file_name="query_results.json",
        mime="application/json",
        use_container_width=True
    )
            elif not st.session_state.db_connection:
                st.warning("⚠️ Please connect to a MySQL database first.")
        else:
            st.info("ℹ️ This feature is available when connected to a MySQL database. Connect using the sidebar.")

    

# Footer with subtle branding
st.markdown("---")
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; opacity: 0.7;">
        <span>Powered by Google Generative AI and LangChain</span>
        <span>SQL Query Assistant v2.0</span>
    </div>
    """, 
    unsafe_allow_html=True)