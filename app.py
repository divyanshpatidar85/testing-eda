import os
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global data storage
data_storage = {
    'df': None, 
    'filename': '',
    'upload_time': None,
    'file_info': {}
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_convert_to_serializable(obj):
    """Convert numpy/pandas objects to JSON serializable format"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj

def format_describe_stats(describe_dict):
    """Format statistical description for better display"""
    formatted = {}
    for column, stats in describe_dict.items():
        formatted[column] = {}
        for stat_name, value in stats.items():
            if pd.isna(value):
                formatted[column][stat_name] = "N/A"
            elif isinstance(value, (int, float)):
                if stat_name == 'count':
                    formatted[column][stat_name] = int(value)
                else:
                    formatted[column][stat_name] = round(float(value), 4)
            else:
                formatted[column][stat_name] = str(value)
    return formatted

def get_file_info(filepath):
    """Get file information"""
    try:
        file_size = os.path.getsize(filepath)
        file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
        return {
            'size': file_size,
            'size_formatted': format_file_size(file_size),
            'modified': file_modified.isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        return {}

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and initial parsing"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400

        # Check file type
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': 'File type not allowed. Please upload CSV, XLS, or XLSX files.'}), 400

        # Secure filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get file information
        file_info = get_file_info(filepath)
        logger.info(f"File uploaded: {filename} ({file_info.get('size_formatted', 'unknown size')})")

        # Parse file based on extension
        try:
            if filename.lower().endswith('.csv'):
                # Try different encodings for CSV
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding)
                        logger.info(f"CSV loaded successfully with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    return jsonify({'error': 'Unable to decode CSV file. Please check file encoding.'}), 500
                    
            else:  # Excel files
                df = pd.read_excel(filepath)
                logger.info(f"Excel file loaded successfully")

            # Validate dataframe
            if df.empty:
                return jsonify({'error': 'The uploaded file is empty or contains no data.'}), 400

            # Store data
            data_storage['df'] = df
            data_storage['filename'] = filename
            data_storage['upload_time'] = datetime.now()
            data_storage['file_info'] = file_info

            # Prepare response
            response_data = {
                'message': 'File uploaded and parsed successfully',
                'filename': filename,
                'columns': df.columns.tolist(),
                'rows': len(df),
                'file_info': file_info,
                'upload_time': data_storage['upload_time'].isoformat(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            }

            logger.info(f"File processed successfully: {len(df)} rows, {len(df.columns)} columns")
            return jsonify(response_data), 200

        except Exception as parse_error:
            logger.error(f"File parsing error: {str(parse_error)}")
            return jsonify({
                'error': f'Failed to parse file: {str(parse_error)}',
                'details': 'Please ensure the file is a valid CSV or Excel file with proper formatting.'
            }), 500

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error during file upload',
            'details': str(e)
        }), 500

@app.route('/getEDA', methods=['GET'])
def get_eda():
    """Generate and return EDA results"""
    try:
        df = data_storage.get('df')

        if df is None:
            return jsonify({'error': 'No file has been uploaded yet. Please upload a file first.'}), 400

        logger.info(f"Generating EDA for dataset: {df.shape[0]} rows, {df.shape[1]} columns")

        # Basic information
        shape = list(df.shape)
        dtypes = df.dtypes.astype(str).to_dict()
        null_counts = df.isnull().sum().to_dict()
        
        # Convert null counts to regular integers
        null_counts = {k: int(v) for k, v in null_counts.items()}

        # Numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_describe = {}
        if numeric_cols:
            numeric_stats = df[numeric_cols].describe()
            numeric_describe = format_describe_stats(numeric_stats.to_dict())

        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_describe = {}
        if categorical_cols:
            cat_stats = df[categorical_cols].describe()
            categorical_describe = format_describe_stats(cat_stats.to_dict())

        # Sample data (first 5 rows)
        head_data = df.head(5).fillna('N/A').to_dict(orient='records')
        
        # Convert any remaining numpy types to native Python types
        for row in head_data:
            for key, value in row.items():
                row[key] = safe_convert_to_serializable(value)

        # Additional insights
        insights = {
            'total_cells': int(df.shape[0] * df.shape[1]),
            'missing_cells': int(df.isnull().sum().sum()),
            'missing_percentage': round(float(df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100, 2),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'duplicate_rows': int(df.duplicated().sum())
        }

        # Column info with more details
        column_info = {}
        for col in df.columns:
            col_data = df[col]
            column_info[col] = {
                'dtype': str(col_data.dtype),
                'non_null_count': int(col_data.count()),
                'null_count': int(col_data.isnull().sum()),
                'unique_count': int(col_data.nunique()),
                'is_numeric': col in numeric_cols,
                'is_categorical': col in categorical_cols
            }

        # Prepare final EDA result
        eda_result = {
            'shape': shape,
            'dtypes': dtypes,
            'null_counts': null_counts,
            'numeric_describe': numeric_describe,
            'categorical_describe': categorical_describe,
            'head': head_data,
            'insights': insights,
            'column_info': column_info,
            'filename': data_storage['filename'],
            'upload_time': data_storage['upload_time'].isoformat() if data_storage['upload_time'] else None
        }

        logger.info("EDA generated successfully")
        return jsonify(eda_result), 200

    except Exception as e:
        logger.error(f"EDA generation error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Failed to generate EDA',
            'details': str(e)
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get current status of the application"""
    try:
        df = data_storage.get('df')
        status = {
            'has_data': df is not None,
            'filename': data_storage.get('filename', ''),
            'upload_time': data_storage.get('upload_time').isoformat() if data_storage.get('upload_time') else None,
            'data_shape': list(df.shape) if df is not None else None,
            'server_time': datetime.now().isoformat()
        }
        return jsonify(status), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_data():
    """Clear stored data"""
    try:
        data_storage['df'] = None
        data_storage['filename'] = ''
        data_storage['upload_time'] = None
        data_storage['file_info'] = {}
        
        logger.info("Data storage cleared")
        return jsonify({'message': 'Data cleared successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large',
        'details': f'Maximum file size allowed is {MAX_FILE_SIZE / 1024 / 1024:.0f}MB'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'error': 'Internal server error',
        'details': 'Please try again or contact support if the problem persists'
    }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Max file size: {MAX_FILE_SIZE / 1024 / 1024:.0f}MB")
    logger.info(f"Allowed extensions: {ALLOWED_EXTENSIONS}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)