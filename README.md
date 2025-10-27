# Vehicle Repairs Data Analysis Project

## 📋 Project Overview

This project performs comprehensive data cleaning, analysis, and visualization of vehicle repair records. It includes automated data preprocessing, outlier detection, categorical standardization, NLP-based tag generation using TF-IDF, and insightful visualizations to support operational decision-making.

---

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.13** - Primary programming language
- **Jupyter Notebooks** - Interactive development and analysis environment

### Data Processing & Analysis
- **pandas 2.x** - Data manipulation and analysis
- **NumPy** - Numerical computing and array operations
- **scikit-learn** - Machine learning library (TF-IDF vectorization)
  - `TfidfVectorizer` - Natural language processing for tag generation

### Visualization
- **Matplotlib** - Static, animated, and interactive visualizations
  - `pyplot` - Plotting interface
  - `ticker` - Axis formatting and customization

### Text Processing
- **re (regex)** - Regular expressions for text cleaning and pattern matching

### Development Environment
- **Visual Studio Code** - Code editor with Jupyter extension
- **Git** - Version control
- **PowerShell** - Terminal environment

---

## 📁 Project Structure

```
qwerty/
│
├── analysis.ipynb                                    # Main analysis notebook (3 cells)
│                                                     # - Cell 1: Data cleaning pipeline
│                                                     # - Cell 2: Visualization generation
│                                                     # - Cell 3: NLP tag generation
│
├── task2.csv                                         # Raw input data (vehicle repair records)
├── cleaned_vehicle_repairs_Cleaned.csv               # Cleaned full dataset
├── transaction_id_with_consolidated_nlp_tags.csv     # Transaction IDs with NLP-generated tags
│
├── top_repair_types.png                              # Top 10 repairs visualization
├── repairs_by_platform.png                           # Platform distribution chart
├── cost_distribution.png                             # Cost histogram
│
├── generate_word_report.py                           # Script to generate Word report
├── Vehicle_Repairs_Analysis_Report.md                # Comprehensive analysis report (Markdown)
└── README.md                                         # This file
```

---

## 🚀 Getting Started

### Prerequisites

```bash
# Install required packages
pip install pandas numpy scikit-learn matplotlib
```

### Running the Analysis

#### Using Jupyter Notebook (Recommended)

**Open and run analysis.ipynb**:
```bash
# Open in VS Code or Jupyter
code analysis.ipynb
# Run all 3 cells sequentially
```

**Cell Breakdown**:
- **Cell 1**: Complete data cleaning pipeline
  - Load CSV data
  - Standardize columns
  - Clean text fields
  - Correct data types
  - Handle missing values
  - Detect and cap outliers
  - Standardize categorical data
  - Save cleaned dataset

- **Cell 2**: Visualization generation
  - Load cleaned data
  - Generate top 10 repair types chart
  - Generate platform distribution chart
  - Generate cost distribution histogram
  - Save as PNG files

- **Cell 3**: NLP tag generation
  - Load raw data
  - Apply cleaning pipeline
  - Generate TF-IDF tags from verbatims
  - Consolidate tags per transaction
  - Export transaction IDs with tags

#### Generating Word Report

```bash
# Install python-docx if not already installed
pip install python-docx

# Run the report generation script
python generate_word_report.py
```

This will create `Vehicle_Repairs_Analysis_Report.docx` with the complete analysis report.

---

## 📊 Features

### 1. Data Cleaning Pipeline
- ✅ **Column Standardization** - Snake_case naming convention
- ✅ **Text Cleaning** - Remove special characters, encoding fixes, lowercase normalization
- ✅ **Data Type Correction** - DateTime and numeric conversions
- ✅ **Missing Value Handling** - Threshold-based row filtering + intelligent imputation
- ✅ **Outlier Detection & Treatment** - IQR method with percentile capping
- ✅ **Categorical Consolidation** - Part name standardization (e.g., steering wheel variants)

### 2. NLP Tag Generation
- 🏷️ **TF-IDF Vectorization** - Extract top 30 significant terms
- 🏷️ **N-gram Analysis** - Capture 1-3 word phrases
- 🏷️ **Tag Consolidation** - Binary presence matrix → comma-separated tags
- 🏷️ **Verbatim Analysis** - Combines customer_verbatim + correction_verbatim

### 3. Visualizations
- 📈 **Top 10 Repair Types** - Horizontal bar chart identifying most frequent repairs
- 📈 **Repairs by Platform** - Platform-wise repair distribution
- 📈 **Cost Distribution** - Histogram showing repair cost patterns

### 4. Output Files
- 📄 **cleaned_vehicle_repairs_Cleaned.csv** - Full cleaned dataset with all transformations applied
- 📄 **transaction_id_with_consolidated_nlp_tags.csv** - Transaction IDs with NLP-generated tags
- 🖼️ **PNG Visualizations** - Three chart files for reporting
  - `top_repair_types.png` - Top 10 most common repairs
  - `repairs_by_platform.png` - Platform-wise distribution
  - `cost_distribution.png` - Cost histogram
- 📝 **Vehicle_Repairs_Analysis_Report.md** - Detailed 2-page analysis report
- 📊 **Vehicle_Repairs_Analysis_Report.docx** - Professional Word document report (generated via script)

---

## 📝 Report Section

### Comprehensive Analysis Report

A detailed **2-page analysis report** is available in `Vehicle_Repairs_Analysis_Report.md`, covering:

#### A. Column Analysis
- Complete dataset overview with column categories
- Data type distribution (numeric, datetime, categorical)
- Data quality observations and encoding issues

#### B. Data Cleaning Summary
1. **Standardization** - Column naming and text normalization
2. **Text Cleaning** - Special character removal, case conversion
3. **Type Corrections** - DateTime and numeric parsing
4. **Missing Values** - Row-level filtering + median/Unknown imputation
5. **Outlier Handling** - IQR detection with 1st-99th percentile capping
6. **Categorical Standardization** - Part name consolidation
7. **Output Generation** - Multiple CSV formats for different use cases

#### C. Visualizations
- **Top 10 Repair Types**: Identifies maintenance priorities
- **Platform Distribution**: Shows platform-specific repair frequencies
- **Cost Distribution**: Reveals typical repair cost ranges

#### D. Generated Tags & Key Takeaways

**NLP Configuration:**
- Method: TF-IDF (Term Frequency-Inverse Document Frequency)
- Features: 30 most significant terms
- N-grams: 1-3 word phrases
- Stop words: English

**Key Insights:**
1. ✅ Data quality improved through systematic cleaning
2. ✅ Top 10 repairs represent majority of operations
3. ✅ Significant platform variance in repair frequency
4. ✅ 30 NLP tags capture key repair themes
5. ✅ Categorical standardization reduced fragmentation
6. ✅ Clean dataset ready for predictive modeling

---

## 🔧 Data Pipeline Details

### Input
- **Source**: `task2.csv`
- **Encoding**: latin1
- **Format**: CSV with mixed data types

### Processing Steps
1. **Load** → Read CSV with error handling
2. **Standardize** → Column names to snake_case
3. **Clean** → Text fields (remove backslashes, encoding errors)
4. **Convert** → Data types (datetime, numeric)
5. **Filter** → Drop rows with >5 missing values
6. **Impute** → Fill remaining gaps (median for numeric, 'Unknown' for categorical)
7. **Cap** → Outliers at 1st and 99th percentiles
8. **Consolidate** → Categorical variations
9. **Vectorize** → Generate TF-IDF tags from verbatims
10. **Export** → Multiple output formats

### Output
- **Cleaned Data**: Full dataset with all transformations applied
- **NLP Tags**: Transaction-level tags for pattern analysis
- **Visualizations**: PNG charts for reporting and presentations

---

## 📈 Use Cases

- **Operational Analytics** - Identify high-frequency repairs and platforms
- **Cost Optimization** - Analyze repair cost distributions for budget planning
- **Predictive Maintenance** - Use cleaned data for ML model training
- **Part Standardization** - Improve inventory management with consolidated categories
- **Trend Analysis** - Track repair patterns over time
- **Resource Allocation** - Prioritize maintenance resources based on data insights

---

## 🤝 Contributing

This project is part of the Axion_Ray_2 repository. For contributions:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is maintained as part of the Axion_Ray_2 repository by Eniyan2113.

---

## 📞 Contact & Support

For questions or issues related to this analysis:
- Repository: [Axion_Ray_2](https://github.com/Eniyan2113/Axion_Ray_2)
- Branch: main

---

## 🔄 Version History

- **v1.0** - Initial release with complete data pipeline
  - Data cleaning and standardization
  - NLP tag generation with TF-IDF
  - Visualization suite
  - Comprehensive documentation

---

## 📚 Additional Resources

- **Markdown Report**: See `Vehicle_Repairs_Analysis_Report.md` for detailed findings in Markdown format
- **Word Report**: Run `generate_word_report.py` to create a professional Word document
- **Interactive Notebook**: Use `analysis.ipynb` for hands-on analysis and customization
- **Data Files**: 
  - Input: `task2.csv` (raw vehicle repair records)
  - Output: `cleaned_vehicle_repairs_Cleaned.csv` (cleaned dataset)
  - Tags: `transaction_id_with_consolidated_nlp_tags.csv` (NLP tags)

---

**Last Updated**: October 27, 2025  
**Data Source**: task2.csv (Vehicle repair records)  
**Analysis Tools**: Python, pandas, scikit-learn, matplotlib  
**Repository**: Axion_Ray_2 (GitHub - Eniyan2113)
