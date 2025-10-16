# CRM Sales Opportunities Analysis Project
**Student ID**: 1164198  
**Course**: COMP647 - Machine Learning and Applications  
**Institution**: Lincoln University

## Project Overview
This project analyzes CRM sales opportunities data to uncover patterns, relationships, and insights that can drive business decision-making. The analysis includes data exploration, preprocessing, integration, and comprehensive exploratory data analysis (EDA) of multi-table CRM database.

## Folder Structure
```
1164198/
│
├── data/                           # Data directory
│   ├── raw/                        # Original unprocessed data files
│   ├── interim/                    # Intermediate preprocessed data
│   └── processed/                  # Cleaned and integrated datasets
│       └── crm_master_dataset.csv  # Final integrated dataset for analysis
│
├── assignment-2.ipynb              # Complete analysis pipeline (all 4 parts combined)
├── assignment_2.py                 # Data preprocessing utility functions
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation (this file)
```

## File Descriptions

### Data Files

#### `data/raw/`
- **Original CRM data files**: Contains the raw data exports from the CRM system including:
  - `accounts.csv` - Company information
  - `products.csv` - Product catalog
  - `sales_pipeline.csv` - Sales opportunities
  - `sales_teams.csv` - Sales agent and manager data
  - `data_dictionary.csv` - Field definitions

#### `data/interim/`
- **Preprocessed tables**: Created automatically by Part 2 of the notebook
- Contains cleaned versions of each raw table with suffix `_preprocessed.csv`
- Includes quality improvements: missing data imputed, outliers handled, types standardized

#### `data/processed/crm_master_dataset.csv`
- **Integrated master dataset**: Combined and cleaned dataset with 8,800 rows × 18 columns
- **Key features**:
  - `opportunity_id`: Unique identifier for each sales opportunity
  - `sales_agent`: Name of the sales representative
  - `product`: Product being sold (GTX Basic, GTXPro, MG Special, etc.)
  - `account`: Customer account name
  - `deal_stage`: Current stage (Won, Lost, Engaging)
  - `engage_date`: Date of first engagement
  - `close_date`: Date deal was closed
  - `close_value`: Final deal value
  - `sector`: Industry sector of the customer
  - `year_established`: Year the customer company was founded
  - `revenue`: Customer company revenue
  - `employees`: Number of employees at customer company
  - `office_location`: Geographic location of customer
  - `subsidiary_of`: Parent company if applicable
  - `series`: Product series category
  - `sales_price`: List price of the product
  - `manager`: Sales manager overseeing the agent
  - `regional_office`: Regional office handling the account

### Analysis Files

#### `assignment-2.ipynb` - Complete Analysis Pipeline
A comprehensive Jupyter notebook containing the entire data science workflow from raw data to insights. The notebook is structured in 4 sequential parts:

**Part 1: Data Exploration**
- Loading and examining 5-table CRM database
- Understanding data types and structures
- Business context and table relationships
- Initial data quality assessment
- Preliminary statistics and observations

**Part 2: Data Preprocessing**
- Systematic data loading and quality assessment
- Duplicate detection and handling
- Missing data pattern analysis with visualizations
- Intelligent imputation strategies (business logic-based)
- Multi-method outlier detection (IQR, Z-score, Modified Z-score)
- Data transformation and standardization
- Quality improvement metrics and reporting
- Saves preprocessed tables to `data/interim/`

**Part 3: Data Integration**
- Loading preprocessed tables from interim directory
- Foreign key relationship validation
- Master dataset creation via left joins
- Integration quality assessment
- Business logic validation (e.g., Won deals have values)
- Saves master dataset to `data/processed/crm_master_dataset.csv`

**Part 4: Exploratory Data Analysis**
- Load master dataset (8,800 rows × 18 columns)
- Descriptive statistics and correlation analysis
- Deal stage distribution analysis
- Pair plots for multi-dimensional relationships
- Temporal performance trends over time
- Sector and regional performance analysis
- Sales agent performance comparison
- Product series effectiveness analysis
- Sales cycle time analysis
- Company size impact on deals
- Win rate analysis across dimensions
- Key business insights and research questions

#### `assignment_2.py` - Preprocessing Utilities
A modular library of data preprocessing functions organized into classes:
- **DataQualityAssessment**: Quality metrics and placeholder pattern detection
- **DuplicateHandler**: Duplicate analysis and removal strategies
- **MissingDataAnalyzer**: Missing data pattern analysis and visualization
- **ImputationStrategies**: Numerical, categorical, and business logic imputation
- **OutlierDetection**: IQR, Z-score, and Modified Z-score methods with visualization
- **DataTransformation**: Column name standardization and type conversion
- **QualityReporting**: Quality comparison and preprocessing reports

All classes include function wrappers for convenient usage.

## Key Findings

### Performance Metrics
- **Total Opportunities**: 8,800 sales opportunities analyzed
- **Win Rate**: 48.1% overall success rate
- **Average Deal Value**: $1,400 (varies significantly by sector and product)
- **Average Sales Cycle**: 46 days from engagement to close

### Critical Insights
1. **Sales price is the strongest predictor** of final deal value (correlation: 0.7-0.8)
2. **Company size matters**: Larger companies (by revenue/employees) tend to have bigger deals
3. **Regional performance varies significantly**: Indicating opportunity for best practice sharing
4. **Product-market fit differs**: Certain products perform better in specific sectors
5. **Agent performance has high variance**: Top performers consistently close higher-value deals

## Potential Research Questions
1. What factors predict deal success (Won vs Lost)?
2. Can we predict deal value before engagement?
3. What drives sales cycle length variation?
4. Which customer segments offer the highest ROI?
5. How can we optimize sales team performance?
6. Is there an optimal product-market fit by sector?
7. What causes the gap between list price and closed value?

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- jupyter

## Getting Started
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Launch Jupyter: `jupyter notebook`
4. Open `assignment-2.ipynb`
5. Run all cells sequentially to execute the complete analysis pipeline

**Note**: The notebook will automatically create `data/interim/` and `data/processed/` directories during execution.
