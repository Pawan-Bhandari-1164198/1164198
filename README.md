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
│   └── processed/                  # Cleaned and integrated datasets
│       └── crm_master_dataset.csv  # Final integrated dataset for analysis
│
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── 00_data_exploration.ipynb  # Initial data exploration
│   ├── 01_data_preprocessing.ipynb # Data cleaning and preprocessing
│   ├── 02_data_integration.ipynb  # Multi-table integration
│   └── 03_data_eda.ipynb         # Exploratory data analysis
│
├── src/                           # Source code for helper functions (utilities)
│
│
├── requirements.txt               # Python dependencies
└── README.md                     # Project documentation (this file)
```

## File Descriptions

### Data Files

#### `data/raw/`
- **Original CRM data files**: Contains the raw data exports from the CRM system including:
  - Opportunities data
  - Account information
  - Sales agent details
  - Product information
  - Regional office data

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

### Notebooks

#### `00_data_exploration.ipynb`
- **Purpose**: Initial exploration and understanding of the raw data
- **Contents**:
  - Loading and examining multiple data tables
  - Understanding data types and structures
  - Identifying relationships between tables
  - Initial data quality assessment
  - Preliminary statistics and observations

#### `01_data_preprocessing.ipynb`
- **Purpose**: Data cleaning and preparation
- **Contents**:
  - Handling missing values
  - Data type conversions
  - Outlier detection and treatment
  - Standardizing formats (dates, currencies, etc.)
  - Creating derived features
  - Data validation and quality checks

#### `02_data_integration.ipynb`
- **Purpose**: Combining multiple data sources into a single dataset
- **Contents**:
  - Defining join keys and relationships
  - Merging opportunities with account data
  - Integrating sales agent and manager information
  - Adding product and pricing details
  - Incorporating regional office data
  - Final validation of integrated dataset
  - Exporting master dataset to CSV

#### `03_data_eda.ipynb`
- **Purpose**: Comprehensive exploratory data analysis
- **Contents**:
  1. **Data Loading & Setup**: Import libraries and load integrated dataset
  2. **Basic Exploration**: Dataset shape, data types, descriptive statistics
  3. **Correlation Analysis**: Relationships between numerical features
  4. **Deal Stage Distribution**: Analysis of won/lost/engaging deals
  5. **Pair Plots**: Multi-dimensional relationships visualization
  6. **Temporal Analysis**: Sales trends over time
  7. **Sector Performance**: Deal values and counts by industry
  8. **Deal Value Distributions**: KDE plots and histograms
  9. **Sales Agent Performance**: Individual and team performance metrics
  10. **Regional Analysis**: Geographic performance patterns
  11. **Product Analysis**: Performance by product series
  12. **Sales Cycle Analysis**: Time from engagement to close
  13. **Company Size Analysis**: Impact of customer characteristics
  14. **Win Rate Analysis**: Success rates across dimensions
  15. **Correlation Heatmap**: Comprehensive correlation matrix
  16. **Key Insights Summary**: Business recommendations

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
3. Open Jupyter Notebook: `jupyter notebook`
4. Start with `00_data_exploration.ipynb` to understand the data
5. Follow through notebooks in sequence for complete analysis

## Last Updated
September 2025 