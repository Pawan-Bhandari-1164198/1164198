# Utility functions for data preprocessing operations
# Modular design allows reuse across different datasets and projects

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

#######################################################################################################################
#######################################################################################################################
class DataQualityAssessment:
    """
    Comprehensive data quality assessment tools
    Provides systematic evaluation of dataset characteristics and issues
    """
    #######################################################################################################################
    @staticmethod
    def assess_table_quality(df, table_name):
        """
        Generate comprehensive quality metrics for a single table
        
        Returns dictionary with completeness, duplication, and type information
        This provides baseline metrics to track improvement throughout preprocessing
        """
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        # Data type distribution analysis
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        object_cols = len(df.select_dtypes(include=['object']).columns)
        datetime_cols = len(df.select_dtypes(include=['datetime']).columns)
        
        return {
            'table_name': table_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'completeness_pct': ((total_cells - missing_cells) / total_cells) * 100,
            'duplicate_rows': duplicate_rows,
            'numeric_columns': numeric_cols,
            'object_columns': object_cols,
            'datetime_columns': datetime_cols
        }
    #######################################################################################################################
    @staticmethod
    def identify_placeholder_patterns(df):
        """
        Detect common placeholder values that indicate missing data
        
        Many datasets use placeholder values instead of proper NaN
        Common patterns include: *, ?, N/A, NULL, empty strings, -1, 0 in contexts where 0 is impossible
        """
        placeholder_patterns = ['*', '?', 'N/A', 'NULL', 'null', 'none', 'NONE', '', ' ']
        placeholder_summary = {}
        
        for col in df.columns:
            col_placeholders = {}
            for pattern in placeholder_patterns:
                if df[col].dtype == 'object':
                    count = (df[col].astype(str).str.strip() == pattern).sum()
                    if count > 0:
                        col_placeholders[pattern] = count
            
            if col_placeholders:
                placeholder_summary[col] = col_placeholders
                
        return placeholder_summary

#######################################################################################################################
#######################################################################################################################
class DuplicateHandler:
    """
    Systematic approach to duplicate detection and removal
    Handles both exact duplicates and business logic duplicates
    """
    #######################################################################################################################
    @staticmethod
    def analyze_duplicates(df, table_name, key_columns=None):
        """
        Comprehensive duplicate analysis including complete and partial duplicates
        
        key_columns: Business-relevant columns that should be unique
        Example: opportunity_id in sales data, account names in customer data
        """
        results = {
            'table_name': table_name,
            'complete_duplicates': df.duplicated().sum(),
            'key_duplicates': 0,
            'column_duplicates': {}
        }
        
        # Business key duplicates
        if key_columns:
            for col in key_columns:
                if col in df.columns:
                    key_dups = df[col].duplicated().sum()
                    results['key_duplicates'] += key_dups
                    if key_dups > 0:
                        results['column_duplicates'][col] = key_dups
        
        # Individual column duplicate rates
        for col in df.columns:
            dup_count = df[col].duplicated().sum()
            dup_rate = (dup_count / len(df)) * 100
            results['column_duplicates'][col] = {
                'count': dup_count,
                'rate_pct': dup_rate
            }
        
        return results
    #######################################################################################################################
    @staticmethod
    def remove_duplicates(df, strategy='complete', subset_columns=None):
        """
        Remove duplicates based on specified strategy
        
        Strategies:
        - 'complete': Remove rows that are identical across all columns
        - 'subset': Remove based on specific columns (business keys)
        - 'keep_first': When duplicates exist, keep first occurrence
        - 'keep_last': When duplicates exist, keep last occurrence
        """
        original_rows = len(df)
        
        if strategy == 'complete':
            df_cleaned = df.drop_duplicates()
        elif strategy == 'subset' and subset_columns:
            df_cleaned = df.drop_duplicates(subset=subset_columns, keep='first')
        else:
            df_cleaned = df.drop_duplicates()
        
        removed_count = original_rows - len(df_cleaned)
        
        return df_cleaned, removed_count

#######################################################################################################################
#######################################################################################################################
class MissingDataAnalyzer:
    """
    Advanced missing data analysis and pattern detection
    Determines whether missing data is random, systematic, or business-logical
    """
    #######################################################################################################################    
    @staticmethod
    def analyze_missing_patterns(df, table_name):
        """
        Comprehensive missing data pattern analysis
        
        Identifies:
        - Columns with missing data and their patterns
        - Correlations between missing values across columns
        - Business context implications
        """
        missing_summary = []
        total_rows = len(df)
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / total_rows) * 100
                missing_summary.append({
                    'column': col,
                    'missing_count': missing_count,
                    'missing_percentage': missing_pct,
                    'data_type': str(df[col].dtype)
                })
        
        # Missing data correlation analysis
        # High correlation between missing values suggests systematic missing patterns
        missing_corr = None
        if missing_summary:
            missing_indicators = df.isnull().astype(int)
            missing_cols = [item['column'] for item in missing_summary]
            if len(missing_cols) > 1:
                missing_corr = missing_indicators[missing_cols].corr()
        
        return {
            'table_name': table_name,
            'missing_summary': missing_summary,
            'missing_correlation': missing_corr,
            'total_missing_cells': df.isnull().sum().sum()
        }
    
    #######################################################################################################################
    @staticmethod
    def visualize_missing_patterns(df, table_name):
        """
        Create visualizations to understand missing data patterns
        
        Generates heatmap showing missing data patterns across rows and columns
        Helps identify systematic vs random missing patterns
        """
        if df.isnull().sum().sum() == 0:
            print(f"No missing data to visualize in {table_name}")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Missing data heatmap
        missing_data = df.isnull()
        sns.heatmap(missing_data, cbar=True, ax=axes[0], cmap='viridis')
        axes[0].set_title(f'{table_name} - Missing Data Pattern')
        axes[0].set_xlabel('Columns')
        axes[0].set_ylabel('Rows')
        
        # Missing data bar chart
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        
        if len(missing_counts) > 0:
            missing_counts.plot(kind='bar', ax=axes[1])
            axes[1].set_title(f'{table_name} - Missing Data Count by Column')
            axes[1].set_xlabel('Columns')
            axes[1].set_ylabel('Missing Count')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

#######################################################################################################################
#######################################################################################################################
class ImputationStrategies:
    """
    Multiple imputation strategies based on data type and business context
    Provides both statistical and domain-aware imputation methods
    """
    #######################################################################################################################
    @staticmethod
    def numerical_imputation(series, strategy='median', group_by=None, group_data=None):
        """
        Numerical data imputation with multiple strategies
        
        Strategies:
        - 'mean': Simple mean imputation
        - 'median': Median imputation (robust to outliers)
        - 'mode': Most frequent value
        - 'group_median': Median within groups (e.g., by category)
        - 'interpolate': Linear interpolation (for time series)
        """
        if strategy == 'mean':
            return series.fillna(series.mean())
        elif strategy == 'median':
            return series.fillna(series.median())
        elif strategy == 'mode':
            mode_val = series.mode()
            return series.fillna(mode_val[0] if len(mode_val) > 0 else series.median())
        elif strategy == 'group_median' and group_by is not None and group_data is not None:
            # Group-based imputation preserves subgroup patterns
            return group_data.groupby(group_by)[series.name].transform(
                lambda x: x.fillna(x.median())
            )
        elif strategy == 'interpolate':
            return series.interpolate()
        else:
            return series.fillna(series.median())
    
    #######################################################################################################################
    @staticmethod
    def categorical_imputation(series, strategy='mode', custom_value=None):
        """
        Categorical data imputation strategies
        
        Strategies:
        - 'mode': Most frequent category
        - 'custom': User-specified value
        - 'unknown': Mark as 'Unknown' category
        """
        if strategy == 'mode':
            mode_val = series.mode()
            return series.fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')
        elif strategy == 'custom' and custom_value:
            return series.fillna(custom_value)
        elif strategy == 'unknown':
            return series.fillna('Unknown')
        else:
            return series.fillna('Unknown')
    
    #######################################################################################################################
    @staticmethod
    def business_logic_imputation(df, column, business_rules):
        """
        Apply business-specific imputation rules
        
        business_rules: Dictionary defining conditions and imputation values
        Example: {'Won': median_of_won_deals, 'Lost': 0, 'Prospecting': np.nan}
        """
        df_imputed = df.copy()
        
        for condition, impute_value in business_rules.items():
            if isinstance(condition, str) and condition in df.columns:
                # Simple column value matching
                mask = df[condition].notna() & df[column].isnull()
                df_imputed.loc[mask, column] = impute_value
            elif callable(condition):
                # Custom condition function
                mask = condition(df) & df[column].isnull()
                df_imputed.loc[mask, column] = impute_value
        
        return df_imputed[column]

#######################################################################################################################
#######################################################################################################################
class OutlierDetection:
    """
    Multiple outlier detection methods with business context consideration
    Combines statistical methods with domain knowledge
    """
    #######################################################################################################################
    @staticmethod
    def iqr_method(series, multiplier=1.5):
        """
        Interquartile Range method for outlier detection
        
        Conservative method that identifies extreme values based on quartile spread
        multiplier: Controls sensitivity (1.5 = standard, 3.0 = very conservative)
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - (multiplier * IQR)
        upper_bound = Q3 + (multiplier * IQR)
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        
        return {
            'outlier_mask': outlier_mask,
            'outlier_count': outlier_mask.sum(),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_indices': series.index[outlier_mask].tolist(),
            'outlier_values': series[outlier_mask].tolist()
        }
    
    #######################################################################################################################
    @staticmethod
    def zscore_method(series, threshold=3):
        """
        Z-score method for outlier detection
        
        Identifies values that are more than 'threshold' standard deviations from the mean
        threshold=3 captures ~99.7% of data in normal distribution
        """
        z_scores = np.abs(stats.zscore(series.dropna()))
        outlier_mask = pd.Series(False, index=series.index)
        
        valid_indices = series.dropna().index
        outlier_mask.loc[valid_indices] = z_scores > threshold
        
        return {
            'outlier_mask': outlier_mask,
            'outlier_count': outlier_mask.sum(),
            'z_scores': z_scores,
            'threshold': threshold,
            'outlier_indices': series.index[outlier_mask].tolist(),
            'outlier_values': series[outlier_mask].tolist()
        }
    
    #######################################################################################################################
    @staticmethod
    def modified_zscore_method(series, threshold=3.5):
        """
        Modified Z-score using median absolute deviation (MAD)
        
        More robust to extreme outliers than standard Z-score
        Uses median instead of mean, making it less sensitive to outliers themselves
        """
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        # Avoid division by zero
        if mad == 0:
            mad = np.median(np.abs(series - series.mean()))
        
        modified_z_scores = 0.6745 * (series - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        
        return {
            'outlier_mask': outlier_mask,
            'outlier_count': outlier_mask.sum(),
            'modified_z_scores': modified_z_scores,
            'threshold': threshold,
            'median': median,
            'mad': mad,
            'outlier_indices': series.index[outlier_mask].tolist(),
            'outlier_values': series[outlier_mask].tolist()
        }
    
    #######################################################################################################################
    @staticmethod
    def visualize_outliers(df, column, methods_results):
        """
        Create comprehensive outlier visualizations
        
        Shows distribution, boxplot, and outlier detection results
        Helps validate outlier detection methods and business decisions
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribution plot
        df[column].hist(bins=50, ax=axes[0,0], alpha=0.7)
        axes[0,0].set_title(f'{column} - Distribution')
        axes[0,0].set_xlabel(column)
        axes[0,0].set_ylabel('Frequency')
        
        # Box plot
        df[column].plot(kind='box', ax=axes[0,1])
        axes[0,1].set_title(f'{column} - Box Plot')
        axes[0,1].set_ylabel(column)
        
        # Q-Q plot for normality assessment
        stats.probplot(df[column].dropna(), dist="norm", plot=axes[1,0])
        axes[1,0].set_title(f'{column} - Q-Q Plot (Normality Check)')
        
        # Outlier comparison
        method_names = list(methods_results.keys())
        outlier_counts = [results['outlier_count'] for results in methods_results.values()]
        
        axes[1,1].bar(method_names, outlier_counts)
        axes[1,1].set_title(f'{column} - Outlier Count by Method')
        axes[1,1].set_ylabel('Number of Outliers')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

#######################################################################################################################
#######################################################################################################################
class DataTransformation:
    """
    Data transformation utilities for preprocessing
    Handles type conversion, scaling, and encoding operations
    """
    #######################################################################################################################
    @staticmethod
    def standardize_column_names(df):
        """
        Standardize column names for consistency
        
        Converts to lowercase, replaces spaces with underscores
        Ensures consistent naming convention across datasets
        """
        df_transformed = df.copy()
        
        new_columns = []
        changes_made = False
        
        for col in df_transformed.columns:
            # Convert to lowercase and replace problematic characters
            new_col = col.lower().strip()
            new_col = new_col.replace(' ', '_').replace('-', '_').replace('.', '_')
            new_col = ''.join(char for char in new_col if char.isalnum() or char == '_')
            
            new_columns.append(new_col)
            if new_col != col:
                changes_made = True
        
        df_transformed.columns = new_columns
        
        return df_transformed, changes_made

    #######################################################################################################################
    @staticmethod
    def convert_data_types(df, type_mappings):
        """
        Convert data types based on mappings
        
        type_mappings: Dictionary mapping column names to target types
        Example: {'date_column': 'datetime64', 'numeric_column': 'float64'}
        """
        df_converted = df.copy()
        conversion_log = {}
        
        for column, target_type in type_mappings.items():
            if column in df_converted.columns:
                original_type = str(df_converted[column].dtype)
                
                try:
                    if target_type == 'datetime64':
                        df_converted[column] = pd.to_datetime(df_converted[column], errors='coerce')
                    elif target_type in ['float64', 'int64']:
                        df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
                    else:
                        df_converted[column] = df_converted[column].astype(target_type)
                    
                    conversion_log[column] = {
                        'from': original_type,
                        'to': str(df_converted[column].dtype),
                        'success': True
                    }
                except Exception as e:
                    conversion_log[column] = {
                        'from': original_type,
                        'to': original_type,
                        'success': False,
                        'error': str(e)
                    }
        
        return df_converted, conversion_log

#######################################################################################################################
#######################################################################################################################
class QualityReporting:
    """
    Generate comprehensive quality reports and comparisons
    Tracks preprocessing impact and provides audit trail
    """
    
    #######################################################################################################################
    @staticmethod
    def compare_quality_metrics(before_metrics, after_metrics):
        """
        Compare quality metrics before and after preprocessing
        
        Generates improvement summary and identifies areas of change
        """
        comparison_report = []
        
        for before, after in zip(before_metrics, after_metrics):
            table_name = before['table_name']
            
            improvement = {
                'table_name': table_name,
                'completeness_improvement': after['completeness_pct'] - before['completeness_pct'],
                'missing_reduction': before['missing_cells'] - after['missing_cells'],
                'duplicate_reduction': before['duplicate_rows'] - after['duplicate_rows'],
                'before_completeness': before['completeness_pct'],
                'after_completeness': after['completeness_pct']
            }
            
            comparison_report.append(improvement)
        
        return comparison_report
    
    #######################################################################################################################
    @staticmethod
    def generate_preprocessing_report(comparison_data, output_path=None):
        """
        Generate comprehensive preprocessing report
        
        Creates detailed report of all preprocessing actions and their impact
        """
        report_lines = []
        report_lines.append("DATA PREPROCESSING QUALITY REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary statistics
        total_missing_reduced = sum(item['missing_reduction'] for item in comparison_data)
        total_duplicates_reduced = sum(item['duplicate_reduction'] for item in comparison_data)
        avg_completeness_improvement = sum(item['completeness_improvement'] for item in comparison_data) / len(comparison_data)
        
        report_lines.append("OVERALL IMPROVEMENTS:")
        report_lines.append(f"Total missing values addressed: {total_missing_reduced:,}")
        report_lines.append(f"Total duplicate rows removed: {total_duplicates_reduced:,}")
        report_lines.append(f"Average completeness improvement: {avg_completeness_improvement:.2f}%")
        report_lines.append("")
        
        # Table-by-table breakdown
        report_lines.append("TABLE-BY-TABLE BREAKDOWN:")
        report_lines.append("-" * 40)
        
        for item in comparison_data:
            report_lines.append(f"\n{item['table_name'].upper()}:")
            report_lines.append(f"  Completeness: {item['before_completeness']:.1f}% → {item['after_completeness']:.1f}%")
            report_lines.append(f"  Missing values reduced: {item['missing_reduction']:,}")
            report_lines.append(f"  Duplicates removed: {item['duplicate_reduction']:,}")
            
            if item['completeness_improvement'] > 0:
                report_lines.append(f"  Improvement: +{item['completeness_improvement']:.1f}%")
            else:
                report_lines.append(f"  No missing data improvements needed")
        
        report_content = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
        
        return report_content

#######################################################################################################################
# FUNCTION-BASED WRAPPERS FOR NOTEBOOK COMPATIBILITY
#######################################################################################################################

def assess_table_quality(df, table_name):
    """Function wrapper for DataQualityAssessment.assess_table_quality"""
    return DataQualityAssessment.assess_table_quality(df, table_name)

def identify_placeholder_patterns(df):
    """Function wrapper for DataQualityAssessment.identify_placeholder_patterns"""
    return DataQualityAssessment.identify_placeholder_patterns(df)

def analyze_duplicates(df, table_name, key_columns=None):
    """Function wrapper for DuplicateHandler.analyze_duplicates"""
    return DuplicateHandler.analyze_duplicates(df, table_name, key_columns)

def remove_duplicates(df, strategy='complete', subset_columns=None):
    """Function wrapper for DuplicateHandler.remove_duplicates"""
    return DuplicateHandler.remove_duplicates(df, strategy, subset_columns)

def analyze_missing_patterns(df, table_name):
    """Function wrapper for MissingDataAnalyzer.analyze_missing_patterns"""
    return MissingDataAnalyzer.analyze_missing_patterns(df, table_name)

def visualize_missing_patterns(df, table_name):
    """Function wrapper for MissingDataAnalyzer.visualize_missing_patterns"""
    return MissingDataAnalyzer.visualize_missing_patterns(df, table_name)

def numerical_imputation(series, strategy='median', group_by=None, group_data=None):
    """Function wrapper for ImputationStrategies.numerical_imputation"""
    return ImputationStrategies.numerical_imputation(series, strategy, group_by, group_data)

def categorical_imputation(series, strategy='mode', custom_value=None):
    """Function wrapper for ImputationStrategies.categorical_imputation"""
    return ImputationStrategies.categorical_imputation(series, strategy, custom_value)

def iqr_outlier_detection(series, multiplier=1.5):
    """Function wrapper for OutlierDetection.iqr_method"""
    return OutlierDetection.iqr_method(series, multiplier)

def zscore_outlier_detection(series, threshold=3):
    """Function wrapper for OutlierDetection.zscore_method"""
    return OutlierDetection.zscore_method(series, threshold)

def modified_zscore_outlier_detection(series, threshold=3.5):
    """Function wrapper for OutlierDetection.modified_zscore_method"""
    return OutlierDetection.modified_zscore_method(series, threshold)

def visualize_outliers(df, column, methods_results):
    """Function wrapper for OutlierDetection.visualize_outliers"""
    return OutlierDetection.visualize_outliers(df, column, methods_results)

def standardize_column_names(df):
    """Function wrapper for DataTransformation.standardize_column_names"""
    return DataTransformation.standardize_column_names(df)

def convert_data_types(df, type_mappings):
    """Function wrapper for DataTransformation.convert_data_types"""
    return DataTransformation.convert_data_types(df, type_mappings)