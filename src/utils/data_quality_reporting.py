import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataQualityReport:
    """Comprehensive data quality assessment and reporting for renewable energy data"""

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize data quality reporter

        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Quality thresholds
        self.thresholds = {
            'max_missing_ratio': 0.05,    # 5% max missing data
            'max_outlier_ratio': 0.1,     # 10% max outliers
            'min_data_completeness': 0.9, # 90% min completeness
            'max_cv_capacity_factor': 0.5, # Max coefficient of variation for capacity factor
            'min_time_coverage': 0.8,     # 80% min time coverage
            'max_duplicates_ratio': 0.01  # 1% max duplicates
        }

        # Quality metrics cache
        self._computed_metrics = {}

    def generate_comprehensive_report(self, data: pd.DataFrame,
                                    dataset_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report

        Args:
            data: DataFrame to analyze
            dataset_info: Optional metadata about the dataset

        Returns:
            Comprehensive quality report dictionary
        """
        report = {
            'dataset_info': dataset_info or {},
            'timestamp': pd.Timestamp.now().isoformat(),
            'summary': {},
            'detailed_metrics': {},
            'quality_score': {},
            'recommendations': []
        }

        # Basic dataset info
        if dataset_info:
            report['dataset_info'] = dataset_info

        # Run all quality checks
        try:
            # Core quality metrics
            report['detailed_metrics']['completeness'] = self._analyze_completeness(data)
            report['detailed_metrics']['accuracy'] = self._analyze_accuracy(data)
            report['detailed_metrics']['consistency'] = self._analyze_consistency(data)
            report['detailed_metrics']['timeliness'] = self._analyze_timeliness(data)
            report['detailed_metrics']['validity'] = self._analyze_validity(data)

            # Domain-specific quality for renewable energy
            report['detailed_metrics']['renewable_energy_quality'] = self._analyze_renewable_energy_quality(data)

            # Calculate overall quality score
            report['quality_score'] = self._calculate_overall_quality_score(report['detailed_metrics'])

            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(report)

            # Create summary
            report['summary'] = self._create_summary(report)

        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            report['error'] = str(e)
            report['quality_score']['overall'] = 0
            report['recommendations'] = ["Data quality assessment failed"]

        # Cache computed metrics
        self._computed_metrics[id(data)] = report

        return report

    def _analyze_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data completeness"""
        completeness = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': {},
            'completeness_ratios': {},
            'overall_completeness': 0
        }

        for column in data.columns:
            missing_count = data[column].isnull().sum()
            completeness['missing_values'][column] = int(missing_count)
            completeness['completeness_ratios'][column] = 1 - (missing_count / len(data))

            if data[column].dtype in ['int64', 'float64']:
                # Check for non-finite values (NaN, inf)
                non_finite_count = (~np.isfinite(data[column].astype(float))).sum()
                completeness['missing_values'][f"{column}_non_finite"] = int(non_finite_count)

        # Overall completeness
        total_cells = completeness['total_rows'] * completeness['total_columns']
        total_missing = sum(completeness['missing_values'].values())
        completeness['overall_completeness'] = 1 - (total_missing / total_cells) if total_cells > 0 else 0

        # Flag columns with poor completeness
        completeness['poor_completeness_columns'] = [
            col for col, ratio in completeness['completeness_ratios'].items()
            if ratio < self.thresholds['min_data_completeness']
        ]

        return completeness

    def _analyze_accuracy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data accuracy through statistical validation"""
        accuracy = {
            'outlier_analysis': {},
            'distribution_analysis': {},
            'plausibility_checks': {},
            'statistical_summary': {}
        }

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            column_data = data[column].dropna()

            if len(column_data) == 0:
                continue

            # Outlier detection using IQR method
            Q1 = column_data.quantile(0.25)
            Q3 = column_data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
            outlier_ratio = len(outliers) / len(column_data) if len(column_data) > 0 else 0

            accuracy['outlier_analysis'][column] = {
                'outlier_count': len(outliers),
                'outlier_ratio': outlier_ratio,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'over_threshold': outlier_ratio > self.thresholds['max_outlier_ratio']
            }

            # Basic statistical summary
            accuracy['statistical_summary'][column] = {
                'mean': float(column_data.mean()),
                'std': float(column_data.std()),
                'min': float(column_data.min()),
                'max': float(column_data.max()),
                'median': float(column_data.median()),
                'skewness': float(column_data.skew()) if not isinstance(column_data.skew(), complex) else 0.0,
                'kurtosis': float(column_data.kurtosis()) if not isinstance(column_data.kurtosis(), complex) else 0.0
            }

        # Plausibility checks (domain-specific)
        accuracy['plausibility_checks'] = self._perform_plausibility_checks(data, numeric_columns)

        return accuracy

    def _analyze_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data consistency"""
        consistency = {
            'duplicate_analysis': {},
            'temporal_consistency': {},
            'logical_consistency': {},
            'data_type_consistency': {}
        }

        # Duplicate analysis
        duplicate_info = {}
        for column in data.columns:
            duplicates = data[column].duplicated().sum()
            duplicate_info[column] = {
                'duplicate_count': int(duplicates),
                'duplicate_ratio': duplicates / len(data),
                'over_threshold': (duplicates / len(data)) > self.thresholds['max_duplicates_ratio']
            }

        # Row-level duplicates
        row_duplicates = data.duplicated().sum()
        duplicate_info['_rows'] = {
            'duplicate_count': int(row_duplicates),
            'duplicate_ratio': row_duplicates / len(data),
            'over_threshold': (row_duplicates / len(data)) > self.thresholds['max_duplicates_ratio']
        }

        # Temporal consistency (if datetime index exists)
        if hasattr(data.index, 'inferred_freq') and data.index.dtype.kind in 'M':
            try:
                time_diff = data.index.to_series().diff().dropna()
                mode_diff = time_diff.mode()
                if len(mode_diff) > 0:
                    expected_gap = mode_diff.iloc[0]
                    temporal_inconsistencies = len(time_diff) - (time_diff == expected_gap).sum()
                    consistency['temporal_consistency'] = {
                        'expected_gap': str(expected_gap),
                        'mode_gap': str(mode_diff.iloc[0]),
                        'inconsistent_timestamps': temporal_inconsistencies,
                        'consistency_ratio': 1 - (temporal_inconsistencies / len(time_diff))
                    }
            except Exception as e:
                logger.warning(f"Temporal consistency analysis failed: {e}")
                consistency['temporal_consistency'] = {'analysis_failed': str(e)}

        consistency['duplicate_analysis'] = duplicate_info

        return consistency

    def _analyze_timeliness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data timeliness"""
        timeliness = {
            'time_coverage': {},
            'data_freshness': {},
            'temporal_gaps': {},
            'update_frequency': {}
        }

        if hasattr(data.index, 'dtype') and data.index.dtype.kind in 'M':
            # Analyze time coverage
            index_series = pd.Series(data.index)
            time_range = index_series.max() - index_series.min()

            if hasattr(index_series, 'inferred_freq') and index_series.inferred_freq:
                expected_points = len(pd.date_range(start=index_series.min(),
                                                  end=index_series.max(),
                                                  freq=index_series.inferred_freq))
                actual_points = len(index_series)
                coverage_ratio = actual_points / expected_points if expected_points > 0 else 0

                timeliness['time_coverage'] = {
                    'expected_points': expected_points,
                    'actual_points': actual_points,
                    'coverage_ratio': coverage_ratio,
                    'adequate_coverage': coverage_ratio >= self.thresholds['min_time_coverage']
                }

            # Calculate temporal gaps
            gaps = index_series.diff().fillna(pd.Timedelta(0))
            large_gaps = gaps[gaps > pd.Timedelta(days=1)]  # Gaps larger than 1 day

            timeliness['temporal_gaps'] = {
                'total_gaps': len(large_gaps),
                'max_gap': str(large_gaps.max()) if len(large_gaps) > 0 else None,
                'average_gap': str(gaps.mean()) if len(gaps) > 0 else None
            }
        else:
            timeliness['time_coverage'] = {'analysis_not_applicable': 'No datetime index detected'}

        return timeliness

    def _analyze_validity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data validity according to domain rules"""
        validity = {
            'range_validation': {},
            'format_validation': {},
            'logical_validation': {},
            'domain_rules': {}
        }

        # Basic range validation for different categories
        validation_rules = {
            'solar_irradiance': {'min': 0, 'max': 1200, 'unit': 'W/m²'},
            'wind_speed': {'min': 0, 'max': 100, 'unit': 'm/s'},
            'capacity_factor': {'min': 0, 'max': 1.5, 'unit': 'fraction'},
            'temperature': {'min': -50, 'max': 80, 'unit': '°C'},
            'pressure': {'min': 0, 'max': 5, 'unit': 'bar'},
            'efficiency': {'min': 0, 'max': 1.0, 'unit': 'fraction'},
            'latitude': {'min': -90, 'max': 90, 'unit': 'degrees'},
            'longitude': {'min': -180, 'max': 180, 'unit': 'degrees'},
            'power_generation': {'min': 0, 'max': None, 'unit': 'MW'},
            'energy_production': {'min': 0, 'max': None, 'unit': 'MWh'}
        }

        for column in data.columns:
            column_lower = column.lower()

            # Match column names to validation rules
            for rule_key, rule in validation_rules.items():
                if rule_key in column_lower:
                    col_data = data[column].dropna()
                    invalid_count = 0

                    if rule['min'] is not None:
                        invalid_count += (col_data < rule['min']).sum()
                    if rule['max'] is not None:
                        invalid_count += (col_data > rule['max']).sum()

                    validity['range_validation'][column] = {
                        'rule_applied': rule_key,
                        'invalid_count': int(invalid_count),
                        'invalid_ratio': invalid_count / len(col_data) if len(col_data) > 0 else 0,
                        'rule_bounds': rule,
                        'data_range': {
                            'min': float(col_data.min()) if len(col_data) > 0 else None,
                            'max': float(col_data.max()) if len(col_data) > 0 else None
                        }
                    }
                    break

        return validity

    def _analyze_renewable_energy_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze renewable energy specific quality metrics"""
        re_quality = {
            'capacity_factor_analysis': {},
            'intermittency_analysis': {},
            'resource_variability': {},
            'data_resolution_quality': {}
        }

        # Capacity factor analysis
        cf_columns = [col for col in data.columns if 'capacity_factor' in col.lower()]
        for col in cf_columns:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                cv_cf = col_data.std() / col_data.mean() if col_data.mean() > 0 else 0
                re_quality['capacity_factor_analysis'][col] = {
                    'mean_cf': float(col_data.mean()),
                    'std_cf': float(col_data.std()),
                    'coefficient_of_variation': cv_cf,
                    'high_variability': cv_cf > self.thresholds['max_cv_capacity_factor'],
                    'cf_range': {'min': float(col_data.min()), 'max': float(col_data.max())}
                }

        # Intermittency analysis (rapid changes)
        power_columns = [col for col in data.columns if any(keyword in col.lower()
                        for keyword in ['power', 'generation', 'electricity'])]
        for col in power_columns:
            col_data = data[col].dropna()
            if len(col_data) > 10:  # Need sufficient data for change analysis
                changes = col_data.pct_change().dropna()
                high_changes = changes.abs()[changes.abs() > 0.5].count()  # >50% change

                re_quality['intermittency_analysis'][col] = {
                    'rapid_change_events': int(high_changes),
                    'max_change_ratio': float(changes.abs().max()) if len(changes) > 0 else 0,
                    'avg_change_rate': float(changes.abs().mean()) if len(changes) > 0 else 0,
                    'high_intermittency': high_changes / len(changes) > 0.1
                }

        return re_quality

    def _calculate_overall_quality_score(self, detailed_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality score from detailed metrics"""
        scores = {}
        weights = {
            'completeness': 0.25,
            'accuracy': 0.20,
            'consistency': 0.20,
            'timeliness': 0.15,
            'validity': 0.20
        }

        # Completion score
        if 'completeness' in detailed_metrics:
            completeness = detailed_metrics['completeness']
            scores['completeness'] = completeness['overall_completeness']

        # Accuracy score
        if 'accuracy' in detailed_metrics:
            accuracy = detailed_metrics['accuracy']
            outlier_ratios = [v['outlier_ratio'] for v in accuracy.get('outlier_analysis', {}).values()]
            scores['accuracy'] = 1 - np.mean(outlier_ratios) if outlier_ratios else 0

        # Consistency score
        if 'consistency' in detailed_metrics:
            consistency = detailed_metrics['consistency']
            duplicate_ratios = [v['duplicate_ratio'] for v in consistency.get('duplicate_analysis', {}).values()]
            scores['consistency'] = 1 - np.mean(duplicate_ratios) if duplicate_ratios else 0

        # Timeliness score
        if 'timeliness' in detailed_metrics:
            timeliness = detailed_metrics['timeliness']
            time_coverage = timeliness.get('time_coverage', {}).get('coverage_ratio', 1)
            scores['timeliness'] = time_coverage

        # Validity score
        if 'validity' in detailed_metrics:
            validity = detailed_metrics['validity']
            invalid_ratios = []
            for section in ['range_validation']:
                if section in validity:
                    section_indicators = [v['invalid_ratio'] for v in validity[section].values()]
                    invalid_ratios.extend(section_indicators)

            scores['validity'] = 1 - np.mean(invalid_ratios) if invalid_ratios else 0

        # Overall weighted score
        overall_score = 0
        total_weight = 0
        for metric, weight in weights.items():
            if metric in scores:
                overall_score += scores[metric] * weight
                total_weight += weight

        if total_weight > 0:
            overall_score /= total_weight

        quality_score = {
            'overall': overall_score,
            'individual_scores': scores,
            'weights': weights,
            'quality_grade': self._get_quality_grade(overall_score)
        }

        return quality_score

    def _create_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Create concise summary of the quality report"""
        summary = {
            'overall_quality_score': report['quality_score']['overall'],
            'quality_grade': report['quality_score']['quality_grade'],
            'total_issues': 0,
            'critical_issues': 0,
            'major_issues': 0
        }

        # Count issues from detailed metrics
        if 'detailed_metrics' in report:
            metrics = report['detailed_metrics']

            # Critical issues
            if 'completeness' in metrics:
                completeness = metrics['completeness']
                if completeness['overall_completeness'] < self.thresholds['min_data_completeness']:
                    summary['critical_issues'] += 1

                poor_columns = len(completeness.get('poor_completeness_columns', []))
                if poor_columns > 0:
                    summary['major_issues'] += poor_columns

            if 'accuracy' in metrics:
                accuracy = metrics['accuracy']
                outlier_issues = sum(1 for v in accuracy.get('outlier_analysis', {}).values()
                                   if v.get('over_threshold', False))
                summary['major_issues'] += outlier_issues

        summary['total_issues'] = summary['critical_issues'] + summary['major_issues']

        return summary

    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade"""
        if score >= 0.9:
            return 'A (Excellent)'
        elif score >= 0.8:
            return 'B (Good)'
        elif score >= 0.7:
            return 'C (Fair)'
        elif score >= 0.6:
            return 'D (Poor)'
        else:
            return 'F (Very Poor)'

    def _perform_plausibility_checks(self, data: pd.DataFrame,
                                   numeric_columns: List[str]) -> Dict[str, Any]:
        """Perform domain-specific plausibility checks"""
        checks = {
            'logical_constraints': {},
            'physical_constraints': {},
            'temporal_constraints': {}
        }

        # Check for logical constraints
        for col in numeric_columns:
            # Negative values where they shouldn't exist
            if any(keyword in col.lower() for keyword in ['generation', 'power', 'energy', 'capacity_factor']):
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    checks['logical_constraints'][col] = {
                        'issue': 'negative_values',
                        'count': int(negative_count),
                        'recommended_action': 'Remove or correct negative values'
                    }

            # Physically implausible values
            if 'capacity_factor' in col.lower():
                cf_range = ((data[col] < 0) | (data[col] > 1.5)).sum()  # Allow slight over-performance
                if cf_range > 0:
                    checks['physical_constraints'][col] = {
                        'issue': 'implausible_capacity_factor',
                        'count': int(cf_range),
                        'recommended_action': 'Check capacity factor bounds (0-1.5 typically)'
                    }

        return checks

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on quality analysis"""
        recommendations = []
        metrics = report.get('detailed_metrics', {})
        quality_score = report.get('quality_score', {}).get('overall', 0)

        # Base recommendations based on overall score
        if quality_score < 0.7:
            recommendations.append("Overall data quality is low. Consider comprehensive data cleaning.")

        # Completion-based recommendations
        if 'completeness' in metrics:
            completeness = metrics['completeness']
            if completeness['overall_completeness'] < self.thresholds['min_data_completeness']:
                recommendations.append("Low data completeness detected. Consider data imputation or gap filling strategies.")

            poor_columns = completeness.get('poor_completeness_columns', [])
            if poor_columns:
                recommendations.append(f"Columns with poor completeness: {', '.join(poor_columns)}. Consider replacing or removing these columns.")

        # Accuracy-based recommendations
        if 'accuracy' in metrics:
            accuracy = metrics['accuracy']
            high_outlier_count = sum(1 for v in accuracy.get('outlier_analysis', {}).values()
                                   if v.get('over_threshold', False))
            if high_outlier_count > 0:
                recommendations.append(f"High outlier count detected in {high_outlier_count} columns. Consider outlier treatment (removal, capping, or transformation).")

        # Consistency recommendations
        if 'consistency' in metrics:
            consistency = metrics['consistency']
            duplicate_over_threshold = [
                col for col, info in consistency.get('duplicate_analysis', {}).items()
                if info.get('over_threshold', False)
            ]
            if duplicate_over_threshold:
                recommendations.append(f"High duplicate ratio in columns: {', '.join(duplicate_over_threshold)}. Consider deduplication strategies.")

        # Timeliness recommendations
        if 'timeliness' in metrics:
            timeliness = metrics['timeliness']
            time_coverage = timeliness.get('time_coverage', {}).get('adequate_coverage', True)
            if not time_coverage:
                recommendations.append("Inadequate time coverage detected. Consider data interpolation or historical data supplementation.")

        # Domain-specific recommendations
        if 'renewable_energy_quality' in metrics:
            re_quality = metrics['renewable_energy_quality']

            # Capacity factor recommendations
            cf_analysis = re_quality.get('capacity_factor_analysis', {})
            high_cv_count = sum(1 for v in cf_analysis.values() if v.get('high_variability', False))
            if high_cv_count > 0:
                recommendations.append(f"High capacity factor variability detected in {high_cv_count} datasets. Consider temporal smoothing or quality control.")

            # Intermittency recommendations
            intermittency = re_quality.get('intermittency_analysis', {})
            high_intermittency_count = sum(1 for v in intermittency.values() if v.get('high_intermittency', False))
            if high_intermittency_count > 0:
                recommendations.append(f"High intermittency detected in {high_intermittency_count} datasets. Consider energy storage or smoothing techniques.")

        # Return sorted by priority (add priority system if needed)
        return list(set(recommendations))  # Remove duplicates

    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None,
                   format: str = 'json') -> str:
        """
        Save quality report to file

        Args:
            report: Quality report dictionary
            filename: Optional filename (auto-generated if None)
            format: Report format ('json', 'markdown', 'html')

        Returns:
            Path to saved report file
        """
        if filename is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data_quality_report_{timestamp}"

        if format == 'json':
            file_path = self.output_dir / f"{filename}.json"
            with open(file_path, 'w') as f:
                pd.Series(report).to_json(f, indent=2)
        elif format == 'markdown':
            file_path = self.output_dir / f"{filename}.md"
            with open(file_path, 'w') as f:
                self._write_markdown_report(f, report)
        elif format == 'html':
            file_path = self.output_dir / f"{filename}.html"
            with open(file_path, 'w') as f:
                self._write_html_report(f, report)
        else:
            raise ValueError(f"Unsupported report format: {format}")

        return str(file_path)

    def _write_markdown_report(self, file, report):
        """Write quality report in Markdown format"""
        file.write("# Data Quality Report\n\n")
        file.write(f"**Generated:** {report['timestamp']}\n\n")

        # Summary section
        summary = report.get('summary', {})
        file.write("## Summary\n\n")
        file.write(f"**Overall Quality Score:** {summary.get('overall_quality_score', 'N/A')}\n")
        file.write(f"**Quality Grade:** {summary.get('quality_grade', 'N/A')}\n")
        file.write(f"**Total Issues:** {summary.get('total_issues', 0)}\n\n")

        # Key metrics
        file.write("## Key Metrics\n\n")
        quality_score = report.get('quality_score', {})
        if 'individual_scores' in quality_score:
            scores = quality_score['individual_scores']
            for metric, score in scores.items():
                file.write(f"- **{metric.title()}:** {score:.2f}\n")
            file.write(f"- **Overall:** {quality_score.get('overall', 0):.2f}\n\n")

        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            file.write("## Recommendations\n\n")
            for i, rec in enumerate(recommendations, 1):
                file.write(f"{i}. {rec}\n")

    def _write_html_report(self, file, report):
        """Write quality report in HTML format"""
        file.write("<!DOCTYPE html>\n<html>\n<head>\n<title>Data Quality Report</title>\n</head>\n<body>\n")
        file.write("<h1>Data Quality Report</h1>\n")
        file.write(f"<p><strong>Generated:</strong> {report['timestamp']}</p>\n")

        # Summary
        summary = report.get('summary', {})
        file.write("<h2>Summary</h2>\n")
        file.write("<ul>\n")
        file.write(f"<li><strong>Overall Quality Score:</strong> {summary.get('overall_quality_score', 'N/A')}</li>\n")
        file.write(f"<li><strong>Quality Grade:</strong> {summary.get('quality_grade', 'N/A')}</li>\n")
        file.write(f"<li><strong>Total Issues:</strong> {summary.get('total_issues', 0)}</li>\n")
        file.write("</ul>\n")

        # Key metrics
        quality_score = report.get('quality_score', {})
        if 'individual_scores' in quality_score:
            file.write("<h2>Key Metrics</h2>\n")
            file.write("<ul>\n")
            scores = quality_score['individual_scores']
            for metric, score in scores.items():
                file.write(f"<li><strong>{metric.title()}:</strong> {score:.2f}</li>\n")
            file.write(f"<li><strong>Overall:</strong> {quality_score.get('overall', 0):.2f}</li>\n")
            file.write("</ul>\n")

        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            file.write("<h2>Recommendations</h2>\n")
            file.write("<ol>\n")
            for rec in recommendations:
                file.write(f"<li>{rec}</li>\n")
            file.write("</ol>\n")

        file.write("</body>\n</html>\n")

    def _perform_plausibility_checks(self, data: pd.DataFrame,
                                   numeric_columns: List[str]) -> Dict[str, Any]:
        """Perform domain-specific plausibility checks"""
        checks = {
            'logical_constraints': {},
            'physical_constraints': {},
            'temporal_constraints': {}
        }

        # Check for logical constraints
        for col in numeric_columns:
            # Negative values where they shouldn't exist
            if any(keyword in col.lower() for keyword in ['generation', 'power', 'energy', 'capacity_factor']):
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    checks['logical_constraints'][col] = {
                        'issue': 'negative_values',
                        'count': int(negative_count),
                        'recommended_action': 'Remove or correct negative values'
                    }

            # Physically implausible values
            if 'capacity_factor' in col.lower():
                cf_range = ((data[col] < 0) | (data[col] > 1.5)).sum()  # Allow slight over-performance
                if cf_range > 0:
                    checks['physical_constraints'][col] = {
                        'issue': 'implausible_capacity_factor',
                        'count': int(cf_range),
                        'recommended_action': 'Check capacity factor bounds (0-1.5 typically)'
                    }

        return checks

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on quality analysis"""
        recommendations = []
        metrics = report.get('detailed_metrics', {})
        quality_score = report.get('quality_score', {}).get('overall', 0)

        # Base recommendations based on overall score
        if quality_score < 0.7:
            recommendations.append("Overall data quality is low. Consider comprehensive data cleaning.")

        # Completion-based recommendations
        if 'completeness' in metrics:
            completeness = metrics['completeness']
            if completeness['overall_completeness'] < self.thresholds['min_data_completeness']:
                recommendations.append("Low data completeness detected. Consider data imputation or gap filling strategies.")

            poor_columns = completeness.get('poor_completeness_columns', [])
            if poor_columns:
                recommendations.append(f"Columns with poor completeness: {', '.join(poor_columns)}. Consider replacing or removing these columns.")

        # Accuracy-based recommendations
        if 'accuracy' in metrics:
            accuracy = metrics['accuracy']
            high_outlier_count = sum(1 for v in accuracy.get('outlier_analysis', {}).values()
                                   if v.get('over_threshold', False))
            if high_outlier_count > 0:
                recommendations.append(f"High outlier count detected in {high_outlier_count} columns. Consider outlier treatment (removal, capping, or transformation).")

        # Consistency recommendations
        if 'consistency' in metrics:
            consistency = metrics['consistency']
            duplicate_over_threshold = [
                col for col, info in consistency.get('duplicate_analysis', {}).items()
                if info.get('over_threshold', False)
            ]
            if duplicate_over_threshold:
                recommendations.append(f"High duplicate ratio in columns: {', '.join(duplicate_over_threshold)}. Consider deduplication strategies.")

        # Timeliness recommendations
        if 'timeliness' in metrics:
            timeliness = metrics['timeliness']
            time_coverage = timeliness.get('time_coverage', {}).get('adequate_coverage', True)
            if not time_coverage:
                recommendations.append("Inadequate time coverage detected. Consider data interpolation or historical data supplementation.")

        # Domain-specific recommendations
        if 'renewable_energy_quality' in metrics:
            re_quality = metrics['renewable_energy_quality']

            # Capacity factor recommendations
            cf_analysis = re_quality.get('capacity_factor_analysis', {})
            high_cv_count = sum(1 for v in cf_analysis.values() if v.get('high_variability', False))
            if high_cv_count > 0:
                recommendations.append(f"High capacity factor variability detected in {high_cv_count} datasets. Consider temporal smoothing or quality control.")

            # Intermittency recommendations
            intermittency = re_quality.get('intermittency_analysis', {})
            high_intermittency_count = sum(1 for v in intermittency.values() if v.get('high_intermittency', False))
            if high_intermittency_count > 0:
                recommendations.append(f"High intermittency detected in {high_intermittency_count} datasets. Consider energy storage or smoothing techniques.")

        # Return sorted by priority (add priority system if needed)
        return list(set(recommendations))  # Remove duplicates