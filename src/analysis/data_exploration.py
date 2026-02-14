import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns


class DataExploration:
    """Class for comprehensive exploratory data analysis of the Jigsaw dataset."""
    
    def __init__(self, csv_path: str, output_dir: str = "outputs/data_exploration"):
        """
        Initialize the DataExploration class.
        
        Args:
            csv_path: Path to the training CSV file
            output_dir: Directory to save analysis outputs
        """
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.findings = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load the CSV file."""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df
    
    def analyze_basic_statistics(self) -> Dict[str, Any]:
        """Analyze basic dataset statistics."""
        print("\n=== Basic Statistics ===")
        stats = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_rows": self.df.duplicated().sum(),
        }
        
        print(f"Total Rows: {stats['total_rows']:,}")
        print(f"Total Columns: {stats['total_columns']}")
        print(f"Memory Usage: {stats['memory_usage_mb']:.2f} MB")
        print(f"Duplicate Rows: {stats['duplicate_rows']}")
        
        self.findings['basic_statistics'] = stats
        return stats
    
    def analyze_target_variable(self) -> Dict[str, Any]:
        """Analyze the target variable (toxicity)."""
        print("\n=== Target Variable Analysis ===")
        target_stats = {
            "target_mean": float(self.df['target'].mean()),
            "target_std": float(self.df['target'].std()),
            "target_min": float(self.df['target'].min()),
            "target_max": float(self.df['target'].max()),
            "target_median": float(self.df['target'].median()),
            "null_targets": int(self.df['target'].isnull().sum()),
        }
        
        # Distribution of target values
        target_bins = pd.cut(self.df['target'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        target_dist = target_bins.value_counts(sort=False).to_dict()
        target_stats['target_distribution'] = {str(k): int(v) for k, v in target_dist.items()}
        
        print(f"Target Mean: {target_stats['target_mean']:.4f}")
        print(f"Target Std: {target_stats['target_std']:.4f}")
        print(f"Target Range: [{target_stats['target_min']:.2f}, {target_stats['target_max']:.2f}]")
        print(f"Target Distribution:\n{target_bins.value_counts(sort=False)}")
        
        self.findings['target_analysis'] = target_stats
        return target_stats
    
    def analyze_toxicity_types(self) -> Dict[str, Any]:
        """Analyze different toxicity types."""
        print("\n=== Toxicity Types Analysis ===")
        toxicity_cols = ['severe_toxicity', 'obscene', 'identity_attack', 
                        'insult', 'threat', 'sexual_explicit']
        
        toxicity_stats = {}
        for col in toxicity_cols:
            if col in self.df.columns:
                toxicity_stats[col] = {
                    "mean": float(self.df[col].mean()),
                    "presence_count": int((self.df[col] > 0).sum()),
                    "presence_percentage": float((self.df[col] > 0).mean() * 100),
                }
        
        for col, stats in toxicity_stats.items():
            print(f"{col}: {stats['presence_count']:,} cases ({stats['presence_percentage']:.2f}%)")
        
        self.findings['toxicity_types'] = toxicity_stats
        return toxicity_stats
    
    def analyze_identity_attributes(self) -> Dict[str, Any]:
        """Analyze identity attribute columns."""
        print("\n=== Identity Attributes Analysis ===")
        identity_cols = ['asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian',
                        'female', 'heterosexual', 'hindu', 'homosexual_gay_or_lesbian',
                        'intellectual_or_learning_disability', 'jewish', 'latino', 'male',
                        'muslim', 'other_disability', 'other_gender', 'other_race_or_ethnicity',
                        'other_religion', 'other_sexual_orientation', 'physical_disability',
                        'psychiatric_or_mental_illness', 'transgender', 'white']
        
        identity_stats = {}
        for col in identity_cols:
            if col in self.df.columns:
                mentions = int((self.df[col] > 0).sum())
                identity_stats[col] = {
                    "mentions": mentions,
                    "percentage": float((self.df[col] > 0).mean() * 100),
                }
        
        # Sort by mentions
        sorted_identities = sorted(identity_stats.items(), key=lambda x: x[1]['mentions'], reverse=True)
        print("Top 10 Most Mentioned Identities:")
        for col, stats in sorted_identities[:10]:
            print(f"  {col}: {stats['mentions']:,} ({stats['percentage']:.2f}%)")
        
        self.findings['identity_attributes'] = identity_stats
        return identity_stats
    
    def analyze_text_statistics(self) -> Dict[str, Any]:
        """Analyze text-based statistics."""
        print("\n=== Text Statistics ===")
        text_stats = {
            "avg_comment_length": float(self.df['comment_text'].str.len().mean()),
            "median_comment_length": float(self.df['comment_text'].str.len().median()),
            "max_comment_length": int(self.df['comment_text'].str.len().max()),
            "min_comment_length": int(self.df['comment_text'].str.len().min()),
            "avg_word_count": float(self.df['comment_text'].str.split().str.len().mean()),
        }
        
        print(f"Average Comment Length: {text_stats['avg_comment_length']:.2f} characters")
        print(f"Average Word Count: {text_stats['avg_word_count']:.2f} words")
        print(f"Comment Length Range: [{text_stats['min_comment_length']}, {text_stats['max_comment_length']}]")
        
        self.findings['text_statistics'] = text_stats
        return text_stats
    
    def analyze_engagement_metrics(self) -> Dict[str, Any]:
        """Analyze engagement metrics."""
        print("\n=== Engagement Metrics ===")
        engagement_cols = ['funny', 'wow', 'sad', 'likes', 'disagree']
        
        engagement_stats = {}
        for col in engagement_cols:
            if col in self.df.columns:
                engagement_stats[col] = {
                    "mean": float(self.df[col].mean()),
                    "median": float(self.df[col].median()),
                    "max": int(self.df[col].max()),
                    "non_zero_count": int((self.df[col] > 0).sum()),
                }
        
        for col, stats in engagement_stats.items():
            print(f"{col}: mean={stats['mean']:.2f}, non-zero={stats['non_zero_count']:,}")
        
        self.findings['engagement_metrics'] = engagement_stats
        return engagement_stats
    
    def analyze_annotator_data(self) -> Dict[str, Any]:
        """Analyze annotator information."""
        print("\n=== Annotator Data ===")
        annotator_stats = {
            "toxicity_annotator_count_mean": float(self.df['toxicity_annotator_count'].mean()),
            "toxicity_annotator_count_max": int(self.df['toxicity_annotator_count'].max()),
            "identity_annotator_count_mean": float(self.df['identity_annotator_count'].mean()),
            "identity_annotator_count_max": int(self.df['identity_annotator_count'].max()),
        }
        
        print(f"Average Toxicity Annotators: {annotator_stats['toxicity_annotator_count_mean']:.2f}")
        print(f"Average Identity Annotators: {annotator_stats['identity_annotator_count_mean']:.2f}")
        
        self.findings['annotator_data'] = annotator_stats
        return annotator_stats
    
    def analyze_labels(self) -> Dict[str, Any]:
        """Analyze label distribution with toxicity threshold (>= 0.5)."""
        print("\n=== Label Analysis (Threshold: target >= 0.5) ===")
        
        # Binary classification with threshold
        threshold = 0.5
        toxic_mask = self.df['target'] >= threshold
        non_toxic_count = (~toxic_mask).sum()
        toxic_count = toxic_mask.sum()
        
        label_stats = {
            "threshold": threshold,
            "toxic_count": int(toxic_count),
            "non_toxic_count": int(non_toxic_count),
            "toxic_percentage": float(toxic_count / len(self.df) * 100),
            "non_toxic_percentage": float(non_toxic_count / len(self.df) * 100),
            "class_imbalance_ratio": float(non_toxic_count / toxic_count) if toxic_count > 0 else None,
        }
        
        print(f"Toxic (target >= {threshold}): {toxic_count:,} ({label_stats['toxic_percentage']:.2f}%)")
        print(f"Non-Toxic (target < {threshold}): {non_toxic_count:,} ({label_stats['non_toxic_percentage']:.2f}%)")
        print(f"Class Imbalance Ratio (Non-Toxic/Toxic): {label_stats['class_imbalance_ratio']:.2f}x")
        
        # Binned distribution
        print("\n=== Toxicity Score Binned Distribution ===")
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5',
                      '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        binned = pd.cut(self.df['target'], bins=bins, labels=bin_labels)
        bin_dist = binned.value_counts(sort=False).to_dict()
        
        label_stats['bin_distribution'] = {k: int(v) for k, v in bin_dist.items()}
        
        for bin_label in bin_labels:
            count = bin_dist.get(bin_label, 0)
            pct = (count / len(self.df) * 100) if len(self.df) > 0 else 0
            print(f"  {bin_label}: {count:,} ({pct:.2f}%)")
        
        self.findings['label_analysis'] = label_stats
        return label_stats
    
    def generate_visualizations(self):
        """Generate and save visualizations."""
        print("\n=== Generating Visualizations ===")
        
        # Target distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['target'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Target (Toxicity Score)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Toxicity Scores')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'target_distribution.png', dpi=100)
        plt.close()
        print("Saved: target_distribution.png")
        
        # Toxicity types comparison
        toxicity_cols = ['severe_toxicity', 'obscene', 'identity_attack', 
                        'insult', 'threat', 'sexual_explicit']
        toxicity_means = [self.df[col].mean() for col in toxicity_cols]
        
        plt.figure(figsize=(10, 6))
        plt.bar(toxicity_cols, toxicity_means, color='steelblue', edgecolor='black')
        plt.ylabel('Mean Score')
        plt.title('Mean Toxicity Type Scores')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'toxicity_types.png', dpi=100)
        plt.close()
        print("Saved: toxicity_types.png")
        
        # Comment length distribution
        plt.figure(figsize=(10, 6))
        comment_lengths = self.df['comment_text'].str.len()
        plt.hist(comment_lengths, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Comment Length (characters)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Comment Lengths')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comment_length_distribution.png', dpi=100)
        plt.close()
        print("Saved: comment_length_distribution.png")
    
    def save_findings(self):
        """Save all findings to a JSON file."""
        output_file = self.output_dir / 'findings.json'
        
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(output_file, 'w') as f:
            json.dump(self.findings, f, indent=2, cls=NumpyEncoder)
        print(f"\nSaved findings to: {output_file}")
    
    def generate_report(self):
        """Generate a comprehensive text report."""
        report_file = self.output_dir / 'analysis_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("JIGSAW UNINTENDED BIAS IN TOXICITY CLASSIFICATION - DATA EXPLORATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic Statistics
            f.write("BASIC STATISTICS\n")
            f.write("-" * 80 + "\n")
            stats = self.findings['basic_statistics']
            f.write(f"Total Rows: {stats['total_rows']:,}\n")
            f.write(f"Total Columns: {stats['total_columns']}\n")
            f.write(f"Memory Usage: {stats['memory_usage_mb']:.2f} MB\n")
            f.write(f"Duplicate Rows: {stats['duplicate_rows']}\n\n")
            
            # Target Analysis
            f.write("TARGET VARIABLE (TOXICITY SCORE)\n")
            f.write("-" * 80 + "\n")
            target = self.findings['target_analysis']
            f.write(f"Mean: {target['target_mean']:.4f}\n")
            f.write(f"Std Dev: {target['target_std']:.4f}\n")
            f.write(f"Range: [{target['target_min']:.2f}, {target['target_max']:.2f}]\n")
            f.write(f"Median: {target['target_median']:.4f}\n\n")
            
            # Toxicity Types
            f.write("TOXICITY TYPES\n")
            f.write("-" * 80 + "\n")
            for tox_type, stats in self.findings['toxicity_types'].items():
                f.write(f"{tox_type}:\n")
                f.write(f"  Mean Score: {stats['mean']:.4f}\n")
                f.write(f"  Presence: {stats['presence_count']:,} ({stats['presence_percentage']:.2f}%)\n")
            f.write("\n")
            
            # Identity Attributes
            f.write("IDENTITY ATTRIBUTES (Top 15)\n")
            f.write("-" * 80 + "\n")
            identities = sorted(self.findings['identity_attributes'].items(),
                              key=lambda x: x[1]['mentions'], reverse=True)
            for col, stats in identities[:15]:
                f.write(f"{col}: {stats['mentions']:,} mentions ({stats['percentage']:.2f}%)\n")
            f.write("\n")
            
            # Text Statistics
            f.write("TEXT STATISTICS\n")
            f.write("-" * 80 + "\n")
            text = self.findings['text_statistics']
            f.write(f"Average Comment Length: {text['avg_comment_length']:.2f} characters\n")
            f.write(f"Median Comment Length: {text['median_comment_length']:.2f} characters\n")
            f.write(f"Comment Length Range: [{text['min_comment_length']}, {text['max_comment_length']}]\n")
            f.write(f"Average Word Count: {text['avg_word_count']:.2f} words\n\n")
            
            # Engagement Metrics
            f.write("ENGAGEMENT METRICS\n")
            f.write("-" * 80 + "\n")
            for metric, stats in self.findings['engagement_metrics'].items():
                f.write(f"{metric}: mean={stats['mean']:.2f}, non-zero={stats['non_zero_count']:,}\n")
            f.write("\n")
            
            # Annotator Data
            f.write("ANNOTATOR DATA\n")
            f.write("-" * 80 + "\n")
            annotator = self.findings['annotator_data']
            f.write(f"Average Toxicity Annotators: {annotator['toxicity_annotator_count_mean']:.2f}\n")
            f.write(f"Average Identity Annotators: {annotator['identity_annotator_count_mean']:.2f}\n")
            f.write("\n")
        
        print(f"Saved report to: {report_file}")
    
    def generate_labels_report(self):
        """Generate a detailed labels report."""
        labels_file = self.output_dir / 'labels_analysis.txt'
        
        with open(labels_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("LABEL ANALYSIS REPORT - TOXICITY CLASSIFICATION\n")
            f.write("=" * 80 + "\n\n")
            
            # Binary Classification with Threshold
            f.write("BINARY CLASSIFICATION (Threshold: target >= 0.5)\n")
            f.write("-" * 80 + "\n")
            label_info = self.findings['label_analysis']
            f.write(f"Toxic Class (target >= 0.5):\n")
            f.write(f"  Count: {label_info['toxic_count']:,}\n")
            f.write(f"  Percentage: {label_info['toxic_percentage']:.2f}%\n\n")
            
            f.write(f"Non-Toxic Class (target < 0.5):\n")
            f.write(f"  Count: {label_info['non_toxic_count']:,}\n")
            f.write(f"  Percentage: {label_info['non_toxic_percentage']:.2f}%\n\n")
            
            f.write(f"CLASS IMBALANCE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Imbalance Ratio (Non-Toxic / Toxic): {label_info['class_imbalance_ratio']:.2f}x\n")
            f.write(f"Majority Class: Non-Toxic ({label_info['non_toxic_percentage']:.2f}%)\n")
            f.write(f"Minority Class: Toxic ({label_info['toxic_percentage']:.2f}%)\n\n")
            
            f.write("INTERPRETATION:\n")
            f.write("-" * 80 + "\n")
            f.write(f"This is a highly imbalanced dataset with {label_info['class_imbalance_ratio']:.1f}x more\n")
            f.write("non-toxic comments than toxic ones. This imbalance needs to be addressed during\n")
            f.write("model training through techniques such as:\n")
            f.write("  - Weighted loss functions\n")
            f.write("  - Class weighting\n")
            f.write("  - Oversampling of minority class\n")
            f.write("  - Undersampling of majority class\n")
            f.write("  - Focal loss\n\n")
            
            # Binned Distribution
            f.write("TOXICITY SCORE DISTRIBUTION (Binned)\n")
            f.write("-" * 80 + "\n")
            f.write("Bin Range          Count          Percentage     Cumulative %\n")
            f.write("-" * 80 + "\n")
            
            bin_dist = label_info['bin_distribution']
            cumulative = 0
            for bin_label in ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5',
                             '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']:
                count = bin_dist.get(bin_label, 0)
                pct = (count / len(self.df) * 100) if len(self.df) > 0 else 0
                cumulative += pct
                f.write(f"{bin_label:<15}  {count:>12,}  {pct:>11.2f}%   {cumulative:>11.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("DATASET CHARACTERISTICS SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Samples: {len(self.df):,}\n")
            f.write(f"Toxic Samples (≥0.5): {label_info['toxic_count']:,}\n")
            f.write(f"Non-Toxic Samples (<0.5): {label_info['non_toxic_count']:,}\n")
            f.write(f"\nClass Distribution Challenges:\n")
            f.write(f"  - Severe class imbalance ({label_info['class_imbalance_ratio']:.1f}x)\n")
            f.write(f"  - Continuous toxicity scores (0-1 range)\n")
            f.write(f"  - Need for stratified splitting during train/val/test\n")
            f.write(f"  - Appropriate metrics: F1, ROC-AUC, Precision-Recall curve\n")
        
        print(f"Saved labels analysis to: {labels_file}")
    
    def export_target_distribution_csv(self):
        """Export target distribution to CSV."""
        csv_file = self.output_dir / 'target_distribution.csv'
        
        label_info = self.findings['label_analysis']
        bin_dist = label_info['bin_distribution']
        
        # Prepare data for CSV
        rows = []
        cumulative = 0
        
        for bin_label in ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5',
                         '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']:
            count = bin_dist.get(bin_label, 0)
            pct = (count / len(self.df) * 100) if len(self.df) > 0 else 0
            cumulative += pct
            
            rows.append({
                'bin_range': bin_label,
                'count': count,
                'percentage': round(pct, 2),
                'cumulative_percentage': round(cumulative, 2),
                'is_toxic': 'Yes' if float(bin_label.split('-')[1]) >= 0.5 else 'No'
            })
        
        # Add summary rows
        rows.append({
            'bin_range': 'TOTAL_NON_TOXIC',
            'count': label_info['non_toxic_count'],
            'percentage': round(label_info['non_toxic_percentage'], 2),
            'cumulative_percentage': round(label_info['non_toxic_percentage'], 2),
            'is_toxic': 'No'
        })
        
        rows.append({
            'bin_range': 'TOTAL_TOXIC',
            'count': label_info['toxic_count'],
            'percentage': round(label_info['toxic_percentage'], 2),
            'cumulative_percentage': 100.0,
            'is_toxic': 'Yes'
        })
        
        # Write to CSV
        df_csv = pd.DataFrame(rows)
        df_csv.to_csv(csv_file, index=False)
        print(f"Saved target distribution CSV to: {csv_file}")
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting data exploration analysis...\n")
        
        self.load_data()
        self.analyze_basic_statistics()
        self.analyze_target_variable()
        self.analyze_toxicity_types()
        self.analyze_identity_attributes()
        self.analyze_text_statistics()
        self.analyze_engagement_metrics()
        self.analyze_annotator_data()
        self.analyze_labels()
        self.generate_visualizations()
        self.save_findings()
        self.generate_report()
        self.generate_labels_report()
        self.export_target_distribution_csv()
        
        print("\n" + "=" * 80)
        print("Analysis complete! Outputs saved to:", self.output_dir)
        print("=" * 80)


if __name__ == "__main__":
    # Run analysis
    explorer = DataExploration(
        csv_path="data/train.csv",
        output_dir="outputs/data_exploration"
    )
    explorer.run_analysis()
