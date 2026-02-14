"""
Boundary Case Handler - Tools for managing uncertain labels [0.4, 0.5].

This module provides tools to:
1. Identify boundary cases with uncertain labels
2. Filter them out for clean training
3. Export them for LLM re-labeling
4. Apply uncertainty-aware training strategies
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any


class BoundaryCaseHandler:
    """Handle boundary cases with uncertain labels."""
    
    def __init__(self, csv_path: str, output_dir: str = "outputs/data_exploration", 
                 boundary_lower: float = 0.4, boundary_upper: float = 0.5):
        """
        Initialize the handler.
        
        Args:
            csv_path: Path to training CSV
            output_dir: Output directory
            boundary_lower: Lower threshold for boundary cases
            boundary_upper: Upper threshold for boundary cases
        """
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.boundary_lower = boundary_lower
        self.boundary_upper = boundary_upper
        self.df = None
        self.boundary_cases = None
        self.threshold = 0.5
    
    def load_data(self) -> pd.DataFrame:
        """Load the CSV file."""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        return self.df
    
    def identify_boundary_cases(self) -> pd.DataFrame:
        """Identify boundary cases with uncertain labels."""
        print(f"\nIdentifying boundary cases in range [{self.boundary_lower}, {self.boundary_upper}]...")
        
        # Group by comment_text
        comment_groups = self.df.groupby('comment_text').agg({
            'target': ['count', 'min', 'max', 'mean'],
        }).reset_index()
        
        comment_groups.columns = ['comment_text', 'count', 'min_target', 'max_target', 'mean_target']
        
        # Find boundary cases - uncertain labels (mix of toxic/non-toxic)
        boundary_comments = comment_groups[
            (comment_groups['min_target'] < self.threshold) & 
            (comment_groups['max_target'] >= self.threshold) &
            (comment_groups['mean_target'] >= self.boundary_lower) &
            (comment_groups['mean_target'] < self.boundary_upper)
        ]['comment_text'].tolist()
        
        # Get all rows matching boundary comments
        self.boundary_cases = self.df[self.df['comment_text'].isin(boundary_comments)].copy()
        
        print(f"Found {len(boundary_comments)} unique boundary comments")
        print(f"Found {len(self.boundary_cases)} rows with boundary comments")
        
        return self.boundary_cases
    
    def get_strategy_analysis(self) -> Dict[str, Any]:
        """Provide strategy analysis for boundary cases."""
        print("\n" + "=" * 80)
        print("BOUNDARY CASE STRATEGY ANALYSIS")
        print("=" * 80)
        
        total_toxic = (self.df['target'] >= self.threshold).sum()
        total_non_toxic = (self.df['target'] < self.threshold).sum()
        
        boundary_toxic = (self.boundary_cases['target'] >= self.threshold).sum()
        boundary_non_toxic = (self.boundary_cases['target'] < self.threshold).sum()
        
        analysis = {
            'total_rows': len(self.df),
            'total_toxic': int(total_toxic),
            'total_non_toxic': int(total_non_toxic),
            'boundary_rows': len(self.boundary_cases),
            'boundary_toxic': int(boundary_toxic),
            'boundary_non_toxic': int(boundary_non_toxic),
            'toxic_loss_if_removed': float(boundary_toxic / total_toxic * 100),
            'non_toxic_loss_if_removed': float(boundary_non_toxic / total_non_toxic * 100),
        }
        
        print(f"Total dataset: {analysis['total_rows']:,} rows")
        print(f"  - Toxic: {analysis['total_toxic']:,} ({analysis['total_toxic']/analysis['total_rows']*100:.2f}%)")
        print(f"  - Non-toxic: {analysis['total_non_toxic']:,} ({analysis['total_non_toxic']/analysis['total_rows']*100:.2f}%)")
        print()
        
        print(f"Boundary cases [{self.boundary_lower}, {self.boundary_upper}]: {analysis['boundary_rows']} rows")
        print(f"  - Toxic: {analysis['boundary_toxic']} ({analysis['toxic_loss_if_removed']:.2f}% of all toxic)")
        print(f"  - Non-toxic: {analysis['boundary_non_toxic']} ({analysis['non_toxic_loss_if_removed']:.2f}% of all non-toxic)")
        print()
        
        print("IMPACT IF REMOVED:")
        print(f"  - Loss {analysis['boundary_toxic']} toxic samples")
        print(f"  - Loss {analysis['boundary_non_toxic']} non-toxic samples")
        print(f"  - Manageable: {analysis['toxic_loss_if_removed'] < 1}% loss")
        
        return analysis
    
    def create_clean_dataset(self, output_file: str = "train_clean.csv") -> str:
        """Create dataset without boundary cases."""
        print(f"\nCreating clean dataset (removing boundary cases)...")
        
        clean_df = self.df[~self.df['comment_text'].isin(self.boundary_cases['comment_text'].unique())].copy()
        
        output_path = self.output_dir / output_file
        clean_df.to_csv(output_path, index=False)
        
        print(f"✓ Saved clean dataset: {output_path}")
        print(f"  - Original: {len(self.df):,} rows")
        print(f"  - Clean: {len(clean_df):,} rows")
        print(f"  - Removed: {len(self.df) - len(clean_df):,} rows")
        
        return str(output_path)
    
    def export_for_llm_labeling(self, output_file: str = "boundary_cases_for_llm.jsonl") -> str:
        """Export boundary cases for LLM labeling."""
        print(f"\nExporting boundary cases for LLM labeling...")
        
        output_path = self.output_dir / output_file
        
        # Get unique boundary comments with their annotations
        comment_groups = self.df.groupby('comment_text').agg({
            'target': ['count', 'min', 'max', 'mean', list],
        }).reset_index()
        
        comment_groups.columns = ['comment_text', 'count', 'min_target', 'max_target', 'mean_target', 'all_targets']
        
        boundary_comments = comment_groups[
            (comment_groups['min_target'] < self.threshold) & 
            (comment_groups['max_target'] >= self.threshold) &
            (comment_groups['mean_target'] >= self.boundary_lower) &
            (comment_groups['mean_target'] < self.boundary_upper)
        ]
        
        # Export as JSONL for LLM
        with open(output_path, 'w') as f:
            for idx, row in boundary_comments.iterrows():
                record = {
                    'id': idx,
                    'comment': row['comment_text'],
                    'human_annotations': {
                        'count': int(row['count']),
                        'scores': [float(s) for s in row['all_targets']],
                        'mean': float(row['mean_target']),
                        'min': float(row['min_target']),
                        'max': float(row['max_target']),
                    },
                    'task': 'Classify this comment as toxic (>=0.5) or non-toxic (<0.5)',
                    'instruction': (
                        'You are a content moderation expert. Rate the toxicity of the comment '
                        'on a scale 0-1, where 0=not toxic, 1=highly toxic. '
                        'Consider: insults, threats, obscene language, identity attacks, etc.'
                    )
                }
                f.write(json.dumps(record) + '\n')
        
        print(f"✓ Exported {len(boundary_comments)} boundary comments")
        print(f"  File: {output_path}")
        print(f"  Format: JSONL (one comment per line)")
        
        return str(output_path)
    
    def apply_uncertainty_weights(self) -> pd.DataFrame:
        """Apply uncertainty weights to boundary cases."""
        print("\nApplying uncertainty weights...")
        
        # Add weight column
        weighted_df = self.df.copy()
        weighted_df['sample_weight'] = 1.0
        
        # Lower weight for boundary cases
        boundary_mask = weighted_df['comment_text'].isin(self.boundary_cases['comment_text'].unique())
        weighted_df.loc[boundary_mask, 'sample_weight'] = 0.5  # Half weight for uncertain examples
        
        # Higher weight for confident toxic examples
        confident_toxic = (weighted_df['target'] >= 0.8)
        weighted_df.loc[confident_toxic, 'sample_weight'] = 2.0
        
        print(f"✓ Added sample weights:")
        print(f"  - Confident toxic (target >= 0.8): weight=2.0")
        print(f"  - Boundary cases [{self.boundary_lower}, {self.boundary_upper}]: weight=0.5")
        print(f"  - Others: weight=1.0")
        
        return weighted_df
    
    def generate_report(self) -> str:
        """Generate comprehensive report."""
        report_file = self.output_dir / 'boundary_cases_analysis.txt'
        
        analysis = self.get_strategy_analysis()
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"BOUNDARY CASE ANALYSIS - UNCERTAIN LABELS [{self.boundary_lower}, {self.boundary_upper}]\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("SITUATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Found {len(self.boundary_cases)} rows with boundary case comments\n")
            f.write(f"These have conflicting annotations that average to [{self.boundary_lower}, {self.boundary_upper}]\n")
            f.write(f"Both toxic and non-toxic annotators existed for these comments\n\n")
            
            f.write("IMPACT OF STRATEGIES\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("STRATEGY 1: REMOVE BOUNDARY CASES\n")
            f.write(f"  Data loss: {analysis['boundary_rows']} rows ({analysis['boundary_rows']/analysis['total_rows']*100:.2f}%)\n")
            f.write(f"  Toxic loss: {analysis['boundary_toxic']} ({analysis['toxic_loss_if_removed']:.3f}%)\n")
            f.write(f"  Verdict: ACCEPTABLE - minimal toxic data loss\n\n")
            
            f.write("STRATEGY 2: USE LLM FOR SOFT LABELS\n")
            f.write(f"  Data kept: {analysis['boundary_rows']} rows\n")
            f.write(f"  Requires: LLM API calls for {len(self.boundary_cases['comment_text'].unique())} unique comments\n")
            f.write(f"  Benefit: Preserve data + get reliable labels\n")
            f.write(f"  Verdict: BEST OPTION - requires effort but optimal\n\n")
            
            f.write("STRATEGY 3: UNCERTAINTY-AWARE TRAINING\n")
            f.write(f"  Data kept: All {analysis['total_rows']:,} rows\n")
            f.write(f"  Apply: Lower sample weights to boundary cases\n")
            f.write(f"  Use: Focal loss or uncertainty-aware loss\n")
            f.write(f"  Verdict: GOOD COMPROMISE - no data loss, model handles uncertainty\n\n")
            
            f.write("RECOMMENDATION\n")
            f.write("=" * 80 + "\n")
            f.write("Given severe class imbalance (11.5x), STRATEGY 2 or 3 preferred:\n")
            f.write("  1. Use LLM to relabel boundary cases (Best)\n")
            f.write("  2. Apply uncertainty-aware training (Good)\n")
            f.write("  3. Remove if time-constrained (Acceptable)\n")
        
        print(f"✓ Saved report: {report_file}")
        return str(report_file)


if __name__ == "__main__":
    handler = BoundaryCaseHandler(
        csv_path="data/train.csv",
        output_dir="outputs/data_exploration",
        boundary_lower=0.4,
        boundary_upper=0.5
    )
    
    handler.load_data()
    handler.identify_boundary_cases()
    handler.get_strategy_analysis()
    
    # Create outputs
    handler.create_clean_dataset()
    handler.export_for_llm_labeling()
    handler.generate_report()
    
    print("\n" + "=" * 80)
    print("Boundary case analysis complete!")
    print("=" * 80)
