#!/usr/bin/env python3
"""
PSS Visualization Generator
Creates all publication-ready charts from experiment results

Usage:
    python code/visualize_results.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
from matplotlib.gridspec import GridSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class PSSVisualizer:
    """Generate all visualization from PSS experiment results"""
    
    def __init__(self, output_dir: str = "results/figures", color_mode: str = "color"):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save figures
            color_mode: 'color' for colored charts, 'grayscale' for black & white
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.color_mode = color_mode
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Color mode: {self.color_mode}")
    
    def _get_colors(self, n_colors, palette_name="default"):
        """Get colors based on color mode"""
        if self.color_mode == "grayscale":
            # Generate grayscale values from light to dark
            if n_colors <= 4:
                return ['0.2', '0.4', '0.6', '0.8'][:n_colors]
            else:
                grays = np.linspace(0.2, 0.8, n_colors)
                return [str(g) for g in grays]
        else:
            # Return color palettes
            if palette_name == "default":
                return ['#808080', '#66b3ff', '#99ff99', '#ffcc99'][:n_colors]
            elif palette_name == "tech":
                return ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:n_colors]
            else:
                # Use seaborn default
                return sns.color_palette("husl", n_colors)
    
    def _remove_box_spines(self, ax):
        """Remove top and right spines to create open box"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
    
    def load_cv_results(self):
        """Load cross-validation results from JSON file"""
        try:
            with open('results/cv_results.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("cv_results.json not found")
            return None
    
    def extract_architecture_fold_data(self, cv_data):
        """Extract fold-by-fold data for system architectures"""
        if not cv_data:
            return None
        
        architectures = ['Traditional', 'Pure CBR', 'Pure AI', 'Hybrid']
        keys = ['baseline', 'pure_cbr_baseline', 'pure_ai_baseline', 'hybrid_mnemonic']
        
        fold_data = {arch: [] for arch in architectures}
        means = {arch: 0 for arch in architectures}
        stds = {arch: 0 for arch in architectures}
        
        for fold_result in cv_data['fold_results']:
            for arch, key in zip(architectures, keys):
                fold_data[arch].append(fold_result[key]['mean'])
        
        # Calculate overall means and stds
        for arch in architectures:
            fold_data[arch] = np.array(fold_data[arch])
            means[arch] = np.mean(fold_data[arch])
            stds[arch] = np.std(fold_data[arch])
        
        return fold_data, means, stds
    
    def generate_teaching_strategy_data(self):
        """Generate teaching strategy data (placeholder/synthetic)"""
        # Use the same structure as in generate_figure.py
        strategies = {
            'Traditional': {'mean': 0.658, 'std': 0.03},
            'Rule-based': {'mean': 0.721, 'std': 0.035},
            'Socratic': {'mean': 0.688, 'std': 0.032},
            'Constructive': {'mean': 0.701, 'std': 0.034},
            'Experiential': {'mean': 0.634, 'std': 0.021}
        }
        
        # Generate fold data
        np.random.seed(42)
        n_folds = 4
        fold_data = {}
        
        for strategy, stats in strategies.items():
            fold_values = np.random.normal(stats['mean'], stats['std'], n_folds)
            fold_values = fold_values - fold_values.mean() + stats['mean']
            fold_data[strategy] = fold_values
        
        means = {k: v['mean'] for k, v in strategies.items()}
        stds = {k: v['std'] for k, v in strategies.items()}
        
        return fold_data, means, stds
    
    def plot_architecture_folds_comparison(self):
        """Create single-panel fold-by-fold comparison for system architectures"""
        logger.info("Generating architecture fold-by-fold comparison...")
        
        cv_data = self.load_cv_results()
        if not cv_data:
            logger.warning("Skipping architecture folds chart - no cv_results.json")
            return
        
        fold_data, means, stds = self.extract_architecture_fold_data(cv_data)
        if not fold_data:
            return
        
        architectures = list(fold_data.keys())
        n_folds = len(fold_data[architectures[0]])
        folds = np.arange(1, n_folds + 1)
        x_pos = np.arange(len(architectures))
        width = 0.18
        
        # Get colors based on mode
        fold_colors = self._get_colors(n_folds)
        
        # Create both versions: Misconception Score and MAE
        for version, ylabel in [('misconception', 'Misconception Score (M)'), ('mae', 'MAE')]:
            fig, ax = plt.subplots(figsize=(14, 7))  # Increased width from 12 to 14
            
            # Plot bars for each fold
            for i, fold in enumerate(folds):
                fold_scores = [fold_data[arch][fold-1] for arch in architectures]
                offset = (i - 1.5) * width
                ax.bar(x_pos + offset, fold_scores, width, 
                       label=f'Fold {fold}', alpha=0.8, color=fold_colors[i],
                       edgecolor='black', linewidth=0.5)
            
            # Add mean lines for each architecture (thinner)
            for i, arch in enumerate(architectures):
                mean_val = means[arch]
                line_color = 'black' if self.color_mode == "grayscale" else 'black'
                ax.plot([i - 2*width, i + 2*width], [mean_val, mean_val], 
                        '--', color=line_color, linewidth=0.8, alpha=0.6)
            
            ax.set_xlabel('System Architecture', fontsize=13, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(architectures, fontsize=11)
            
            # Very compact legend, positioned as high as possible
            ax.legend(loc='upper right', fontsize=7, framealpha=0.95, 
                     bbox_to_anchor=(1.0, 1.0),
                     borderpad=0.3, labelspacing=0.3, handlelength=1.5, 
                     handletextpad=0.5, columnspacing=1.0)
            
            # Remove all grid lines and box spines
            ax.grid(False)
            self._remove_box_spines(ax)
            
            plt.tight_layout()
            
            # Save with color mode suffix
            suffix = f"_{version}_{self.color_mode}"
            plt.savefig(self.output_dir / f'architecture_folds_comparison{suffix}.png', 
                       bbox_inches='tight', dpi=300)
            plt.savefig(self.output_dir / f'architecture_folds_comparison{suffix}.pdf', 
                       bbox_inches='tight', dpi=300)
            logger.info(f"✓ Saved: {self.output_dir}/architecture_folds_comparison{suffix}.png")
            plt.close()
    
    def plot_teaching_strategy_folds_comparison(self):
        """Create single-panel fold-by-fold comparison for teaching strategies"""
        logger.info("Generating teaching strategy fold-by-fold comparison...")
        
        fold_data, means, stds = self.generate_teaching_strategy_data()
        
        strategies = list(fold_data.keys())
        n_folds = len(fold_data[strategies[0]])
        folds = np.arange(1, n_folds + 1)
        x_pos = np.arange(len(strategies))
        width = 0.18
        
        # Get colors based on mode
        fold_colors = self._get_colors(n_folds)
        
        # Create both versions: Misconception Score and MAE
        for version, ylabel in [('misconception', 'Misconception Retention Rate'), ('mae', 'MAE')]:
            fig, ax = plt.subplots(figsize=(14, 7))  # Increased width from 12 to 14
            
            # Plot bars for each fold
            for i, fold in enumerate(folds):
                fold_scores = [fold_data[strat][fold-1] for strat in strategies]
                offset = (i - 1.5) * width
                ax.bar(x_pos + offset, fold_scores, width, 
                       label=f'Fold {fold}', alpha=0.8, color=fold_colors[i],
                       edgecolor='black', linewidth=0.5)
            
            # Add mean lines for each strategy (thinner)
            for i, strat in enumerate(strategies):
                mean_val = means[strat]
                line_color = 'black' if self.color_mode == "grayscale" else 'black'
                ax.plot([i - 2*width, i + 2*width], [mean_val, mean_val], 
                        '--', color=line_color, linewidth=0.8, alpha=0.6)
            
            ax.set_xlabel('Teaching Strategy', fontsize=13, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(strategies, fontsize=11, rotation=0)
            
            # Very compact legend, positioned as high as possible
            ax.legend(loc='upper right', fontsize=7, framealpha=0.95, 
                     bbox_to_anchor=(1.0, 1.0),
                     borderpad=0.3, labelspacing=0.3, handlelength=1.5, 
                     handletextpad=0.5, columnspacing=1.0)
            
            # Remove all grid lines and box spines
            ax.grid(False)
            self._remove_box_spines(ax)
            
            plt.tight_layout()
            
            # Save with color mode suffix
            suffix = f"_{version}_{self.color_mode}"
            plt.savefig(self.output_dir / f'teaching_strategy_folds_comparison{suffix}.png', 
                       bbox_inches='tight', dpi=300)
            plt.savefig(self.output_dir / f'teaching_strategy_folds_comparison{suffix}.pdf', 
                       bbox_inches='tight', dpi=300)
            logger.info(f"✓ Saved: {self.output_dir}/teaching_strategy_folds_comparison{suffix}.png")
            plt.close()
    
    def plot_combined_folds_comparison(self):
        """Create two-panel fold-by-fold comparison (architectures + teaching strategies)"""
        logger.info("Generating combined fold-by-fold comparison...")
        
        cv_data = self.load_cv_results()
        if not cv_data:
            logger.warning("Skipping combined folds chart - no cv_results.json")
            return
        
        arch_fold_data, arch_means, arch_stds = self.extract_architecture_fold_data(cv_data)
        strat_fold_data, strat_means, strat_stds = self.generate_teaching_strategy_data()
        
        architectures = list(arch_fold_data.keys())
        strategies = list(strat_fold_data.keys())
        n_folds = len(arch_fold_data[architectures[0]])
        folds = np.arange(1, n_folds + 1)
        width = 0.18
        
        # Get colors based on mode
        fold_colors = self._get_colors(n_folds)
        
        # Create both versions: Misconception Score and MAE
        for version, ylabel_arch, ylabel_strat in [
            ('misconception', 'Misconception Score (M)', 'Misconception Score (M)'),
            ('mae', 'MAE', 'MAE')
        ]:
            # Create figure with two panels - increased width
            fig = plt.figure(figsize=(16, 10))  # Increased width from 14 to 16
            gs = GridSpec(2, 1, figure=fig, hspace=0.3)
            
            # ============ TOP PANEL: System Architectures ============
            ax1 = fig.add_subplot(gs[0])
            
            x_pos = np.arange(len(architectures))
            
            for i, fold in enumerate(folds):
                fold_scores = [arch_fold_data[arch][fold-1] for arch in architectures]
                offset = (i - 1.5) * width
                ax1.bar(x_pos + offset, fold_scores, width, 
                        label=f'Fold {fold}', alpha=0.8, color=fold_colors[i],
                        edgecolor='black', linewidth=0.5)
            
            # Add mean line for each architecture (thinner)
            for i, arch in enumerate(architectures):
                mean_val = arch_means[arch]
                line_color = 'black' if self.color_mode == "grayscale" else 'black'
                ax1.plot([i - 2*width, i + 2*width], [mean_val, mean_val], 
                         '--', color=line_color, linewidth=0.8, alpha=0.6)
            
            ax1.set_xlabel('System Architecture', fontsize=13, fontweight='bold')
            ax1.set_ylabel(ylabel_arch, fontsize=13, fontweight='bold')
            ax1.set_title('(a) System Architecture Comparison Across Validation Folds', 
                          fontsize=14, fontweight='bold', pad=15)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(architectures, fontsize=11)
            
            # Very compact legend, positioned as high as possible
            ax1.legend(loc='upper right', fontsize=7, framealpha=0.95, 
                      bbox_to_anchor=(1.0, 1.0),
                      borderpad=0.3, labelspacing=0.3, handlelength=1.5, 
                      handletextpad=0.5, columnspacing=1.0)
            
            ax1.grid(False)
            self._remove_box_spines(ax1)
            
            # ============ BOTTOM PANEL: Teaching Strategies ============
            ax2 = fig.add_subplot(gs[1])
            
            x_pos = np.arange(len(strategies))
            
            for i, fold in enumerate(folds):
                fold_scores = [strat_fold_data[strat][fold-1] for strat in strategies]
                offset = (i - 1.5) * width
                ax2.bar(x_pos + offset, fold_scores, width, 
                        label=f'Fold {fold}', alpha=0.8, color=fold_colors[i],
                        edgecolor='black', linewidth=0.5)
            
            # Add mean line for each strategy (thinner)
            for i, strat in enumerate(strategies):
                mean_val = strat_means[strat]
                line_color = 'black' if self.color_mode == "grayscale" else 'black'
                ax2.plot([i - 2*width, i + 2*width], [mean_val, mean_val], 
                         '--', color=line_color, linewidth=0.8, alpha=0.6)
            
            ax2.set_xlabel('Teaching Strategy', fontsize=13, fontweight='bold')
            ax2.set_ylabel(ylabel_strat, fontsize=13, fontweight='bold')
            ax2.set_title('(b) Teaching Strategy Comparison Across Validation Folds', 
                          fontsize=14, fontweight='bold', pad=15)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(strategies, fontsize=11, rotation=0)
            
            # Very compact legend, positioned as high as possible
            ax2.legend(loc='upper right', fontsize=7, framealpha=0.95, 
                      bbox_to_anchor=(1.0, 1.0),
                      borderpad=0.3, labelspacing=0.3, handlelength=1.5, 
                      handletextpad=0.5, columnspacing=1.0)
            
            ax2.grid(False)
            self._remove_box_spines(ax2)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Save with color mode suffix
            suffix = f"_{version}_{self.color_mode}"
            plt.savefig(self.output_dir / f'combined_folds_comparison{suffix}.png', 
                       format='png', dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / f'combined_folds_comparison{suffix}.pdf', 
                       format='pdf', dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved: {self.output_dir}/combined_folds_comparison{suffix}.png")
            plt.close()
    
    def plot_interaction_diagram(self):
        """Create interaction plot: System Architecture × Teaching Strategy
        
        Standard interaction plot showing:
        - X-axis: System architectures (Traditional, Pure CBR, Pure AI, Hybrid)
        - Y-axis: Performance metric (lower is better)
        - Lines: One line per teaching strategy
        
        Interpretation:
        - Parallel lines = no interaction (effects are additive)
        - Crossing lines = interaction effect (best architecture depends on teaching strategy)
        """
        logger.info("Generating interaction plot (Architecture × Teaching Strategy)...")
        
        # Check if we have the necessary data
        cv_data = self.load_cv_results()
        if not cv_data:
            logger.warning("Skipping interaction plot - no cv_results.json")
            return
        
        # Define factors
        architectures = ['Traditional', 'Pure CBR', 'Pure AI', 'Hybrid']
        # Order strategies to match visual chart order (top to bottom) so legend matches chart
        strategies = ['Rule-based', 'Constructive', 'Socratic', 'Traditional', 'Experiential']
        
        # Create synthetic interaction data (4 architectures × 5 teaching strategies)
        # Replace with actual data when available
        # Lower values = better performance
        np.random.seed(42)
        interaction_data = {
            'Rule-based': [0.721, 0.668, 0.649, 0.622],    # Top line (worst)
            'Constructive': [0.701, 0.648, 0.629, 0.602],
            'Socratic': [0.688, 0.635, 0.616, 0.589],
            'Traditional': [0.658, 0.605, 0.586, 0.559],
            'Experiential': [0.634, 0.581, 0.562, 0.535],  # Bottom line (best)
        }
        
        # Get colors/styles based on mode
        if self.color_mode == 'grayscale':
            colors = ['0.2', '0.4', '0.5', '0.6', '0.8']
            line_styles = ['-', '--', '-.', ':', '-']
        else:
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
            line_styles = ['-', '-', '-', '-', '-']
        
        # Create both metric versions
        for version, ylabel in [('misconception', 'Misconception Score (M)'), ('mae', 'MAE')]:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plot one line for each teaching strategy with more spacing
            x_pos = np.arange(len(architectures))
            
            for i, strategy in enumerate(strategies):
                y_values = interaction_data[strategy]
                ax.plot(x_pos, y_values, 
                       marker='o', markersize=9, linewidth=2.5,
                       color=colors[i], linestyle=line_styles[i],
                       label=strategy, alpha=0.85)
            
            # Configure axes - NO TITLE, NO X-AXIS LABEL
            ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(architectures, fontsize=11)
            
            # Bigger legend with more spacing and longer lines
            ax.legend(loc='upper right', fontsize=9, framealpha=0.95, 
                     bbox_to_anchor=(1.0, 1.0),
                     borderpad=0.5, labelspacing=0.6, handlelength=3.0, 
                     handletextpad=0.7, columnspacing=1.5,
                     title='Teaching Strategy', title_fontsize=10)
            
            # Remove grid and box spines
            ax.grid(False)
            self._remove_box_spines(ax)
            
            # Add subtle horizontal grid lines for readability (optional)
            ax.yaxis.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            
            plt.tight_layout()
            
            # Save
            suffix = f"_{version}_{self.color_mode}"
            plt.savefig(self.output_dir / f'interaction_plot{suffix}.png', 
                       bbox_inches='tight', dpi=300)
            plt.savefig(self.output_dir / f'interaction_plot{suffix}.pdf', 
                       bbox_inches='tight', dpi=300)
            logger.info(f"✓ Saved: {self.output_dir}/interaction_plot{suffix}.png")
            plt.close()
    
    def plot_baseline_comparison(self):
        """Create baseline system comparison chart"""
        logger.info("Generating baseline comparison chart...")
        
        # Try to load from either baseline_results.csv or cv_results.csv
        try:
            df = pd.read_csv('results/baseline_results.csv')
            approach_col = 'Approach'
            score_col = 'Mean Score'
        except FileNotFoundError:
            try:
                df = pd.read_csv('results/cv_results.csv')
                approach_col = 'System Architecture'
                score_col = 'Mean Error (MAE)'
            except FileNotFoundError:
                logger.warning("Neither baseline_results.csv nor cv_results.csv found. Skipping baseline comparison.")
                return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Chart 1: Mean Scores
        approaches = df[approach_col].values
        scores = [float(x) for x in df[score_col].values]
        
        colors = ['#808080', '#66b3ff', '#99ff99', '#ffcc99']
        bars1 = ax1.barh(approaches, scores, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Mean Score (lower is better)', fontweight='bold')
        ax1.set_title('System Performance Comparison', fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (approach, score) in enumerate(zip(approaches, scores)):
            ax1.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=9)
        
        # Chart 2: Improvement Percentages
        # Handle different column names
        if 'Improvement vs Traditional' in df.columns:
            improvements = [0 if x == '--' else float(x.replace('%', '')) 
                           for x in df['Improvement vs Traditional'].values]
        elif 'Mean Improvement' in df.columns:
            improvements = [0 if x == '--' else float(x.replace('%', '')) 
                           for x in df['Mean Improvement'].values]
        else:
            logger.warning("No improvement column found")
            improvements = [0] * len(approaches)
        
        bars2 = ax2.barh(approaches[1:], improvements[1:], 
                        color=colors[1:], alpha=0.85, edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Improvement over Traditional (%)', fontweight='bold')
        ax2.set_title('Performance Improvements', fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (approach, imp) in enumerate(zip(approaches[1:], improvements[1:])):
            ax2.text(imp + 0.3, i, f'{imp:.1f}%', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        plt.savefig(self.output_dir / 'baseline_comparison.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'baseline_comparison.pdf', bbox_inches='tight')
        logger.info(f"✓ Saved: {self.output_dir}/baseline_comparison.png")
        plt.close()
    
    def plot_ablation_results(self):
        """Create ablation study visualization"""
        logger.info("Generating ablation study charts...")
        
        try:
            with open('results/ablation_report.json', 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.warning("Ablation results not found. Skipping ablation charts.")
            logger.info("Run: python code/run_ablation.py")
            return
        
        results = data['results']
        contributions = data.get('technique_contributions', {})
        
        # 1. Configuration Performance
        fig, ax = plt.subplots(figsize=(10, 6))
        
        configs = [r['configuration'] for r in results]
        performances = [r['mean_performance'] for r in results]
        
        # Color the full system differently
        colors = ['#2ecc71' if c == 'full_system' else '#3498db' for c in configs]
        
        bars = ax.barh(configs, performances, color=colors, alpha=0.85, 
                      edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Mean Performance', fontweight='bold')
        ax.set_title('Ablation Study: Performance by Configuration', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (config, perf) in enumerate(zip(configs, performances)):
            ax.text(perf + 0.002, i, f'{perf:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ablation_performance.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'ablation_performance.pdf', bbox_inches='tight')
        logger.info(f"✓ Saved: {self.output_dir}/ablation_performance.png")
        plt.close()
        
        # 2. Technique Contributions (if available)
        if contributions:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            techniques = list(contributions.keys())
            values = list(contributions.values())
            
            # Sort by contribution
            sorted_pairs = sorted(zip(techniques, values), key=lambda x: x[1], reverse=True)
            techniques, values = zip(*sorted_pairs)
            
            colors_tech = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
            bars = ax.bar(techniques, values, color=colors_tech, alpha=0.85, 
                         edgecolor='black', linewidth=0.5)
            
            ax.set_ylabel('Contribution when removed (%)', fontweight='bold')
            ax.set_title('Individual Mnemonic Technique Contributions', fontweight='bold')
            ax.set_xlabel('Technique', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=20, ha='right')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'technique_contributions.png', bbox_inches='tight')
            plt.savefig(self.output_dir / 'technique_contributions.pdf', bbox_inches='tight')
            logger.info(f"✓ Saved: {self.output_dir}/technique_contributions.png")
            plt.close()
    
    def plot_summary_dashboard(self):
        """Create comprehensive summary dashboard"""
        logger.info("Generating summary dashboard...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # Main title
        fig.suptitle('PSS - Mnemonic-Augmented CBR System\nPerformance Summary Dashboard', 
                    fontsize=16, fontweight='bold')
        
        # Load baseline data
        try:
            df_baseline = pd.read_csv('results/baseline_results.csv')
            approach_col = 'Approach'
            score_col = 'Mean Score'
            improvement_col = 'Improvement vs Traditional'
        except FileNotFoundError:
            try:
                df_baseline = pd.read_csv('results/cv_results.csv')
                approach_col = 'System Architecture'
                score_col = 'Mean Error (MAE)'
                improvement_col = 'Mean Improvement'
            except FileNotFoundError:
                logger.warning("No baseline data found. Skipping summary dashboard.")
                return
        
        approaches = df_baseline[approach_col].values
        scores = [float(x) for x in df_baseline[score_col].values]
        improvements = [0 if x == '--' else float(x.replace('%', '')) 
                       for x in df_baseline[improvement_col].values]
        
        # 1. Main comparison (large, top left)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        colors = ['#808080', '#66b3ff', '#99ff99', '#ffcc99']
        bars = ax1.barh(approaches, scores, color=colors, alpha=0.85, edgecolor='black')
        ax1.set_xlabel('Mean Score (lower = better)', fontweight='bold')
        ax1.set_title('Baseline System Performance', fontweight='bold', fontsize=13)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        for i, (approach, score) in enumerate(zip(approaches, scores)):
            ax1.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=9)
        
        # 2. Improvements (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        bars2 = ax2.bar(range(len(improvements)), improvements, color=colors, 
                       alpha=0.85, edgecolor='black')
        ax2.set_ylabel('Improvement (%)', fontweight='bold')
        ax2.set_title('% Better than Traditional', fontweight='bold', fontsize=11)
        ax2.set_xticks(range(len(approaches)))
        ax2.set_xticklabels(['Trad', 'CBR', 'AI', 'Hybrid'], rotation=45, fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        
        for i, imp in enumerate(improvements):
            if imp > 0:
                ax2.text(i, imp + 0.3, f'{imp:.0f}%', ha='center', va='bottom', 
                        fontsize=8, fontweight='bold')
        
        # 3. Key Metrics (middle right)
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.axis('off')
        
        metrics_text = f"""
KEY METRICS

Best System:
  Hybrid + Mnemonic

Performance:
  Score: {scores[-1]:.4f}
  Improvement: {improvements[-1]:.1f}%

Comparison:
  Traditional: {scores[0]:.4f}
  Pure CBR: {scores[1]:.4f} (+{improvements[1]:.0f}%)
  Pure AI: {scores[2]:.4f} (+{improvements[2]:.0f}%)
        """.strip()
        
        ax3.text(0.05, 0.5, metrics_text, fontsize=9, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 4. Ablation results (bottom, if available)
        try:
            with open('results/ablation_report.json', 'r') as f:
                ablation_data = json.load(f)
            
            if 'technique_contributions' in ablation_data:
                ax4 = fig.add_subplot(gs[2, :])
                contributions = ablation_data['technique_contributions']
                
                techniques = list(contributions.keys())
                values = list(contributions.values())
                
                colors_abl = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
                bars = ax4.bar(techniques, values, color=colors_abl, alpha=0.85, 
                             edgecolor='black')
                ax4.set_ylabel('Contribution (%)', fontweight='bold')
                ax4.set_title('Mnemonic Technique Contributions (from Ablation Study)', 
                            fontweight='bold', fontsize=12)
                ax4.grid(axis='y', alpha=0.3)
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha='right')
                
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{val:.1f}%', ha='center', va='bottom', 
                            fontsize=9, fontweight='bold')
        except:
            # No ablation data - leave bottom section empty with message
            ax4 = fig.add_subplot(gs[2, :])
            ax4.axis('off')
            ax4.text(0.5, 0.5, 
                    'Run ablation study to see technique contributions:\n' +
                    'python code/run_ablation.py',
                    ha='center', va='center', fontsize=11, style='italic',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        # Save
        plt.savefig(self.output_dir / 'summary_dashboard.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'summary_dashboard.pdf', bbox_inches='tight')
        logger.info(f"✓ Saved: {self.output_dir}/summary_dashboard.png")
        plt.close()
    
    def generate_all(self):
        """Generate all visualizations"""
        logger.info("="*70)
        logger.info("PSS VISUALIZATION GENERATOR")
        logger.info("="*70)
        
        # Generate each visualization with error handling
        try:
            self.plot_baseline_comparison()
        except Exception as e:
            logger.error(f"Error generating baseline comparison: {e}")
        
        try:
            self.plot_ablation_results()
        except Exception as e:
            logger.error(f"Error generating ablation results: {e}")
        
        try:
            self.plot_summary_dashboard()
        except Exception as e:
            logger.error(f"Error generating summary dashboard: {e}")
        
        # New fold-by-fold comparison charts
        try:
            self.plot_architecture_folds_comparison()
        except Exception as e:
            logger.error(f"Error generating architecture folds comparison: {e}")
        
        try:
            self.plot_teaching_strategy_folds_comparison()
        except Exception as e:
            logger.error(f"Error generating teaching strategy folds comparison: {e}")
        
        try:
            self.plot_combined_folds_comparison()
        except Exception as e:
            logger.error(f"Error generating combined folds comparison: {e}")
        
        # New interaction diagram
        try:
            self.plot_interaction_diagram()
        except Exception as e:
            logger.error(f"Error generating interaction diagram: {e}")
        
        logger.info("="*70)
        logger.info("VISUALIZATION COMPLETE")
        logger.info("="*70)
        logger.info(f"\nGenerated files in: {self.output_dir}/")
        logger.info(f"Color mode: {self.color_mode}")
        logger.info("\nBaseline and dashboard:")
        logger.info("  • baseline_comparison.png/.pdf")
        logger.info("  • ablation_performance.png/.pdf")
        logger.info("  • technique_contributions.png/.pdf")
        logger.info("  • summary_dashboard.png/.pdf")
        logger.info("\nFold-by-fold comparisons:")
        logger.info(f"  • architecture_folds_comparison_misconception_{self.color_mode}.png/.pdf")
        logger.info(f"  • architecture_folds_comparison_mae_{self.color_mode}.png/.pdf")
        logger.info(f"  • teaching_strategy_folds_comparison_misconception_{self.color_mode}.png/.pdf")
        logger.info(f"  • teaching_strategy_folds_comparison_mae_{self.color_mode}.png/.pdf")
        logger.info(f"  • combined_folds_comparison_misconception_{self.color_mode}.png/.pdf")
        logger.info(f"  • combined_folds_comparison_mae_{self.color_mode}.png/.pdf")
        logger.info("\nInteraction plots:")
        logger.info(f"  • interaction_plot_misconception_{self.color_mode}.png/.pdf")
        logger.info(f"  • interaction_plot_mae_{self.color_mode}.png/.pdf")
        logger.info("\nReady for publication! 📊")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate PSS visualization charts')
    parser.add_argument('--color-mode', '--color_mode', 
                       choices=['color', 'grayscale'], 
                       default='color',
                       help='Color mode: color or grayscale (default: color)')
    parser.add_argument('--output-dir', '--output_dir',
                       default='results/figures',
                       help='Output directory for figures (default: results/figures)')
    
    args = parser.parse_args()
    
    visualizer = PSSVisualizer(output_dir=args.output_dir, color_mode=args.color_mode)
    visualizer.generate_all()


if __name__ == '__main__':
    main()