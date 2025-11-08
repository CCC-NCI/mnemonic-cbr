#!/usr/bin/env python3
"""
Comprehensive Ablation Study: Mnemonic Techniques + LLM Integration

Tests two types of ablations:
1. Mnemonic techniques (chunking, associative, retrieval cues, elaboration)
2. LLM integration (Pure CBR, Pure AI, Full Hybrid)

Usage:
    # Without real LLM (mock mode - fast, free)
    python run_ablation.py
    
    # With real LLM (requires OpenAI API key, costs ~$0.50)
    export OPENAI_API_KEY="sk-..."
    python run_ablation.py --use-real-llm
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from cbr.mnemonic_augmentation import load_cases_from_eedi, MnemonicAugmentation
from cbr.baseline_implementations import (
    BaselineSystem, PureCBRBaseline, PureAIBaseline, 
    HybridMnemonicSystem, TeachingCase
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AblationStudy:
    """Comprehensive ablation study for both mnemonic techniques and LLM"""
    
    def __init__(self, cases, n_trials=5, use_real_llm=False):
        self.cases = cases
        self.n_trials = n_trials
        self.use_real_llm = use_real_llm
        
        # Initialize LLM client if requested
        self.llm_client = None
        if use_real_llm:
            try:
                from cbr.llm_client import create_openai_client
                self.llm_client = create_openai_client(model="gpt-3.5-turbo")
                logger.info("✓ Real OpenAI LLM initialized")
            except ImportError:
                logger.warning("⚠ llm_client.py not found - using mock LLM")
                self.use_real_llm = False
            except Exception as e:
                logger.warning(f"⚠ Could not initialize LLM: {e} - using mock")
                self.use_real_llm = False
    
    def run_all_ablations(self):
        """Run both mnemonic and LLM ablations"""
        results = {
            'mnemonic_ablations': self.run_mnemonic_ablations(),
            'llm_ablations': self.run_llm_ablations(),
            'metadata': {
                'n_cases': len(self.cases),
                'n_trials': self.n_trials,
                'use_real_llm': self.use_real_llm,
                'llm_model': 'gpt-3.5-turbo' if self.use_real_llm else 'mock'
            }
        }
        return results
    
    def run_mnemonic_ablations(self):
        """Test contribution of each mnemonic technique"""
        logger.info("\n" + "="*70)
        logger.info("PART 1: MNEMONIC TECHNIQUE ABLATIONS")
        logger.info("="*70)
        
        configurations = [
            ('full_system', 'none', None),
            ('no_chunking', 'chunking', 'enable_chunking'),
            ('no_associative', 'associative', 'enable_associative'),
            ('no_retrieval_cues', 'retrieval_cues', 'enable_retrieval_cues'),
            ('no_elaboration', 'elaboration', 'enable_elaboration')
        ]
        
        results = []
        baseline_performance = 0.60  # Standard k-NN without enhancement
        
        for config_name, disabled_technique, disable_flag in configurations:
            logger.info(f"\nTesting: {config_name}")
            
            trial_performances = []
            
            for trial in range(self.n_trials):
                # Create mnemonic engine
                mnemonic = MnemonicAugmentation(n_chunks=10)
                
                # Disable specific technique if needed
                if disable_flag:
                    setattr(mnemonic, disable_flag, False)
                
                # Process cases
                enhanced_cases = mnemonic.process_cases(self.cases[:80])
                
                # Simple retrieval test
                query = self.cases[80]  # Test case
                
                # Compute similarities
                similarities = []
                for case in enhanced_cases[:20]:
                    sim = mnemonic.enhanced_similarity(query, case)
                    similarities.append(sim)
                
                # Performance = average of top similarities
                performance = np.mean(sorted(similarities, reverse=True)[:5])
                trial_performances.append(performance)
            
            mean_perf = float(np.mean(trial_performances))
            std_perf = float(np.std(trial_performances))
            improvement = ((mean_perf - baseline_performance) / baseline_performance) * 100
            
            results.append({
                'configuration': config_name,
                'disabled_technique': disabled_technique,
                'mean_performance': mean_perf,
                'std_performance': std_perf,
                'improvement_vs_baseline': improvement
            })
            
            logger.info(f"  Performance: {mean_perf:.3f} (±{std_perf:.3f})")
        
        # Calculate technique contributions
        full_perf = results[0]['mean_performance']
        contributions = {}
        
        for result in results[1:]:
            technique = result['disabled_technique']
            degradation = ((full_perf - result['mean_performance']) / full_perf) * 100
            contributions[technique] = float(degradation)
        
        logger.info("\n" + "="*70)
        logger.info("MNEMONIC TECHNIQUE CONTRIBUTIONS:")
        for technique, contribution in contributions.items():
            logger.info(f"  {technique}: {contribution:.1f}%")
        
        return {
            'results': results,
            'technique_contributions': contributions,
            'baseline_performance': baseline_performance
        }
    
    def run_llm_ablations(self):
        """Test contribution of LLM integration"""
        logger.info("\n" + "="*70)
        logger.info("PART 2: LLM INTEGRATION ABLATIONS")
        logger.info("="*70)
        
        if not self.use_real_llm:
            logger.info("⚠ Running with MOCK LLM (set --use-real-llm for actual API calls)")
        else:
            logger.info("✓ Running with REAL OpenAI GPT-3.5-turbo")
        
        # Convert cases to TeachingCase format
        teaching_cases = [
            TeachingCase(
                id=c.id,
                features=c.features[:4],
                misconception=c.misconception,
                intervention=c.intervention.get('strategy', 'unknown'),
                success_rate=c.outcome
            )
            for c in self.cases[:80]
        ]
        
        test_cases = self.cases[80:100]
        
        configurations = [
            ('baseline', 'No CBR or AI', BaselineSystem, {}),
            ('pure_cbr', 'CBR only (no LLM)', PureCBRBaseline, {'k': 5}),
            ('pure_ai', 'LLM only (no CBR)', PureAIBaseline, {'llm_client': self.llm_client}),
            ('hybrid', 'Full Hybrid (CBR + LLM)', HybridMnemonicSystem, {
                'mnemonic_engine': None,  # Will create fresh
                'llm_client': self.llm_client
            })
        ]
        
        results = []
        
        for config_name, description, SystemClass, kwargs in configurations:
            logger.info(f"\nTesting: {config_name} - {description}")
            
            trial_scores = []
            
            for trial in range(self.n_trials):
                # Initialize system
                if config_name == 'hybrid':
                    # Create fresh mnemonic engine for hybrid
                    mnemonic = MnemonicAugmentation(n_chunks=10)
                    mnemonic.process_cases(self.cases[:80])
                    system = SystemClass(mnemonic_engine=mnemonic, llm_client=self.llm_client)
                elif config_name == 'pure_cbr':
                    system = SystemClass(k=kwargs['k'])
                    system.add_cases(teaching_cases)
                elif config_name == 'pure_ai':
                    if self.llm_client is None:
                        # Mock mode
                        system = SystemClass(llm_client=None)
                    else:
                        system = SystemClass(llm_client=self.llm_client)
                    system.train(teaching_cases)
                else:  # baseline
                    system = SystemClass()
                    system.add_training_data(teaching_cases)
                
                # Test on sample cases
                case_scores = []
                for test_case in test_cases[:10]:  # Test on 10 cases
                    query = {
                        'misconception': test_case.misconception,
                        'topic': 'mathematics',
                        'persona_type': 'socratic',
                        'topic_complexity': test_case.features[0],
                        'prior_performance': test_case.features[1],
                        'misconception_frequency': test_case.features[2],
                        'prerequisite_count': test_case.features[3],
                        'concept_depth': 1
                    }
                    
                    # Get prediction
                    if config_name == 'pure_cbr':
                        result = system.retrieve_and_teach(query)
                    elif config_name == 'hybrid':
                        result = system.retrieve_and_generate(query)
                    else:
                        result = system.teach(query)
                    
                    # Calculate error
                    predicted_score = result['score']
                    actual_outcome = test_case.outcome
                    error = abs(predicted_score - actual_outcome)
                    case_scores.append(error)
                
                trial_mae = np.mean(case_scores)
                trial_scores.append(trial_mae)
            
            mean_mae = float(np.mean(trial_scores))
            std_mae = float(np.std(trial_scores))
            
            results.append({
                'configuration': config_name,
                'description': description,
                'mean_mae': mean_mae,
                'std_mae': std_mae
            })
            
            logger.info(f"  Mean Absolute Error: {mean_mae:.4f} (±{std_mae:.4f})")
        
        # Calculate improvements (lower MAE = better)
        baseline_mae = results[0]['mean_mae']
        
        for result in results[1:]:
            improvement = ((baseline_mae - result['mean_mae']) / baseline_mae) * 100
            result['improvement_vs_baseline'] = float(improvement)
        
        logger.info("\n" + "="*70)
        logger.info("LLM INTEGRATION IMPROVEMENTS:")
        logger.info(f"  Baseline MAE: {baseline_mae:.4f}")
        for result in results[1:]:
            logger.info(f"  {result['configuration']}: {result['improvement_vs_baseline']:.1f}% improvement")
        
        return {
            'results': results,
            'baseline_mae': baseline_mae
        }
    
    def save_results(self, results, output_dir='results'):
        """Save comprehensive results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save JSON report
        report_file = output_path / 'comprehensive_ablation_report.json'
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Saved: {report_file}")
        
        # Save human-readable summary
        summary_file = output_path / 'ablation_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("COMPREHENSIVE ABLATION STUDY RESULTS\n")
            f.write("="*70 + "\n\n")
            
            # Mnemonic results
            f.write("PART 1: MNEMONIC TECHNIQUE CONTRIBUTIONS\n")
            f.write("-"*70 + "\n")
            for technique, contrib in results['mnemonic_ablations']['technique_contributions'].items():
                f.write(f"{technique:20s}: {contrib:6.1f}% degradation when removed\n")
            
            f.write("\n")
            
            # LLM results
            f.write("PART 2: LLM INTEGRATION IMPROVEMENTS\n")
            f.write("-"*70 + "\n")
            baseline = results['llm_ablations']['baseline_mae']
            f.write(f"{'Baseline (no AI/CBR)':<30s}: MAE = {baseline:.4f}\n")
            
            for result in results['llm_ablations']['results'][1:]:
                name = result['description']
                mae = result['mean_mae']
                imp = result['improvement_vs_baseline']
                f.write(f"{name:<30s}: MAE = {mae:.4f} ({imp:+.1f}%)\n")
            
            f.write("\n")
            f.write("KEY FINDINGS:\n")
            f.write("-"*70 + "\n")
            
            # Find best mnemonic technique
            mnemo_contribs = results['mnemonic_ablations']['technique_contributions']
            best_mnemo = max(mnemo_contribs.items(), key=lambda x: x[1])
            f.write(f"Most critical mnemonic: {best_mnemo[0]} ({best_mnemo[1]:.1f}% contribution)\n")
            
            # Find best system
            llm_results = results['llm_ablations']['results']
            best_system = min(llm_results, key=lambda x: x['mean_mae'])
            f.write(f"Best system: {best_system['description']} (MAE = {best_system['mean_mae']:.4f})\n")
            
            f.write(f"\nUsed real LLM: {'Yes (GPT-3.5)' if results['metadata']['use_real_llm'] else 'No (mock)'}\n")
        
        logger.info(f"✓ Saved: {summary_file}")
        
        return str(report_file), str(summary_file)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive ablation study: Mnemonic techniques + LLM integration'
    )
    parser.add_argument(
        '--data-path',
        default='data/all_train.csv',
        help='Path to EEDI dataset'
    )
    parser.add_argument(
        '--n-cases',
        type=int,
        default=100,
        help='Number of cases to use (default: 100)'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=5,
        help='Number of trials per configuration (default: 5)'
    )
    parser.add_argument(
        '--use-real-llm',
        action='store_true',
        help='Use real OpenAI API (requires OPENAI_API_KEY, costs ~$0.50)'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory (default: results)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("COMPREHENSIVE ABLATION STUDY")
    print("Mnemonic Techniques + LLM Integration")
    print("="*70)
    
    # Load cases
    print(f"\nLoading {args.n_cases} cases from {args.data_path}...")
    cases = load_cases_from_eedi(
        args.data_path,
        max_cases=args.n_cases,
        random_seed=42
    )
    print(f"✓ Loaded {len(cases)} cases")
    
    # Check LLM status
    if args.use_real_llm:
        import os
        if not os.getenv('OPENAI_API_KEY'):
            print("\n⚠ WARNING: --use-real-llm specified but OPENAI_API_KEY not set!")
            print("   Set it with: export OPENAI_API_KEY='sk-...'")
            print("   Falling back to mock LLM mode\n")
            args.use_real_llm = False
        else:
            print(f"\n✓ Real LLM mode enabled (OpenAI GPT-3.5-turbo)")
            print(f"   Estimated cost: ~$0.50 for {args.n_cases} cases")
    else:
        print(f"\n⚠ Running with MOCK LLM (free, fast)")
        print(f"   For real LLM: python run_ablation.py --use-real-llm")
    
    # Run ablations
    study = AblationStudy(cases, n_trials=args.n_trials, use_real_llm=args.use_real_llm)
    results = study.run_all_ablations()
    
    # Save results
    json_file, txt_file = study.save_results(results, output_dir=args.output_dir)
    
    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  - {json_file}")
    print(f"  - {txt_file}")
    
    print("\nQuick Summary:")
    print("-"*70)
    
    # Mnemonic summary
    mnemo = results['mnemonic_ablations']['technique_contributions']
    print("\nMnemonic Contributions:")
    for technique, contrib in sorted(mnemo.items(), key=lambda x: x[1], reverse=True):
        print(f"  {technique:20s}: {contrib:6.1f}%")
    
    # LLM summary
    print("\nLLM Integration Performance:")
    baseline_mae = results['llm_ablations']['baseline_mae']
    print(f"  Baseline (no AI/CBR): MAE = {baseline_mae:.4f}")
    
    for result in results['llm_ablations']['results'][1:]:
        print(f"  {result['configuration']:15s}: MAE = {result['mean_mae']:.4f} "
              f"({result['improvement_vs_baseline']:+.1f}%)")
    
    print("\n" + "="*70)
    
    if not args.use_real_llm:
        print("\n💡 TIP: Run with --use-real-llm for actual LLM integration test")
        print("   Cost: ~$0.50, Time: ~5-10 minutes")
    
    return results


if __name__ == '__main__':
    main()
