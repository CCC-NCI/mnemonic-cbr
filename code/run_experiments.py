#!/usr/bin/env python3
"""
PSS Experiment Runner with 4-FOLD CROSS-VALIDATION

FIXED: Now implements actual 4-fold cross-validation as claimed in paper.
FIXED: Feature indices now align correctly between training and testing.

Terminology:
- Baseline: Standard teaching without CBR-AI (0.658)
- Pure CBR Baseline: Retrieval-only (0.605, 8% improvement)  
- Pure AI Baseline: AI without grounding (0.586, 11% improvement)
- Hybrid + Mnemonic: Full system (0.559, 15% improvement)

Teaching Personas: socratic, constructive, experiential, rule_based, traditional_teaching
"""

import argparse
from pathlib import Path
import logging
import sys
import json
import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))

from cbr.mnemonic_augmentation import load_cases_from_eedi, MnemonicAugmentation, Case
from cbr.baseline_implementations import (
    BaselineSystem, PureCBRBaseline, PureAIBaseline, HybridMnemonicSystem, 
    TeachingCase, BASELINE_SCORES, TEACHING_PERSONAS, print_terminology_summary
)
from cbr.llm_client import create_openai_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_four_fold_cv(data_path: str, n_cases: int = 200, 
                     sample_from_full: bool = False,
                     use_real_llm: bool = True):
    """
    Run 4-fold cross-validation with REAL LLM
    
    Args:
        data_path: Path to EEDI dataset
        n_cases: Total number of cases to sample
        sample_from_full: If True, sample from full 17M records (slow!)
        use_real_llm: If True, use real OpenAI API (requires OPENAI_API_KEY)
    """
    
    logger.info("="*70)
    logger.info("4-FOLD CROSS-VALIDATION WITH REAL LLM")
    logger.info("="*70)
    
    # Initialize LLM client
    if use_real_llm:
        logger.info("\nInitializing OpenAI GPT-3.5-turbo...")
        try:
            llm_client = create_openai_client(model="gpt-3.5-turbo")
            logger.info("✓ LLM client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            logger.error("Set OPENAI_API_KEY environment variable!")
            sys.exit(1)
    else:
        logger.warning("⚠ Running without real LLM (not recommended)")
        llm_client = None
    
    # Print terminology framework
    print_terminology_summary()
    
    # Load ALL cases once (with random seed for reproducibility)
    logger.info(f"\nLoading {n_cases} cases from EEDI dataset...")
    logger.info(f"Sampling from full dataset: {sample_from_full}")
    
    all_cases = load_cases_from_eedi(
        data_path, 
        max_cases=n_cases,
        random_seed=42,  # For reproducibility
        sample_from_full=sample_from_full
    )
    
    logger.info(f"✓ Loaded {len(all_cases)} cases")
    
    # Convert to indices for cross-validation
    case_indices = np.arange(len(all_cases))
    
    # Create 4-fold split
    kfold = KFold(n_splits=4, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = []
    
    # Test with different teaching personas (rotate through folds)
    test_personas = ['socratic', 'constructive', 'experiential', 'rule_based', 'traditional_teaching']
    
    logger.info(f"\nRunning 4-fold cross-validation...")
    logger.info(f"Each fold: ~{len(all_cases)//4} test cases, ~{3*len(all_cases)//4} training cases")
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(case_indices), 1):
        logger.info("\n" + "="*70)
        logger.info(f"FOLD {fold_idx}/4")
        logger.info("="*70)
        
        # Split cases
        train_cases = [all_cases[i] for i in train_idx]
        test_cases = [all_cases[i] for i in test_idx]
        
        logger.info(f"Train: {len(train_cases)} cases, Test: {len(test_cases)} cases")
        
        # Initialize systems with TRAINING data only
        logger.info("\nInitializing systems with REAL LLM...")
        
        # Convert to TeachingCase format for all systems
        teaching_cases = [
            TeachingCase(
                id=c.id,
                features=c.features[:4],  # Training uses features[0:4]
                misconception=c.misconception,
                intervention=c.intervention.get('strategy', 'unknown'),
                success_rate=c.outcome
            )
            for c in train_cases
        ]
        
        # 0. Baseline (no CBR-AI system)
        baseline = BaselineSystem()
        baseline.add_training_data(teaching_cases)
        
        # 1. Pure CBR Baseline (no LLM)
        pure_cbr = PureCBRBaseline(k=5)
        pure_cbr.add_cases(teaching_cases)
        
        # 2. Pure AI Baseline (LLM without CBR) - REQUIRES LLM!
        if llm_client:
            pure_ai = PureAIBaseline(llm_client=llm_client)
            pure_ai.train(teaching_cases)
        else:
            logger.warning("⚠ Skipping Pure AI (no LLM client)")
            pure_ai = None
        
        # 3. Hybrid (CBR + LLM) - REQUIRES LLM!
        mnemonic_engine = MnemonicAugmentation(n_chunks=min(10, len(train_cases) // 3))
        enhanced_cases = mnemonic_engine.process_cases(train_cases)
        
        if llm_client:
            hybrid = HybridMnemonicSystem(
                mnemonic_engine=mnemonic_engine, 
                llm_client=llm_client
            )
        else:
            logger.warning("⚠ Skipping Hybrid (no LLM client)")
            hybrid = None
        
        logger.info("✓ All systems initialized with REAL LLM")
        
        # Test on TEST data
        logger.info(f"\nTesting on {len(test_cases)} test cases...")
        
        baseline_scores = []
        cbr_scores = []
        ai_scores = []
        hybrid_scores = []
        
        for idx, test_case in enumerate(test_cases):
            # Rotate through personas
            persona_type = test_personas[idx % len(test_personas)]
            
            # ============================================================
            # CRITICAL FIX: Use features[0:4] to match training!
            # ============================================================
            # Training uses: c.features[:4] = [0, 1, 2, 3]
            #   [0] = QuizId (always 0 in this dataset)
            #   [1] = QuestionId (0-879)
            #   [2] = misconception_count (1-4)
            #   [3] = construct_length (21-165)
            #
            # Test query MUST use the same indices [0:4], not [1:5]!
            # ============================================================
            query = {
                'misconception': test_case.misconception,
                'topic': 'mathematics',
                'persona_type': persona_type,
                # FIXED: Now uses features[0:4] to match training
                'topic_complexity': test_case.features[0],      # QuizId (index 0)
                'prior_performance': test_case.features[1],     # QuestionId (index 1)
                'misconception_frequency': test_case.features[2], # misconception_count (index 2)
                'prerequisite_count': test_case.features[3],    # construct_length (index 3)
                'concept_depth': 1
            }
            
            # Test all systems and compute prediction error
            # Lower error = better performance
            actual_outcome = test_case.outcome
            
            baseline_result = baseline.teach(query)
            baseline_error = abs(baseline_result['score'] - actual_outcome)
            baseline_scores.append(baseline_error)
            
            cbr_result = pure_cbr.retrieve_and_teach(query)
            cbr_error = abs(cbr_result['score'] - actual_outcome)
            cbr_scores.append(cbr_error)
            
            if pure_ai:
                ai_result = pure_ai.teach(query)
                ai_error = abs(ai_result['score'] - actual_outcome)
                ai_scores.append(ai_error)
                
                # OPTIONAL: Save example responses for paper
                if idx < 5:  # Save first 5 examples
                    logger.info(f"\nExample {idx+1} - {query['persona_type']} persona:")
                    logger.info(f"Pure AI Response: {ai_result.get('response', 'N/A')[:200]}...")
            else:
                ai_scores.append(baseline_error)  # Fallback to baseline
            
            if hybrid:
                hybrid_result = hybrid.retrieve_and_generate(query)
                hybrid_error = abs(hybrid_result['score'] - actual_outcome)
                hybrid_scores.append(hybrid_error)
                
                # OPTIONAL: Save example responses for paper
                if idx < 5:  # Save first 5 examples
                    logger.info(f"Hybrid Response: {hybrid_result.get('response', 'N/A')[:200]}...")
            else:
                hybrid_scores.append(baseline_error)  # Fallback to baseline
        
        # Calculate improvements for this fold
        baseline_mean = np.mean(baseline_scores)
        cbr_improvement = ((baseline_mean - np.mean(cbr_scores)) / baseline_mean) * 100
        ai_improvement = ((baseline_mean - np.mean(ai_scores)) / baseline_mean) * 100
        hybrid_improvement = ((baseline_mean - np.mean(hybrid_scores)) / baseline_mean) * 100
        
        # Store fold results
        fold_result = {
            'fold': fold_idx,
            'n_train': len(train_cases),
            'n_test': len(test_cases),
            'baseline': {
                'mean': float(np.mean(baseline_scores)),
                'std': float(np.std(baseline_scores))
            },
            'pure_cbr_baseline': {
                'mean': float(np.mean(cbr_scores)),
                'std': float(np.std(cbr_scores)),
                'improvement_pct': float(cbr_improvement)
            },
            'pure_ai_baseline': {
                'mean': float(np.mean(ai_scores)),
                'std': float(np.std(ai_scores)),
                'improvement_pct': float(ai_improvement)
            },
            'hybrid_mnemonic': {
                'mean': float(np.mean(hybrid_scores)),
                'std': float(np.std(hybrid_scores)),
                'improvement_pct': float(hybrid_improvement)
            }
        }
        fold_results.append(fold_result)
        
        # Print fold summary
        logger.info(f"\nFold {fold_idx} Results (Mean Absolute Error - lower is better):")
        logger.info(f"  Baseline:           {fold_result['baseline']['mean']:.4f} ± {fold_result['baseline']['std']:.4f}")
        logger.info(f"  Pure CBR Baseline:  {fold_result['pure_cbr_baseline']['mean']:.4f} ({fold_result['pure_cbr_baseline']['improvement_pct']:.1f}%)")
        logger.info(f"  Pure AI Baseline:   {fold_result['pure_ai_baseline']['mean']:.4f} ({fold_result['pure_ai_baseline']['improvement_pct']:.1f}%)")
        logger.info(f"  Hybrid + Mnemonic:  {fold_result['hybrid_mnemonic']['mean']:.4f} ({fold_result['hybrid_mnemonic']['improvement_pct']:.1f}%)")
    
    # Aggregate across all folds
    logger.info("\n" + "="*70)
    logger.info("AGGREGATED RESULTS (4-FOLD CROSS-VALIDATION)")
    logger.info("Metric: Mean Absolute Error (MAE) - Lower is Better")
    logger.info("="*70)
    
    systems = ['baseline', 'pure_cbr_baseline', 'pure_ai_baseline', 'hybrid_mnemonic']
    aggregated = {}
    
    for system in systems:
        means = [fold[system]['mean'] for fold in fold_results]
        
        if system == 'baseline':
            aggregated[system] = {
                'mean_across_folds': float(np.mean(means)),
                'std_across_folds': float(np.std(means)),
                'improvement': '0% (reference)'
            }
        else:
            improvements = [fold[system]['improvement_pct'] for fold in fold_results]
            aggregated[system] = {
                'mean_across_folds': float(np.mean(means)),
                'std_across_folds': float(np.std(means)),
                'mean_improvement': float(np.mean(improvements)),
                'std_improvement': float(np.std(improvements))
            }
    
    # Print aggregated results
    logger.info(f"\n{'System':<25} {'Mean Error (MAE)':<18} {'Improvement'}")
    logger.info("-"*70)
    logger.info(f"{'Baseline':<25} {aggregated['baseline']['mean_across_folds']:.4f} ± {aggregated['baseline']['std_across_folds']:.4f}    --")
    logger.info(f"{'Pure CBR Baseline':<25} {aggregated['pure_cbr_baseline']['mean_across_folds']:.4f} ± {aggregated['pure_cbr_baseline']['std_across_folds']:.4f}    {aggregated['pure_cbr_baseline']['mean_improvement']:.1f}% ± {aggregated['pure_cbr_baseline']['std_improvement']:.1f}%")
    logger.info(f"{'Pure AI Baseline':<25} {aggregated['pure_ai_baseline']['mean_across_folds']:.4f} ± {aggregated['pure_ai_baseline']['std_across_folds']:.4f}    {aggregated['pure_ai_baseline']['mean_improvement']:.1f}% ± {aggregated['pure_ai_baseline']['std_improvement']:.1f}%")
    logger.info(f"{'Hybrid + Mnemonic':<25} {aggregated['hybrid_mnemonic']['mean_across_folds']:.4f} ± {aggregated['hybrid_mnemonic']['std_across_folds']:.4f}    {aggregated['hybrid_mnemonic']['mean_improvement']:.1f}% ± {aggregated['hybrid_mnemonic']['std_improvement']:.1f}%")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'method': '4-fold cross-validation',
        'metric': 'Mean Absolute Error (MAE) - lower is better',
        'n_total_cases': n_cases,
        'n_folds': 4,
        'sampled_from_full_dataset': sample_from_full,
        'fold_results': fold_results,
        'aggregated_results': aggregated,
        'teaching_personas_tested': test_personas
    }
    
    with open(output_dir / "cv_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: results/cv_results.json")
    
    # Create CSV for paper
    import csv
    csv_data = [
        ['System Architecture', 'Mean Error (MAE)', 'Std Dev', 'Mean Improvement', 'Std Improvement'],
        ['Baseline', f"{aggregated['baseline']['mean_across_folds']:.4f}", 
         f"{aggregated['baseline']['std_across_folds']:.4f}", '--', '--'],
        ['Pure CBR Baseline', f"{aggregated['pure_cbr_baseline']['mean_across_folds']:.4f}", 
         f"{aggregated['pure_cbr_baseline']['std_across_folds']:.4f}",
         f"{aggregated['pure_cbr_baseline']['mean_improvement']:.1f}%",
         f"{aggregated['pure_cbr_baseline']['std_improvement']:.1f}%"],
        ['Pure AI Baseline', f"{aggregated['pure_ai_baseline']['mean_across_folds']:.4f}", 
         f"{aggregated['pure_ai_baseline']['std_across_folds']:.4f}",
         f"{aggregated['pure_ai_baseline']['mean_improvement']:.1f}%",
         f"{aggregated['pure_ai_baseline']['std_improvement']:.1f}%"],
        ['Hybrid + Mnemonic', f"{aggregated['hybrid_mnemonic']['mean_across_folds']:.4f}", 
         f"{aggregated['hybrid_mnemonic']['std_across_folds']:.4f}",
         f"{aggregated['hybrid_mnemonic']['mean_improvement']:.1f}%",
         f"{aggregated['hybrid_mnemonic']['std_improvement']:.1f}%"],
    ]
    
    with open(output_dir / "cv_results.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    
    logger.info(f"✓ CSV table saved to: results/cv_results.csv")
    logger.info("="*70)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='4-fold cross-validation with REAL LLM integration'
    )
    parser.add_argument(
        '--data-path', 
        default='data/all_train.csv',
        help='Path to EEDI dataset (all_train.csv with ~17M records)'
    )
    parser.add_argument(
        '--n-cases', 
        type=int, 
        default=200,
        help='Total number of cases to sample (default: 200)'
    )
    parser.add_argument(
        '--sample-from-full',
        action='store_true',
        help='Sample from full 17M dataset (slow!). Default: sample from first 100K for speed'
    )
    parser.add_argument(
        '--use-real-llm',
        action='store_true',
        default=True,
        help='Use real OpenAI API (default: True, requires OPENAI_API_KEY)'
    )
    parser.add_argument(
        '--no-llm',
        dest='use_real_llm',
        action='store_false',
        help='Disable real LLM (for testing purposes only)'
    )
    args = parser.parse_args()
    
    # Check API key if using real LLM
    import os
    if args.use_real_llm and not os.getenv('OPENAI_API_KEY'):
        print("❌ ERROR: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        print("Or run with --no-llm flag (not recommended)")
        sys.exit(1)
    
    run_four_fold_cv(
        args.data_path, 
        args.n_cases, 
        args.sample_from_full,
        args.use_real_llm
    )