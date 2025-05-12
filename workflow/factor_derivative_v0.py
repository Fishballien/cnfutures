# -*- coding: utf-8 -*-
"""
Factor Testing Pipeline Automation Script

This script automates the execution of factor testing and processing tasks
according to a predefined workflow. It manages directory mappings, task sequencing,
parameter management, and provides execution control through triggers.

Created on: May 7, 2025
"""

import os
import sys
import subprocess
import time
import json
import yaml
import toml
from pathlib import Path
from datetime import datetime
import argparse

# %% 1. Setup

def parse_arguments():
    parser = argparse.ArgumentParser(description='Factor Testing Pipeline Automation')
    # Control parameters
    parser.add_argument('--start_step', type=int, default=1, help='Start execution from this step (default: 1)')
    parser.add_argument('--skip_steps', type=str, default='', help='Comma-separated list of steps to skip')
    
    # Base parameters
    parser.add_argument('--ind_cate_name', type=str, required=True, help='Indicator category name')
    parser.add_argument('--org_name', type=str, required=True, help='Organization name')
    parser.add_argument('--batch_path_name', type=str, required=True, help='Batch path name')
    
    # Directory parameters
    parser.add_argument('--factor_factory_dir', type=str, required=True, help='Path to factor_factory directory')
    parser.add_argument('--factor_test_dir', type=str, required=True, help='Path to factor_test directory')
    parser.add_argument('--factor_dir', type=str, required=True, help='Path to factors directory')
    parser.add_argument('--config_dir', type=str, default='configs', help='Directory to save configuration files')
    
    # Date parameters for Step 2: test basic factors
    parser.add_argument('--test_basic_date_start', type=str, default='20210101', help='Start date for testing basic factors')
    parser.add_argument('--test_basic_date_end', type=str, default='20250401', help='End date for testing basic factors')
    
    # Date parameters for Step 3: rolling eval basic
    parser.add_argument('--roll_eval_basic_pstart', type=str, default='20230701', help='Prediction start date for rolling evaluation of basic factors')
    parser.add_argument('--roll_eval_basic_puntil', type=str, default='20250401', help='Prediction until date for rolling evaluation of basic factors')
    
    # Date parameters for Step 4: rolling select basic
    parser.add_argument('--roll_select_basic_pstart', type=str, default='20230701', help='Prediction start date for rolling selection of basic factors')
    parser.add_argument('--roll_select_basic_puntil', type=str, default='20250401', help='Prediction until date for rolling selection of basic factors')
    
    # Date parameters for Step 4-1-1: rolling merge basic
    parser.add_argument('--roll_merge_basic_pstart', type=str, default='20230701', help='Prediction start date for rolling merge of basic factors')
    parser.add_argument('--roll_merge_basic_puntil', type=str, default='20250401', help='Prediction until date for rolling merge of basic factors')
    
    # Date parameters for Step 4-1-2: rolling select trade method
    parser.add_argument('--roll_select_trade_pstart', type=str, default='20230701', help='Prediction start date for rolling select trade method')
    parser.add_argument('--roll_select_trade_puntil', type=str, default='20250401', help='Prediction until date for rolling select trade method')
    
    # Date parameters for Step 4-2-2: batch test selected
    parser.add_argument('--batch_test_selected_date_start', type=str, default='20210101', help='Start date for batch test selected')
    parser.add_argument('--batch_test_selected_date_end', type=str, default='20250401', help='End date for batch test selected')
    
    # Date parameters for Step 4-2-3: rolling eval ts
    parser.add_argument('--roll_eval_ts_pstart', type=str, default='20230701', help='Prediction start date for rolling evaluation of time series')
    parser.add_argument('--roll_eval_ts_puntil', type=str, default='20250401', help='Prediction until date for rolling evaluation of time series')
    
    # Date parameters for Step 4-2-4: rolling select ts
    parser.add_argument('--roll_select_ts_pstart', type=str, default='20230701', help='Prediction start date for rolling selection of time series')
    parser.add_argument('--roll_select_ts_puntil', type=str, default='20250401', help='Prediction until date for rolling selection of time series')
    
    # Date parameters for Step 4-2-5: rolling merge ts
    parser.add_argument('--roll_merge_ts_pstart', type=str, default='20230701', help='Prediction start date for rolling merge of time series')
    parser.add_argument('--roll_merge_ts_puntil', type=str, default='20250401', help='Prediction until date for rolling merge of time series')
    
    # Date parameters for Step 4-2-6: rolling select trade method ts
    parser.add_argument('--roll_select_trade_ts_pstart', type=str, default='20230701', help='Prediction start date for rolling select trade method time series')
    parser.add_argument('--roll_select_trade_ts_puntil', type=str, default='20250401', help='Prediction until date for rolling select trade method time series')
    
    # Worker parameters
    parser.add_argument('--test_wkr', type=int, default=4, help='Number of test workers')
    parser.add_argument('--eval_wkr', type=int, default=4, help='Number of evaluation workers')
    parser.add_argument('--merge_wkr', type=int, default=4, help='Number of merge workers')
    
    # Selection parameters
    parser.add_argument('--select_version', type=str, required=True, help='Selection version')
    parser.add_argument('--merge_version', type=str, required=True, help='Merge version')
    parser.add_argument('--select_trade_method_version', type=str, required=True, help='Select trade method version')
    
    # Test parameters
    parser.add_argument('--batch_test_name', type=str, required=True, help='Batch test name')
    parser.add_argument('--rolling_eval_name', type=str, required=True, help='Rolling evaluation name')
    parser.add_argument('--rolling_select_name', type=str, required=True, help='Rolling selection name')
    parser.add_argument('--rolling_merge_name', type=str, required=True, help='Rolling merge name')
    parser.add_argument('--rolling_select_trade_method_name', type=str, required=True, help='Rolling select trade method name')
    
    # Time series parameters
    parser.add_argument('--generate_batch_config_name', type=str, required=True, help='Generate batch config name')
    parser.add_argument('--batch_test_name_ts', type=str, required=True, help='Batch test name for time series')
    parser.add_argument('--rolling_eval_name_ts', type=str, required=True, help='Rolling evaluation name for time series')
    parser.add_argument('--rolling_select_name_ts', type=str, required=True, help='Rolling selection name for time series')
    parser.add_argument('--rolling_merge_name_ts', type=str, required=True, help='Rolling merge name for time series')
    parser.add_argument('--rolling_select_trade_method_name_ts', type=str, required=True, 
                        help='Rolling select trade method name for time series')
    parser.add_argument('--select_version_ts', type=str, required=True, help='Selection version for time series')
    parser.add_argument('--merge_version_ts', type=str, required=True, help='Merge version for time series')
    parser.add_argument('--select_trade_method_version_ts', type=str, required=True, 
                        help='Select trade method version for time series')
    
    # Factor parameters
    parser.add_argument('--org_fac_name', type=str, default='', help='Original factor name (single factor)')
    parser.add_argument('--org_fac_name_list', type=str, default='', help='Comma-separated list of original factor names')
    
    return parser.parse_args()

# %% Directory Mapping
def setup_directory_mapping(args):
    # Define directory paths for different components
    return {
        'factor_factory': args.factor_factory_dir,
        'factor_test': args.factor_test_dir,
        'factor_dir': args.factor_dir,
    }

# %% Task Mapping
def setup_task_mapping():
    # Map step numbers to their execution details (directory and script)
    return {
        1: ('factor_factory', 'run_ts_trans_by_path_batch_multi_factors'),
        2: ('factor_test', 'run_batch_test_by_path'),
        3: ('factor_test', 'run_rolling_eval'),
        '3-1': ('factor_test', 'analysis/eval_analysis/analysis_eval_by_path_and_test'),
        4: ('factor_test', 'run_rolling_select_basic_features'),
        '4-1-1': ('factor_test', 'run_rolling_merge_selected_basic_features'),
        '4-1-2': ('factor_test', 'run_rolling_select_trade_method'),
        '4-2-1': ('factor_factory', 'run_generate_batch_from_selected_basic_fac'),
        '4-2-2': ('factor_test', 'run_batch_test_by_all_selected'),
        '4-2-3': ('factor_test', 'run_rolling_eval'),
        '4-2-4': ('factor_test', 'run_rolling_select_ts_trans'),
        '4-2-5': ('factor_test', 'run_rolling_merge_selected_ts_trans'),
        '4-2-6': ('factor_test', 'run_rolling_select_trade_method'),
    }

# %% Derived Parameters
def generate_derived_parameters(params):
    """Generate derived parameters based on input parameters"""
    derived = {}
    
    # Basic evaluation name
    derived['final_path_name'] = f"{params['org_name']}_{params['batch_path_name']}"
    
    # Step 3: Rolling evaluation name
    derived['eval_name'] = f"{params['ind_cate_name']}_{derived['final_path_name']}_{params['batch_test_name']}"
    
    # Step 4: Select name with version
    derived['select_name'] = f"{derived['eval_name']}_{params['select_version']}"
    
    # Step 4-1-1: Merge name with version
    derived['merge_name'] = f"{derived['select_name']}_{params['merge_version']}"
    
    # Step 4-1-2: Select trade method name
    derived['select_trade_method_name'] = f"{derived['merge_name']}_{params['select_trade_method_version']}"
    
    # Step 4-2-3: Time series evaluation name
    derived['eval_name_ts'] = f"{params['generate_batch_config_name']}_{params['batch_test_name_ts']}_{derived['eval_name']}"
    
    # Step 4-2-4: Time series select name
    derived['select_name_ts'] = f"{derived['eval_name_ts']}_{params['select_version_ts']}"
    
    # Step 4-2-5: Time series merge name
    derived['merge_name_ts'] = f"{derived['select_name_ts']}_{params['merge_version_ts']}"
    
    # Step 4-2-6: Time series select trade method name
    derived['select_trade_method_name_ts'] = f"{derived['merge_name_ts']}_{params['select_trade_method_version_ts']}"
    
    return derived

# %% Helper Functions
def create_config_file(params, step, template_path, output_path):
    """Create or modify configuration files based on templates"""
    # Implementation depends on specific format requirements
    try:
        # Load template
        if template_path.endswith('.toml'):
            with open(template_path, 'r') as f:
                config = toml.load(f)
        elif template_path.endswith('.yaml') or template_path.endswith('.yml'):
            with open(template_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            with open(template_path, 'r') as f:
                config = json.load(f)
        
        # Modify config based on step-specific requirements
        # This would need to be customized for each type of config file
        
        # Save modified config
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if output_path.endswith('.toml'):
            with open(output_path, 'w') as f:
                toml.dump(config, f)
        elif output_path.endswith('.yaml') or output_path.endswith('.yml'):
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=4)
                
        return True
    except Exception as e:
        print(f"Error creating config file: {e}")
        return False

def get_original_factor_names(factor_dir, ind_cate_name, org_name):
    """Get list of original factor names from the directory"""
    base_path = Path(factor_dir) / ind_cate_name / org_name
    if not os.path.exists(base_path):
        print(f"Error: Path not found: {base_path}")
        return []
    
    return [f.stem for f in Path(base_path).glob("*.parquet")]

def run_process(directory, script, args, cwd=None):
    """Run a subprocess with the given arguments"""
    if cwd is None:
        cwd = os.getcwd()
    
    full_command = [sys.executable, script] + args
    print(f"Running: {' '.join(full_command)}")
    print(f"Working directory: {cwd}")
    
    try:
        process = subprocess.Popen(
            full_command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Real-time output processing
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Get return code
        return_code = process.poll()
        
        # Print any errors if the command failed
        if return_code != 0:
            error_output = process.stderr.read()
            print(f"Error (code {return_code}):\n{error_output}")
            return False
        
        return True
    except Exception as e:
        print(f"Error running process: {e}")
        return False

# %% 2. Execution Functions

# %% 1. Basic Transformation
def step_1_basic_transformation(params, dir_map):
    """Run the basic transformation on original factors"""
    print("\n========== Step 1: Basic Factor Transformation ==========")
    
    args = [
        "--batch_path_name", params["batch_path_name"],
        "--ind_cate_name", params["ind_cate_name"],
        "--org_name", params["org_name"]
    ]
    
    script_path = os.path.join(dir_map["factor_factory"], "run_ts_trans_by_path_batch_multi_factors.py")
    return run_process("factor_factory", script_path, args, cwd=dir_map["factor_factory"])

# %% 2. Test Basic Transformed Factors
def step_2_test_basic_factors(params, derived_params, dir_map):
    """Test the basic transformed factors"""
    print("\n========== Step 2: Test Basic Transformed Factors ==========")
    
    args = [
        "--ind_cate_name", params["ind_cate_name"],
        "--final_path_name", derived_params["final_path_name"],
        "--tag_name", "default",
        "--batch_test_name", params["batch_test_name"]
    ]
    
    if params.get("test_basic_date_start") and params.get("test_basic_date_end"):
        args.extend([
            "--date_start", params["test_basic_date_start"],
            "--date_end", params["test_basic_date_end"]
        ])
    
    script_path = os.path.join(dir_map["factor_test"], "run_batch_test_by_path.py")
    return run_process("factor_test", script_path, args, cwd=dir_map["factor_test"])

# %% 3. Rolling Evaluation of Basic Factors
def step_3_rolling_eval_basic(params, derived_params, dir_map):
    """Rolling evaluation of basic transformation factors"""
    print("\n========== Step 3: Rolling Evaluation of Basic Factors ==========")
    
    args = [
        "--eval_name", derived_params["eval_name"],
        "--eval_rolling_name", params["rolling_eval_name"],
        "--pstart", params["roll_eval_basic_pstart"],
        "--puntil", params["roll_eval_basic_puntil"]
    ]
    
    script_path = os.path.join(dir_map["factor_test"], "run_rolling_eval.py")
    return run_process("factor_test", script_path, args, cwd=dir_map["factor_test"])

# %% 3-1. Full Sample Factor Evaluation Analysis
def step_3_1_factor_eval_analysis(params, derived_params, dir_map):
    """Analysis of full sample factor evaluation"""
    print("\n========== Step 3-1: Full Sample Factor Evaluation Analysis ==========")
    
    # This step would need a specific implementation depending on the analysis script
    print("Note: Step 3-1 is not implemented and would need customization")
    return True  # Skip for now

# %% 4. Basic Factors Selection
def step_4_rolling_select_basic(params, derived_params, dir_map):
    """Rolling selection of basic factors"""
    print("\n========== Step 4: Basic Factors Selection ==========")
    
    args = [
        "--select_name", derived_params["select_name"],
        "--rolling_select_name", params["rolling_select_name"],
        "--pstart", params["roll_select_basic_pstart"],
        "--puntil", params["roll_select_basic_puntil"],
        "--mode", "rolling"
    ]
    
    script_path = os.path.join(dir_map["factor_test"], "run_rolling_select_basic_features.py")
    return run_process("factor_test", script_path, args, cwd=dir_map["factor_test"])

# %% 4-1-1. Rolling Merge Selected Basic Factors
def step_4_1_1_rolling_merge_basic(params, derived_params, dir_map):
    """Rolling merge of selected basic factors"""
    print("\n========== Step 4-1-1: Rolling Merge Selected Basic Factors ==========")
    
    args = [
        "--merge_name", derived_params["merge_name"],
        "--rolling_merge_name", params["rolling_merge_name"],
        "--pstart", params["roll_merge_basic_pstart"],
        "--puntil", params["roll_merge_basic_puntil"],
        "--mode", "rolling",
        "--max_workers", str(params["merge_wkr"])
    ]
    
    script_path = os.path.join(dir_map["factor_test"], "run_rolling_merge_selected_basic_features.py")
    return run_process("factor_test", script_path, args, cwd=dir_map["factor_test"])

# %% 4-1-2. Rolling Select Trade Method
def step_4_1_2_rolling_select_trade(params, derived_params, dir_map):
    """Rolling selection of trade methods for basic factors"""
    print("\n========== Step 4-1-2: Rolling Select Trade Method ==========")
    
    args = [
        "--select_name", derived_params["select_trade_method_name"],
        "--rolling_select_name", params["rolling_select_trade_method_name"],
        "--pstart", params["roll_select_trade_pstart"],
        "--puntil", params["roll_select_trade_puntil"],
        "--mode", "rolling"
    ]
    
    script_path = os.path.join(dir_map["factor_test"], "run_rolling_select_trade_method.py")
    return run_process("factor_test", script_path, args, cwd=dir_map["factor_test"])

# %% 4-2-1. Generate Batch from Selected Basic Factors
def step_4_2_1_generate_batch(params, derived_params, dir_map, org_fac_names):
    """Generate batch from selected basic factors"""
    print("\n========== Step 4-2-1: Generate Batch from Selected Basic Factors ==========")
    
    if not org_fac_names:
        print("Error: No original factor names provided")
        return False
        
    results = []
    for org_fac_name in org_fac_names:
        print(f"Processing original factor: {org_fac_name}")
        
        # Update configuration for this specific factor if needed
        config_params = params.copy()
        config_params["org_fac_name"] = org_fac_name
        
        args = [
            "--generate_batch_config_name", params["generate_batch_config_name"],
        ]
        
        script_path = os.path.join(dir_map["factor_factory"], "run_generate_batch_from_selected_basic_fac.py")
        result = run_process("factor_factory", script_path, args, cwd=dir_map["factor_factory"])
        results.append(result)
        
        if not result:
            print(f"Failed to process original factor: {org_fac_name}")
            # Decide whether to continue or abort based on your requirements
            
    return all(results)  # Return True only if all succeeded

# %% 4-2-2. Batch Test Selected Basic Factors
def step_4_2_2_batch_test_selected(params, derived_params, dir_map):
    """Batch test all selected basic factors"""
    print("\n========== Step 4-2-2: Batch Test Selected Basic Factors ==========")
    
    args = [
        "--generate_batch_name", params["generate_batch_config_name"],
        "--tag_name", "default",
        "--batch_test_name", params["batch_test_name_ts"],
        "--base_eval_name", derived_params["eval_name"],
        "--date_start", params["batch_test_selected_date_start"],
        "--date_end", params["batch_test_selected_date_end"],
        "--eval_wkr", str(params["eval_wkr"])
    ]
    
    script_path = os.path.join(dir_map["factor_test"], "run_batch_test_by_all_selected.py")
    return run_process("factor_test", script_path, args, cwd=dir_map["factor_test"])

# %% 4-2-3. Rolling Evaluation of Time Series Derived Factors
def step_4_2_3_rolling_eval_ts(params, derived_params, dir_map):
    """Rolling evaluation of time series derived factors"""
    print("\n========== Step 4-2-3: Rolling Evaluation of Time Series Derived Factors ==========")
    
    args = [
        "--eval_name", derived_params["eval_name_ts"],
        "--eval_rolling_name", params["rolling_eval_name_ts"],
        "--pstart", params["roll_eval_ts_pstart"],
        "--puntil", params["roll_eval_ts_puntil"],
        "--eval_type", "rolling",
        "--n_workers", str(params["eval_wkr"])
    ]
    
    script_path = os.path.join(dir_map["factor_test"], "run_rolling_eval.py")
    return run_process("factor_test", script_path, args, cwd=dir_map["factor_test"])

# %% 4-2-4. Rolling Selection of Time Series Derived Factors
def step_4_2_4_rolling_select_ts(params, derived_params, dir_map):
    """Rolling selection of time series derived factors"""
    print("\n========== Step 4-2-4: Rolling Selection of Time Series Derived Factors ==========")
    
    args = [
        "--select_name", derived_params["select_name_ts"],
        "--rolling_select_name", params["rolling_select_name_ts"],
        "--pstart", params["roll_select_ts_pstart"],
        "--puntil", params["roll_select_ts_puntil"],
        "--mode", "rolling"
    ]
    
    script_path = os.path.join(dir_map["factor_test"], "run_rolling_select_ts_trans.py")
    return run_process("factor_test", script_path, args, cwd=dir_map["factor_test"])

# %% 4-2-5. Rolling Merge Selected Time Series Factors
def step_4_2_5_rolling_merge_ts(params, derived_params, dir_map):
    """Rolling merge of selected time series factors"""
    print("\n========== Step 4-2-5: Rolling Merge Selected Time Series Factors ==========")
    
    args = [
        "--merge_name", derived_params["merge_name_ts"],
        "--rolling_merge_name", params["rolling_merge_name_ts"],
        "--pstart", params["roll_merge_ts_pstart"],
        "--puntil", params["roll_merge_ts_puntil"],
        "--mode", "rolling",
        "--max_workers", str(params["merge_wkr"]) 
    ]
    
    script_path = os.path.join(dir_map["factor_test"], "run_rolling_merge_selected_ts_trans.py")
    return run_process("factor_test", script_path, args, cwd=dir_map["factor_test"])

# %% 4-2-6. Rolling Select Trade Method for Time Series
def step_4_2_6_rolling_select_trade_ts(params, derived_params, dir_map):
    """Rolling selection of trade methods for time series factors"""
    print("\n========== Step 4-2-6: Rolling Select Trade Method for Time Series ==========")
    
    args = [
        "--select_name", derived_params["select_trade_method_name_ts"],
        "--rolling_select_name", params["rolling_select_trade_method_name_ts"],
        "--pstart", params["roll_select_trade_ts_pstart"],
        "--puntil", params["roll_select_trade_ts_puntil"],
        "--mode", "rolling"
    ]
    
    script_path = os.path.join(dir_map["factor_test"], "run_rolling_select_trade_method.py")
    return run_process("factor_test", script_path, args, cwd=dir_map["factor_test"])

# %% 3. Main Execution Function
def run_pipeline(args):
    """Main function to run the pipeline"""
    print("=== Factor Testing Pipeline Automation ===")
    print(f"Starting execution at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    dir_map = setup_directory_mapping(args)
    task_map = setup_task_mapping()
    
    # Convert parameters
    params = vars(args)
    start_step = params.pop('start_step')
    skip_steps = [s.strip() for s in params.pop('skip_steps').split(',') if s.strip()]
    
    # Process org_fac_name/org_fac_name_list
    org_fac_names = []
    if params['org_fac_name']:
        org_fac_names = [params['org_fac_name']]
    elif params['org_fac_name_list']:
        org_fac_names = [s.strip() for s in params['org_fac_name_list'].split(',') if s.strip()]
    else:
        # Auto-detect from directory
        org_fac_names = get_original_factor_names(dir_map['factor_dir'], params['ind_cate_name'], params['org_name'])
        if not org_fac_names:
            print("Error: Could not determine original factor names")
            return False
    
    print(f"Original factor names: {org_fac_names}")
    
    # Generate derived parameters
    derived_params = generate_derived_parameters(params)
    
    # Save configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f"{params['ind_cate_name']}_{timestamp}.json"
    config_path = os.path.join(params['config_dir'], config_filename)
    
    os.makedirs(params['config_dir'], exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump({
            'params': params, 
            'derived_params': derived_params,
            'org_fac_names': org_fac_names,
            'timestamp': timestamp,
            'start_step': start_step,
            'skip_steps': skip_steps
        }, f, indent=4)
    
    print(f"Configuration saved to: {config_path}")
    
    # Execute tasks
    steps = [
        ('1', step_1_basic_transformation),
        ('2', step_2_test_basic_factors),
        ('3', step_3_rolling_eval_basic),
        ('3-1', step_3_1_factor_eval_analysis),
        ('4', step_4_rolling_select_basic),
        ('4-1-1', step_4_1_1_rolling_merge_basic),
        ('4-1-2', step_4_1_2_rolling_select_trade),
        ('4-2-1', step_4_2_1_generate_batch),
        ('4-2-2', step_4_2_2_batch_test_selected),
        ('4-2-3', step_4_2_3_rolling_eval_ts),
        ('4-2-4', step_4_2_4_rolling_select_ts),
        ('4-2-5', step_4_2_5_rolling_merge_ts),
        ('4-2-6', step_4_2_6_rolling_select_trade_ts),
    ]
    
    for step_id, step_func in steps:
        # Convert step_id to number for comparison (if numeric)
        step_num = float(step_id) if step_id.replace('-', '').isdigit() else step_id
        
        # Skip steps before start_step or explicitly skipped
        if ((isinstance(step_num, (int, float)) and step_num < start_step) or 
            step_id in skip_steps):
            print(f"Skipping step {step_id}")
            continue
        
        print(f"\n=== Executing Step {step_id} ===")
        
        # Special handling for steps that work with individual factors
        if step_id == '4-2-1':
            success = step_func(params, derived_params, dir_