import pandas as pd
import numpy as np
import pickle
import os


def load_scenarios_from_csv(
    nurses_path="dataset_output/nurses.csv",
    reqs_path="dataset_output/requirements.csv",
    prefs_path="dataset_output/preferences.csv",
    output_path="data/scenarios.pkl",
):
    """
    CSV 파일들을 읽어 시나리오별 딕셔너리로 구조화하여 저장합니다.
    """
    print("=== Loading CSV Data... ===")
    
    # 1. Read CSVs
    try:
        df_nurses = pd.read_csv(nurses_path)
        df_reqs = pd.read_csv(reqs_path)
        df_prefs = pd.read_csv(prefs_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    # 2. Group by Scenario ID
    scenario_ids = df_nurses['scenario_id'].unique()
    scenarios = {}
    
    total_scenarios = len(scenario_ids)
    print(f"Found {total_scenarios} scenarios. Processing...")

    for i, sc_id in enumerate(scenario_ids):
        # Progress Log
        if (i+1) % 5 == 0:
            print(f"  -> Processing scenario {i+1}/{total_scenarios} (ID: {sc_id})")
            
        # A. Nurses Data
        sc_nurses = df_nurses[df_nurses['scenario_id'] == sc_id].copy()
        nurses_list = []
        for _, row in sc_nurses.iterrows():
            nurses_list.append({
                'id': int(row['nurse_id']),
                'experience': row['experience_years'] / 20.0, # 정규화 (Max 20년 가정)
                'is_night_only': int(row['is_night_only']),
                'team_id': int(row['team_id']) if pd.notna(row['team_id']) else 0,
                'min_off': int(row['min_off_per_month'])
            })
            
        N = len(nurses_list)
        
        # B. Requirements (Min Staff per Day)
        sc_reqs = df_reqs[df_reqs['scenario_id'] == sc_id]
        max_day = sc_reqs['day'].max()
        D = max_day # 1-based index라면 max_day가 곧 일수
        
        # 일자별/근무별 최소 인원: {day_idx: {'D': 6, 'E': 5, ...}}
        min_staff = {}
        for d in range(1, D + 1):
            day_reqs = sc_reqs[sc_reqs['day'] == d]
            day_dict = {'D': 0, 'E': 0, 'N': 0}
            for _, r in day_reqs.iterrows():
                day_dict[r['shift_type']] = int(r['min_staff'])
            min_staff[d-1] = day_dict # 0-based index 변환
            
        # C. Preferences (Requests)
        sc_prefs = df_prefs[df_prefs['scenario_id'] == sc_id]
        requests = {}
        for _, row in sc_prefs.iterrows():
            n_id = int(row['nurse_id'])
            d_idx = int(row['day']) - 1 # 0-based
            r_type = row['request_type']
            
            if n_id not in requests:
                requests[n_id] = {}
            
            # r_type 매핑: OFF -> 'O', PREF_D -> 'D' 등
            if r_type == 'OFF':
                requests[n_id][d_idx] = 'O'
            elif r_type == 'PREF_D':
                requests[n_id][d_idx] = 'D'
            elif r_type == 'PREF_E':
                requests[n_id][d_idx] = 'E'
            elif r_type == 'PREF_N':
                requests[n_id][d_idx] = 'N'

        # D. Config & Initial Roster
        config = {
            'max_consecutive_work': 5,
            'max_consecutive_night': 3,
            'min_staff_dynamic': min_staff # 동적 할당
        }
        
        initial_roster = np.zeros((N, D), dtype=int) # All Off
        
        scenarios[sc_id] = {
            'nurses': nurses_list,
            'config': config,
            'requests': requests,
            'initial_roster': initial_roster,
            'meta': {'N': N, 'D': D, 'id': sc_id}
        }
        
    # 3. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(scenarios, f)
        
    print(f"=== Saved {len(scenarios)} scenarios to {output_path} ===")
    return scenarios

if __name__ == "__main__":
    load_scenarios_from_csv()

