import pandas as pd
import numpy as np
from datetime import datetime
import calendar
import os

# ------------------------------------
# 설정
# ------------------------------------
OUTPUT_DIR = "./dataset_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scenario ID → (year, month) 매핑
scenarios = {
    1:(2024,1), 2:(2024,2), 3:(2024,3), 4:(2024,4), 5:(2024,5),
    6:(2024,6), 7:(2024,7), 8:(2024,8), 9:(2024,9), 10:(2024,10),
    11:(2024,11), 12:(2024,12), 13:(2025,1), 14:(2025,2), 15:(2025,3)
}

rng = np.random.default_rng(42)

nurses_rows = []
req_rows = []
pref_rows = []

# ------------------------------------
# 데이터 생성 루프
# ------------------------------------
for sid, (year, month) in scenarios.items():
    days_in_month = calendar.monthrange(year, month)[1]

    # 랜덤하게 팀 존재 여부 결정
    has_team = rng.random() < 0.5
    if has_team:
        n_teams = rng.integers(2,5)
        team_assign = rng.integers(0, n_teams, 40)
    else:
        team_assign = [None]*40

    # 나이트 전담 4~7명
    night_only_count = int(rng.integers(4,8))
    night_only_ids = set(rng.choice(40, night_only_count, replace=False))

    # -------------------------
    # Nurses Table
    # -------------------------
    for nid in range(40):
        nurses_rows.append({
            "scenario_id": sid,
            "nurse_id": nid,
            "experience_years": float(rng.uniform(0.5,15)),
            "is_night_only": 1 if nid in night_only_ids else 0,
            "team_id": team_assign[nid],
            "min_off_per_month": int(rng.integers(7,10))
        })

    # -------------------------
    # Requirements Table
    # -------------------------
    for day in range(1, days_in_month + 1):
        weekday = datetime(year, month, day).weekday()  # 0=Mon, 6=Sun

        # 평일 기본값
        base = {
            "D": rng.integers(12,16),
            "E": rng.integers(12,16),
            "N": rng.integers(10,13)
        }

        # 주말 조정
        if weekday >= 5:
            base["D"] -= 1
            base["E"] -= 1
            base["N"] -= 2

        for shift in ["D","E","N"]:
            req_rows.append({
                "scenario_id": sid,
                "day": day,
                "shift_type": shift,
                "min_staff": int(max(base[shift],1))
            })

    # -------------------------
    # Preferences Table
    # -------------------------
    for nid in range(40):
        for day in range(1, days_in_month + 1):
            r = rng.random()
            if r < 0.10:
                pref_rows.append({"scenario_id": sid, "nurse_id": nid, "day": day, "request_type":"OFF", "weight":1.0})
            elif r < 0.13:
                pref_rows.append({"scenario_id": sid, "nurse_id": nid, "day": day, "request_type":"PREF_D", "weight":1.0})
            elif r < 0.16:
                pref_rows.append({"scenario_id": sid, "nurse_id": nid, "day": day, "request_type":"PREF_E", "weight":1.0})
            elif r < 0.19:
                pref_rows.append({"scenario_id": sid, "nurse_id": nid, "day": day, "request_type":"PREF_N", "weight":1.0})


# ------------------------------------
# DataFrame 변환 & CSV 저장
# ------------------------------------
nurses_df = pd.DataFrame(nurses_rows)
req_df = pd.DataFrame(req_rows)
pref_df = pd.DataFrame(pref_rows)

nurses_df.to_csv(f"{OUTPUT_DIR}/nurses.csv", index=False)
req_df.to_csv(f"{OUTPUT_DIR}/requirements.csv", index=False)
pref_df.to_csv(f"{OUTPUT_DIR}/preferences.csv", index=False)

print("✅ Dataset 생성 완료!")
print(f"- {OUTPUT_DIR}/nurses.csv")
print(f"- {OUTPUT_DIR}/requirements.csv")
print(f"- {OUTPUT_DIR}/preferences.csv")
