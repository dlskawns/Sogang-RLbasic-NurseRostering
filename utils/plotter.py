import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os


def plot_experiment_results(log_dir: str = "logs") -> None:
    """
    logs 폴더의 모든 CSV를 읽어 알고리즘별 학습 곡선과 요약 바 차트를 그립니다.

    생성되는 이미지:
    - results_learning_curves.png : Total Reward / Hard Violations 곡선
    - results_summary_bar.png     : 알고리즘별 Best Hard / Best Score 바 차트
    """

    # 1. Load Data
    all_files = glob.glob(os.path.join(log_dir, "*.csv"))
    if not all_files:
        print("No log files found.")
        return

    dfs = []
    for f in all_files:
        try:
            filename = os.path.basename(f)             # 예: ppo_seed42_2025...
            algo_name = filename.split("_")[0].upper() # POO / DQN / ...
            mtime = os.path.getmtime(f)

            tmp = pd.read_csv(f)
            tmp["Algorithm"] = algo_name
            tmp["run"] = filename
            tmp["mtime"] = mtime
            # 숫자 컬럼 캐스팅 (일부가 문자열일 수 있음)
            for col in ["episode", "hard_violations", "total_reward", "final_score"]:
                if col in tmp.columns:
                    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

            dfs.append(tmp)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        print("No valid logs parsed.")
        return

    df = pd.concat(dfs, ignore_index=True)

    sns.set_theme(style="darkgrid")

    # 2. Learning Curves (모든 실행 포함, 알고리즘별 평균 + 표준편차)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.lineplot(
        data=df,
        x="episode",
        y="total_reward",
        hue="Algorithm",
        ax=axes[0],
        errorbar="sd",
    )
    axes[0].set_title("Learning Curve (Total Reward)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")

    sns.lineplot(
        data=df,
        x="episode",
        y="hard_violations",
        hue="Algorithm",
        ax=axes[1],
        errorbar="sd",
    )
    axes[1].set_title("Constraint Violations (Lower is Better)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Hard Violations Count")

    plt.tight_layout()
    lc_path = "results_learning_curves.png"
    plt.savefig(lc_path)
    print(f"[Saved] {lc_path}")
    plt.close(fig)

    # 3. 알고리즘별 Best Hard / Best Score 요약 바 차트
    summary_rows = []
    for algo, g_algo in df.groupby("Algorithm"):
        # 최신 실행(run) 하나만 선택 (가장 최근 mtime)
        latest_run = g_algo.sort_values("mtime")["run"].iloc[-1]
        g = g_algo[g_algo["run"] == latest_run].copy()

        # Hard Violations 최소인 지점 중 Score 최대인 에피소드 선택
        if "hard_violations" not in g.columns or g["hard_violations"].isna().all():
            continue
        min_hard = g["hard_violations"].min()
        cand = g[g["hard_violations"] == min_hard]
        best_row = cand.loc[cand["final_score"].astype(float).idxmax()]

        summary_rows.append(
            {
                "Algorithm": algo,
                "best_hard": int(best_row["hard_violations"]),
                "best_score": float(best_row["final_score"]),
            }
        )

    if not summary_rows:
        print("No summary rows computed; skip bar plot.")
        return

    summary_df = pd.DataFrame(summary_rows)
    print("\n[Summary]\n", summary_df)

    # Bar plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(data=summary_df, x="Algorithm", y="best_hard", ax=axes[0])
    axes[0].set_title("Best Hard Violations (Lower is Better)")
    axes[0].set_ylabel("Hard Violations")

    sns.barplot(data=summary_df, x="Algorithm", y="best_score", ax=axes[1])
    axes[1].set_title("Best Final Score (Higher is Better)")
    axes[1].set_ylabel("Final Score")

    plt.tight_layout()
    bar_path = "results_summary_bar.png"
    plt.savefig(bar_path)
    print(f"[Saved] {bar_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_experiment_results()

