import sys
from pathlib import Path

project_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_path))

import pandas as pd
from models.classifier import AITextClassifier
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime


def save_results_to_txt(results_dict, experiment_name, filename_prefix="experiment_results"):
    """
    将实验结果保存为 TXT 文件
    
    Args:
        results_dict: 包含实验结果的字典
        experiment_name: 实验名称
        filename_prefix: 文件名前缀
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建 results 目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 保存为 TXT
    txt_path = results_dir / f"{filename_prefix}_{timestamp}.txt"
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"实验名称: {experiment_name}\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # 写入结果表格
        f.write(f"{'LLM':<20} {'Split':<12} {'Accuracy':<12} {'F1':<12}\n")
        f.write("-" * 70 + "\n")
        
        for name, metrics in results_dict.items():
            f.write(f"{name:<20} {metrics['Split']:<12} {metrics['Accuracy']:<12.4f} {metrics['F1']:<12.4f}\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"   TXT 已保存: {txt_path}")
    return txt_path


def run_experiment_llms():
    print("Running Experiment 2: Different LLMs Detection")
    print("=" * 60)

    all_data = pd.read_csv("data/features_llms.csv")

    # 标签转换
    if all_data['label'].iloc[0] in ['human', 'llm']:
        all_data['label'] = all_data['label'].map({'human': 0, 'llm': 1})

    train_sources = ["GPT-4o", "Claude-3.5", "Llama-3-70B"]
    test_sources = ["Google-PALM", "gpt3.5"]
    
    spec_cols = [
        "low_freq_energy", "high_freq_energy", "spec_entropy", "avg_amp", "peak_freq",
        "energy_ratio", "p_mean", "p_var", "p_skew", "p_kurt", 
        "spec_flatness", "autocorr_1", "psd_max", "psd_mean"
    ]
    # CoEdit 模式：1维 ROUGE-2 相似度
    gram_cols = ["rouge2_similarity"]
    
    df_train = all_data[all_data['source'].isin(train_sources)].copy()
    df_test = all_data[all_data['source'].isin(test_sources)].copy()

    print(f"训练源: {', '.join(train_sources)}")
    print(f"测试源: {', '.join(test_sources)}")
    print(f"训练样本数: {len(df_train)}")
    print(f"测试样本数: {len(df_test)}")

    if len(df_train) == 0 or len(df_test) == 0:
        print("训练集或测试集为空，请检查 source 划分。")
        return

    # 训练分类器（仅使用指定训练源）
    clf = AITextClassifier(grammar_mode='coedit')
    clf.train(df_train[spec_cols + gram_cols], df_train['label'])

    results = {}

    eval_splits = [
        ("train", train_sources, "In-Domain"),
        ("test", test_sources, "Held-Out")
    ]

    for _, source_list, split_name in eval_splits:
        for llm in source_list:
            sub_df = all_data[all_data['source'] == llm]
            if len(sub_df) == 0:
                continue

            X_sub = sub_df[spec_cols + gram_cols]
            y_sub = sub_df['label']

            y_pred = clf.predict(X_sub)
            metrics = clf.evaluate(y_sub, y_pred)
            results[llm] = {
                'Split': split_name,
                'Accuracy': metrics['accuracy'],
                'F1': metrics['f1']
            }
            print(f"[{split_name}][{llm}] Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    # 保存结果到 TXT
    save_results_to_txt(results, "Different LLMs Detection", "llm_comparison")

    # Plotting
    labels = list(results.keys())
    accs = [results[l]['Accuracy'] for l in labels]
    f1s = [results[l]['F1'] for l in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8,6))
    rects1 = ax.bar(x - width/2, accs, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, f1s, width, label='F1 Score')
    
    ax.set_ylabel('Scores')
    ax.set_title('Detection Performance Across Different LLMs')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig("results/figures/experiment_llm_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 60)
    print("Experiment 2 complete.")
    print("Figure saved to: results/figures/experiment_llm_comparison.png")
    print("=" * 60)


if __name__ == "__main__":
    if not os.path.exists("results/figures"):
        os.makedirs("results/figures")
    run_experiment_llms()