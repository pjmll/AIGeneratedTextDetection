import sys
from pathlib import Path

project_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_path))

import pandas as pd
from models.classifier import AITextClassifier
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime


def save_results_to_txt(results_dict, experiment_name, filename_prefix="experiment_results"):
    """
    将实验结果保存为 TXT 文件
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    txt_path = results_dir / f"{filename_prefix}_{timestamp}.txt"
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"实验名称: {experiment_name}\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'数据源':<30} {'Accuracy':<12} {'F1':<12}\n")
        f.write("-" * 70 + "\n")
        
        for name, metrics in results_dict.items():
            f.write(f"{name:<30} {metrics['Accuracy']:<12.4f} {metrics['F1']:<12.4f}\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"   TXT 已保存: {txt_path}")
    return txt_path


def run_experiment_attack():
    print("Running Experiment 4: Paraphrase Attack (Human Revise)")
    print("=" * 60)
    
    # Train the base classifier
    df_train = pd.read_csv("data/features_paraphrase.csv")
    
    # 标签转换
    if df_train['label'].iloc[0] in ['human', 'llm']:
        df_train['label'] = df_train['label'].map({'human': 0, 'llm': 1})
    
    spec_cols = [
        "low_freq_energy", "high_freq_energy", "spec_entropy", "avg_amp", "peak_freq",
        "energy_ratio", "p_mean", "p_var", "p_skew", "p_kurt", 
        "spec_flatness", "autocorr_1", "psd_max", "psd_mean"
    ]
    # CoEdit 模式：1维 ROUGE-2 相似度
    gram_cols = ["rouge2_similarity"]
    
    clf = AITextClassifier(grammar_mode='coedit')
    clf.train(df_train[spec_cols + gram_cols], df_train['label'])
    
    if not os.path.exists("data/features_paraphrase.csv"):
        print("Please run 'python data/extract_real_data.py' first!")
        return
        
    df_test = pd.read_csv("data/features_paraphrase.csv")
    
    # 标签转换
    if df_test['label'].iloc[0] in ['human', 'llm']:
        df_test['label'] = df_test['label'].map({'human': 0, 'llm': 1})
    
    # 获取所有数据源
    sources = df_test['source'].unique()
    results = {}
    
    for source in sources:
        sub_df = df_test[df_test['source'] == source]
        X_sub = sub_df[spec_cols + gram_cols]
        y_true = sub_df['label']
        
        y_pred = clf.predict(X_sub)
        metrics = clf.evaluate(y_true, y_pred)
        results[source] = {'Accuracy': metrics['accuracy'], 'F1': metrics['f1']}
        print(f"[{source}] Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    # 保存结果到 TXT
    save_results_to_txt(results, "Paraphrase Attack Detection", "paraphrase_attack")
    
    # 按 Normal/Paraphrase 分组画图
    normal_sources = [s for s in sources if 'Normal' in s]
    para_sources = [s for s in sources if 'Paraphrase' in s]
    
    # 只取有配对的数据（Normal + Paraphrase 成对出现）
    paired_models = []
    for normal in normal_sources:
        model_name = normal.replace('_Normal', '')
        para_name = model_name + '_Paraphrase'
        if para_name in para_sources:
            paired_models.append((model_name, normal, para_name))
    
    if paired_models:
        x = np.arange(len(paired_models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        normal_accs = [results[normal]['Accuracy'] for _, normal, _ in paired_models]
        para_accs = [results[para]['Accuracy'] for _, _, para in paired_models]
        
        rects1 = ax.bar(x - width/2, normal_accs, width, label='Normal AI')
        rects2 = ax.bar(x + width/2, para_accs, width, label='Paraphrased AI')
        
        ax.set_ylabel('Accuracy')
        ax.set_title('Robustness Against Paraphrasing Attacks')
        ax.set_xticks(x)
        ax.set_xticklabels([model for model, _, _ in paired_models])
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig("results/figures/experiment_paraphrase_attack.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # 如果没有配对数据，使用原来的简单对比方式
        sub_normal = df_test[df_test['source'] == 'GPT-4o_Normal']
        sub_para = df_test[df_test['source'] == 'GPT-4o_Paraphrase']
        
        if len(sub_normal) > 0 and len(sub_para) > 0:
            y_pred_norm = clf.predict(sub_normal[spec_cols + gram_cols])
            y_true_norm = sub_normal['label']
            metrics_norm = clf.evaluate(y_true_norm, y_pred_norm)
            
            y_pred_para = clf.predict(sub_para[spec_cols + gram_cols])
            y_true_para = sub_para['label']
            metrics_para = clf.evaluate(y_true_para, y_pred_para)
            
            llms = ['Normal AI', 'Paraphrased AI']
            x = np.arange(len(llms))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(6,5))
            rects1 = ax.bar(x - width/2, [metrics_norm['accuracy'], metrics_para['accuracy']], width, label='Accuracy')
            rects2 = ax.bar(x + width/2, [metrics_norm['f1'], metrics_para['f1']], width, label='F1 Score')
            
            ax.set_ylabel('Scores')
            ax.set_title('Robustness Against Paraphrasing Attacks')
            ax.set_xticks(x)
            ax.set_xticklabels(llms)
            ax.legend(loc='lower left')
            ax.set_ylim([0, 1])
            
            plt.tight_layout()
            plt.savefig("results/figures/experiment_paraphrase_attack.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nStandard Detection F1: {metrics_norm['f1']:.4f}")
            print(f"Paraphrased Detection F1: {metrics_para['f1']:.4f}")
    
    print("\n" + "=" * 60)
    print("Experiment 4 complete.")
    print("Figure saved to: results/figures/experiment_paraphrase_attack.png")
    print("=" * 60)


if __name__ == "__main__":
    if not os.path.exists("results/figures"):
        os.makedirs("results/figures")
    run_experiment_attack()