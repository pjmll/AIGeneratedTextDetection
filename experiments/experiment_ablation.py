import sys
from pathlib import Path

project_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_path))

import pandas as pd
from models.classifier import AITextClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import os
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
        f.write(f"{'特征组合':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
        f.write("-" * 70 + "\n")
        
        for name, metrics in results_dict.items():
            f.write(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"   TXT 已保存: {txt_path}")
    return txt_path


def run_experiment_1():
    print("Running Experiment 1: Ablation Study on LLM Data")
    print("=" * 60)
    
    df = pd.read_csv("data/features_llms.csv")
    print(f"加载数据: {len(df)} 条样本")
    
    # 标签转换
    if df['label'].iloc[0] in ['human', 'llm']:
        df['label'] = df['label'].map({'human': 0, 'llm': 1})
    
    spec_cols = [
        "low_freq_energy", "high_freq_energy", "spec_entropy", "avg_amp", "peak_freq",
        "energy_ratio", "p_mean", "p_var", "p_skew", "p_kurt", 
        "spec_flatness", "autocorr_1", "psd_max", "psd_mean"
    ]
    gram_cols = ["rouge2_similarity"]
    
    features_sets = {
        "Grammar Only": df[gram_cols],
        "Spectral Only": df[spec_cols],
        "Fusion": df[spec_cols + gram_cols]
    }
    
    y = df['label']
    results_dict = {}
    
    plt.figure(figsize=(8,6))
    
    for name, X in features_sets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = AITextClassifier(grammar_mode='coedit')
        clf.train(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba_ai(X_test)
        
        metrics = clf.evaluate(y_test, y_pred)
        results_dict[name] = metrics
        print(f"[{name}] Acc:{metrics['accuracy']:.2f} P:{metrics['precision']:.2f} R:{metrics['recall']:.2f} F1:{metrics['f1']:.2f}")
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        
        if name == "Fusion":
            clf.plot_confusion_matrix(y_test, y_pred, save_path="results/figures/cm_fusion_llms.png")
            clf.plot_feature_importance(save_path="results/figures/feature_imp_llms.png")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - LLM Data')
    plt.legend(loc="lower right")
    plt.savefig("results/figures/roc_curve_llms.png")
    plt.close()
    
    # 保存结果到 TXT
    save_results_to_txt(results_dict, "Ablation Study - LLM Data", "ablation_llms")
    
    print("\nExperiment 1 complete. Figures saved to results/figures/")


def run_experiment_hc3():
    """在 HC3 数据集上运行消融实验"""
    print("\n" + "=" * 60)
    print("Running Experiment 1b: Ablation Study on HC3 Data")
    print("=" * 60)
    
    df = pd.read_csv("data/features_coedit.csv")
    print(f"加载数据: {len(df)} 条样本")
    print(f"标签分布: 0={sum(df['label']==0)}, 1={sum(df['label']==1)}")
    
    spec_cols = [
        "low_freq_energy", "high_freq_energy", "spec_entropy", "avg_amp", "peak_freq",
        "energy_ratio", "p_mean", "p_var", "p_skew", "p_kurt", 
        "spec_flatness", "autocorr_1", "psd_max", "psd_mean"
    ]
    gram_cols = ["rouge2_similarity"]
    
    features_sets = {
        "Grammar Only": df[gram_cols],
        "Spectral Only": df[spec_cols],
        "Fusion": df[spec_cols + gram_cols]
    }
    
    y = df['label']
    results_dict = {}
    
    plt.figure(figsize=(8,6))
    
    for name, X in features_sets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = AITextClassifier(grammar_mode='coedit')
        clf.train(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba_ai(X_test)
        
        metrics = clf.evaluate(y_test, y_pred)
        results_dict[name] = metrics
        print(f"[{name}] Acc:{metrics['accuracy']:.2f} P:{metrics['precision']:.2f} R:{metrics['recall']:.2f} F1:{metrics['f1']:.2f}")
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        
        if name == "Fusion":
            clf.plot_confusion_matrix(y_test, y_pred, save_path="results/figures/cm_fusion_hc3.png")
            clf.plot_feature_importance(save_path="results/figures/feature_imp_hc3.png")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - HC3 Data (with R2)')
    plt.legend(loc="lower right")
    plt.savefig("results/figures/roc_curve_hc3.png")
    plt.close()
    
    # 保存结果到 TXT
    save_results_to_txt(results_dict, "Ablation Study - HC3 Data", "ablation_hc3")
    
    print("\nExperiment 1b complete. Figures saved to results/figures/")


if __name__ == "__main__":
    if not os.path.exists("results/figures"):
        os.makedirs("results/figures")
    
    # 运行两个实验
    run_experiment_1()
    run_experiment_hc3()