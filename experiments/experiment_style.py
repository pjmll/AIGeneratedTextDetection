import sys
from pathlib import Path

project_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_path))

import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from features.spectral_features import SpectralFeatureExtractor
from features.grammar_features import CoEditGrammarExtractor
from models.classifier import AITextClassifier
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
        
        f.write(f"{'文本风格':<30} {'Accuracy':<12} {'F1':<12}\n")
        f.write("-" * 70 + "\n")
        
        for name, metrics in results_dict.items():
            f.write(f"{name:<30} {metrics['Accuracy']:<12.4f} {metrics['F1']:<12.4f}\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"   TXT 已保存: {txt_path}")
    return txt_path


def run_experiment_style():
    print("Running Experiment 3: Different Task / Prompt Styles (Writing vs XSum News)")
    print("=" * 60)
    
    base_dir = "data/normal_data"
    files = {
        "Creative Writing": os.path.join(base_dir, "writing.GPT-4o.normal.test_data.json"),
        "News Summary (XSum)": os.path.join(base_dir, "xsum.GPT-4o.normal.test_data.json"),
    }
    
    features_list = []
    
    # 使用 CoEdit 模式（1维 ROUGE-2）
    spec_ext = SpectralFeatureExtractor(model_name="local_models/gpt2")
    gram_ext = CoEditGrammarExtractor(model_path="local_models/coedit", use_cache=True)
    
    # 临时小批量提取特征
    for style, path in files.items():
        if not os.path.exists(path):
            print(f"Skipping {style}, file not found")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)[:100]  # 取100条演示
            
        for item in tqdm(data, desc=f"Extracting {style}"):
            f_spec = spec_ext.extract_features(item['text'])
            f_gram = gram_ext.extract_features(item['text'])
            lbl = 0 if str(item['label']).lower() == 'human' else 1
            features_list.append([style, lbl, item['text']] + f_spec + f_gram)
            
    if hasattr(gram_ext, 'close'):
        gram_ext.close()
    
    # 14维频谱特征 + 1维 CoEdit 语法特征
    spec_cols = [
        "low_freq_energy", "high_freq_energy", "spec_entropy", "avg_amp", "peak_freq",
        "energy_ratio", "p_mean", "p_var", "p_skew", "p_kurt", 
        "spec_flatness", "autocorr_1", "psd_max", "psd_mean"
    ]
    gram_cols = ["rouge2_similarity"]
    
    cols = ["style", "label", "text"] + spec_cols + gram_cols
    df = pd.DataFrame(features_list, columns=cols)
    
    # Load base training model (trained on LLM data)
    df_train = pd.read_csv("data/features_llms.csv")
    
    # 标签转换
    if df_train['label'].iloc[0] in ['human', 'llm']:
        df_train['label'] = df_train['label'].map({'human': 0, 'llm': 1})
    if df['label'].iloc[0] in ['human', 'llm']:
        df['label'] = df['label'].map({'human': 0, 'llm': 1})
    
    clf = AITextClassifier(grammar_mode='coedit')
    clf.train(df_train[spec_cols + gram_cols], df_train['label'])
    
    results = {}
    for style in ["Creative Writing", "News Summary (XSum)"]:
        sub = df[df['style'] == style]
        if len(sub) == 0: continue
            
        X_sub = sub[spec_cols + gram_cols]
        y_sub = sub['label'].astype(int)
        
        y_pred = clf.predict(X_sub)
        metrics = clf.evaluate(y_sub, y_pred)
        results[style] = {'Accuracy': metrics['accuracy'], 'F1': metrics['f1']}
        print(f"[{style}] Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    # 保存结果到 TXT
    save_results_to_txt(results, "Style Generalization (Writing vs XSum)", "style_comparison")

    # Plot
    labels = list(results.keys())
    accs = [results[l]['Accuracy'] for l in labels]
    f1s = [results[l]['F1'] for l in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6,5))
    ax.bar(x - width/2, accs, width, label='Accuracy')
    ax.bar(x + width/2, f1s, width, label='F1 Score')
    ax.set_ylabel('Scores')
    ax.set_title('Detection Generalization Across Text Styles')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig("results/figures/experiment_style_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 60)
    print("Experiment 3 complete.")
    print("Figure saved to: results/figures/experiment_style_comparison.png")
    print("=" * 60)


if __name__ == "__main__":
    if not os.path.exists("results/figures"):
        os.makedirs("results/figures")
    run_experiment_style()