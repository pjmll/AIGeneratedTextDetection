from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib


class AITextClassifier:
    def __init__(self, grammar_mode='statistical'):
        self.model = None
        self.single_model = None
        self.spec_model = None
        self.gram_model = None
        self.feature_type = None
        self.feature_names_ = None
        self.spec_cols_ = []
        self.gram_cols_ = []
        self.grammar_mode = grammar_mode
        self.spec_weight_ = 0.75
        self.gram_weight_ = 0.25
        self.threshold_ = 0.5
        self.best_validation_f1_ = None

        if grammar_mode == 'coedit':
            self.gram_cols = ["rouge2_similarity"]
        else:
            self.gram_cols = [
                "grammar_errors", "spelling_errors", "grammar_rate", "spelling_rate",
                "sent_len_var", "punct_entropy", "word_count"
            ]

    def train(self, X_train, y_train):
        if hasattr(X_train, 'columns'):
            columns = list(X_train.columns)
        else:
            columns = [f'feature_{i}' for i in range(X_train.shape[1])]
        self.feature_names_ = columns

        spec_cols = [
            "low_freq_energy", "high_freq_energy", "spec_entropy", "avg_amp", "peak_freq",
            "energy_ratio", "p_mean", "p_var", "p_skew", "p_kurt",
            "spec_flatness", "autocorr_1", "psd_max", "psd_mean"
        ]

        self.spec_cols_ = [col for col in spec_cols if col in columns]
        self.gram_cols_ = [col for col in self.gram_cols if col in columns]

        has_spec = len(self.spec_cols_) > 0
        has_gram = len(self.gram_cols_) > 0

        if has_spec and has_gram:
            self.feature_type = 'fusion'
            self._train_fusion_model(X_train, y_train)
        else:
            self.feature_type = 'spectral_only' if has_spec else 'grammar_only' if has_gram else 'unknown'
            self._train_single_model(X_train, y_train)

    def _create_spec_model(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', ExtraTreesClassifier(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=5,
                random_state=42
            ))
        ])

    def _create_gram_model(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ))
        ])

    def _create_single_model(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=200,
                max_depth=3,
                min_samples_leaf=5,
                random_state=42
            ))
        ])

    def _train_fusion_model(self, X_train, y_train):
        X_spec = X_train[self.spec_cols_]
        X_gram = X_train[self.gram_cols_]

        y_array = np.asarray(y_train)
        _, class_counts = np.unique(y_array, return_counts=True)
        use_stratify = len(class_counts) > 1 and np.min(class_counts) > 1

        can_split = len(X_train) >= 20 and len(np.unique(y_array)) > 1
        if can_split:
            X_spec_fit, X_spec_val, X_gram_fit, X_gram_val, y_fit, y_val = train_test_split(
                X_spec,
                X_gram,
                y_array,
                test_size=0.2,
                random_state=42,
                stratify=y_array if use_stratify else None
            )

            spec_model = self._create_spec_model()
            gram_model = self._create_gram_model()
            spec_model.fit(X_spec_fit, y_fit)
            gram_model.fit(X_gram_fit, y_fit)

            spec_probs = spec_model.predict_proba(X_spec_val)[:, 1]
            gram_probs = gram_model.predict_proba(X_gram_val)[:, 1]

            best_f1 = -1.0
            best_weight = self.spec_weight_
            best_threshold = self.threshold_

            for spec_weight in [0.65, 0.70, 0.75, 0.80, 0.85]:
                gram_weight = 1.0 - spec_weight
                fused_probs = spec_weight * spec_probs + gram_weight * gram_probs
                for threshold in [0.35, 0.40, 0.45, 0.50, 0.55]:
                    preds = (fused_probs >= threshold).astype(int)
                    score = f1_score(y_val, preds, zero_division=0)
                    if score > best_f1:
                        best_f1 = score
                        best_weight = spec_weight
                        best_threshold = threshold

            self.spec_weight_ = best_weight
            self.gram_weight_ = 1.0 - best_weight
            self.threshold_ = best_threshold
            self.best_validation_f1_ = best_f1

        self.spec_model = self._create_spec_model()
        self.gram_model = self._create_gram_model()
        self.spec_model.fit(X_spec, y_array)
        self.gram_model.fit(X_gram, y_array)
        self.model = self

    def _train_single_model(self, X_train, y_train):
        self.single_model = self._create_single_model()
        self.single_model.fit(X_train, y_train)
        self.model = self.single_model

    def predict(self, X_test):
        proba = self.predict_proba(X_test)[:, 1]
        threshold = self.threshold_ if self.feature_type == 'fusion' else 0.5
        return (proba >= threshold).astype(int)

    def predict_proba(self, X_test):
        if self.feature_type == 'fusion':
            spec_probs = self.spec_model.predict_proba(X_test[self.spec_cols_])[:, 1]
            gram_probs = self.gram_model.predict_proba(X_test[self.gram_cols_])[:, 1]
            fused_probs = self.spec_weight_ * spec_probs + self.gram_weight_ * gram_probs
            fused_probs = np.clip(fused_probs, 0.0, 1.0)
            return np.column_stack([1.0 - fused_probs, fused_probs])

        return self.single_model.predict_proba(X_test)

    def predict_proba_ai(self, X_test):
        return self.predict_proba(X_test)[:, 1]

    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_feature_importance(self, save_path=None):
        if self.feature_names_ is None:
            print("[Warning] No feature names available.")
            return

        try:
            if self.feature_type == 'fusion':
                spec_importances = self.spec_model.named_steps['model'].feature_importances_
                gram_importance = np.abs(self.gram_model.named_steps['model'].coef_).ravel()

                importances_map = {}
                for name, value in zip(self.spec_cols_, spec_importances):
                    importances_map[name] = float(value)
                for name, value in zip(self.gram_cols_, gram_importance):
                    importances_map[name] = float(value)

                names = [name for name in self.feature_names_ if name in importances_map]
                importances = np.array([importances_map[name] for name in names], dtype=float)

                total = importances.sum()
                if total > 0:
                    importances = importances / total
            else:
                importances = self.single_model.named_steps['model'].feature_importances_
                names = self.feature_names_

            idx = np.argsort(importances)
            plt.figure(figsize=(8, max(4, len(names) * 0.3)))
            plt.barh(range(len(idx)), importances[idx], align='center')
            plt.yticks(range(len(idx)), [names[i] for i in idx])
            plt.xlabel('Importance')
            plt.title('Feature Importances')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        except Exception as e:
            print(f"[Warning] Failed to plot feature importance: {e}")

    def save_model(self, filepath):
        state = {
            'single_model': self.single_model,
            'spec_model': self.spec_model,
            'gram_model': self.gram_model,
            'feature_type': self.feature_type,
            'feature_names_': self.feature_names_,
            'spec_cols_': self.spec_cols_,
            'gram_cols_': self.gram_cols_,
            'grammar_mode': self.grammar_mode,
            'spec_weight_': self.spec_weight_,
            'gram_weight_': self.gram_weight_,
            'threshold_': self.threshold_,
            'best_validation_f1_': self.best_validation_f1_
        }
        joblib.dump(state, filepath)

    def load_model(self, filepath):
        state = joblib.load(filepath)
        self.single_model = state.get('single_model')
        self.spec_model = state.get('spec_model')
        self.gram_model = state.get('gram_model')
        self.feature_type = state.get('feature_type')
        self.feature_names_ = state.get('feature_names_')
        self.spec_cols_ = state.get('spec_cols_', [])
        self.gram_cols_ = state.get('gram_cols_', [])
        self.grammar_mode = state.get('grammar_mode', self.grammar_mode)
        self.spec_weight_ = state.get('spec_weight_', 0.75)
        self.gram_weight_ = state.get('gram_weight_', 0.25)
        self.threshold_ = state.get('threshold_', 0.5)
        self.best_validation_f1_ = state.get('best_validation_f1_')
        self.model = self if self.feature_type == 'fusion' else self.single_model


def create_classifier(grammar_mode='statistical'):
    return AITextClassifier(grammar_mode=grammar_mode)