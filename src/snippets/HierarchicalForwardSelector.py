import pandas as pd
import statsmodels.api as sm
import numpy as np
from collections import Counter
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import math

class HierarchicalForwardSelector:
    """
    Esegue una selezione forward delle feature per modelli di regressione lineare,
    rispettando il principio di gerarchia e permettendo un'esecuzione step-by-step.
    """
    def __init__(self, 
                 max_degree=3, 
                 max_features=20, 
                 p_value_threshold=0.05, 
                 min_adj_r2_improvement=0.002, 
                 verbose=True):
        self.max_degree = max_degree
        self.max_features = max_features
        self.p_value_threshold = p_value_threshold
        self.min_adj_r2_improvement = min_adj_r2_improvement
        self.verbose = verbose

        # Attributi per salvare lo stato
        self.final_model_ = None
        self.selected_features_ = []
        self._is_fitted = False
        self._selection_complete = False

        # Attributi per la selezione step-by-step
        self.X_ = None
        self.y_ = None
        self.step_count_ = 0
        self.selected_features_set_ = set()
        self.candidate_features_set_ = set()
        self.best_adj_r2_ = -np.inf

    # ... [Metodi privati _parse_feature, _format_feature, etc. rimangono invariati] ...
    def _parse_feature(self, feature_str):
        components = feature_str.split('*')
        degree_dict = Counter()
        for comp in components:
            if '^' in comp:
                var, exp = comp.split('^')
                degree_dict[var.strip()] += int(exp)
            else:
                degree_dict[comp.strip()] += 1
        return degree_dict

    def _format_feature(self, feature_dict):
        sorted_items = sorted(feature_dict.items())
        parts = []
        for var, exp in sorted_items:
            parts.append(var if exp == 1 else f"{var}^{exp}")
        return '*'.join(parts)

    def _get_total_degree(self, feature_dict):
        return sum(feature_dict.values())

    def _get_divisors(self, feature_dict):
        var_power_ranges = []
        for var, max_exp in feature_dict.items():
            var_power_ranges.append([(var, exp) for exp in range(max_exp + 1)])

        divisor_tuples = product(*var_power_ranges)

        divisors_set = set()
        for dt in divisor_tuples:
            temp_dict = {var: exp for var, exp in dt if exp > 0}
            if temp_dict:
                divisors_set.add(self._format_feature(temp_dict))
        return divisors_set

    def _build_design_matrix(self, X_original, feature_names):
        df = pd.DataFrame(index=X_original.index)
        for feature_str in feature_names:
            feature_dict = self._parse_feature(feature_str)
            term_column = pd.Series(1, index=X_original.index, dtype=float)
            for var, exp in feature_dict.items():
                term_column *= X_original[var]**exp
            df[feature_str] = term_column
        return sm.add_constant(df, prepend=True)


    def _initialize_fit(self, X, y):
        """Inizializza o resetta lo stato del selettore per un nuovo fit."""
        self.X_ = X
        self.y_ = y
        self.step_count_ = 0
        self.selected_features_set_ = set()
        initial_features = [self._parse_feature(col) for col in X.columns]
        self.candidate_features_set_ = {self._format_feature(f) for f in initial_features}

        base_model = sm.OLS(y, sm.add_constant(pd.DataFrame(index=X.index))).fit()
        self.best_adj_r2_ = base_model.rsquared_adj
        self._selection_complete = False
        self._is_fitted = False

        if self.verbose:
            print("🚀 Avvio della selezione forward con PRINCIPIO DI GERARCHIA:")
            print(f" - Grado massimo: {self.max_degree}, Max features: {self.max_features}")
            print(f" - Soglia p-value: {self.p_value_threshold}, Min miglioramento Adj. R²: {self.min_adj_r2_improvement}")
            print(f"Modello iniziale (solo intercetta): Adjusted R² = {self.best_adj_r2_:.4f}\n")

    def step(self):
        """Esegue un singolo passo della selezione forward."""
        if self._selection_complete:
            if self.verbose: print("Selezione già completata.")
            return False

        self.step_count_ += 1
        if self.verbose: print(f"--- PASSO {self.step_count_} ---")

        if len(self.selected_features_set_) >= self.max_features:
            if self.verbose: print(f"🛑 Raggiunto il limite di {len(self.selected_features_set_)} feature. Arresto.")
            self._selection_complete = True
            return False

        candidate_features_list = sorted(list(self.candidate_features_set_))
        if self.verbose: print(f"🔍 Valutazione di {len(candidate_features_list)} candidati...")

        best_candidate_str = None
        current_best_adj_r2 = self.best_adj_r2_

        for candidate_str in candidate_features_list:
            # Logica di valutazione del candidato...
            candidate_dict = self._parse_feature(candidate_str)
            required_divisors = self._get_divisors(candidate_dict)
            features_to_test_set = self.selected_features_set_.union(required_divisors)

            X_temp = self._build_design_matrix(self.X_, list(features_to_test_set))
            try:
                model = sm.OLS(self.y_, X_temp).fit()
                if candidate_str in model.pvalues:
                    adj_r2 = model.rsquared_adj
                    p_value = model.pvalues[candidate_str]
                    if adj_r2 > current_best_adj_r2 and p_value < self.p_value_threshold:
                        current_best_adj_r2 = adj_r2
                        best_candidate_str = candidate_str
            except np.linalg.LinAlgError:
                continue

        if best_candidate_str is None:
            if self.verbose: print("🛑 Nessun nuovo candidato valido trovato. Arresto.")
            self._selection_complete = True
            return False

        improvement = current_best_adj_r2 - self.best_adj_r2_
        if improvement < self.min_adj_r2_improvement:
            if self.verbose: print(f"🛑 Miglioramento insufficiente (+{improvement:.4f}). Arresto.")
            self._selection_complete = True
            return False

        # Aggiornamento dello stato con il miglior candidato
        self.best_adj_r2_ = current_best_adj_r2
        best_candidate_dict = self._parse_feature(best_candidate_str)
        divisors_to_add = self._get_divisors(best_candidate_dict)

        if self.verbose: print(f"✅ Miglior candidato: '{best_candidate_str}'. Aggiunta di {len(divisors_to_add)} termini.")

        self.selected_features_set_.update(divisors_to_add)
        self.candidate_features_set_.difference_update(divisors_to_add)
        if self.verbose: print(f"   Nuovo Adjusted R²: {self.best_adj_r2_:.4f} (Miglioramento: +{improvement:.4f})")

        # Generazione nuovi candidati
        new_candidates_added = 0
        for initial_feat_dict in [self._parse_feature(col) for col in self.X_.columns]:
            new_candidate_dict = best_candidate_dict + initial_feat_dict
            if self._get_total_degree(new_candidate_dict) <= self.max_degree:
                new_candidate_str = self._format_feature(new_candidate_dict)
                if new_candidate_str not in self.selected_features_set_ and new_candidate_str not in self.candidate_features_set_:
                    self.candidate_features_set_.add(new_candidate_str)
                    new_candidates_added += 1

        if self.verbose: print(f"   Generati {new_candidates_added} nuovi candidati.\n")
        return True

    def fit(self, X, y):
        """Esegue il processo completo di selezione delle feature."""
        self._initialize_fit(X, y)
        while self.step():
            pass
        self._finalize_model()
        return self

    def _finalize_model(self):
        """Costruisce e salva il modello finale dopo la selezione."""
        if self.verbose: print("\n--- SELEZIONE COMPLETATA ---")

        self.selected_features_ = sorted(list(self.selected_features_set_))

        if not self.selected_features_:
            if self.verbose: print("Nessuna feature è stata selezionata.")
            return

        if self.verbose: print(f"Variabili finali selezionate: {self.selected_features_}")

        X_final = self._build_design_matrix(self.X_, self.selected_features_)
        self.final_model_ = sm.OLS(self.y_, X_final).fit()
        self._is_fitted = True

    def summary(self):
        """Stampa il sommario del modello finale OLS."""
        if self.final_model_:
            print("\n--- SOMMARIO DEL MODELLO FINALE ---")
            print(self.final_model_.summary())
        elif self._is_fitted:
            print("Nessuna feature è stata selezionata. Impossibile generare un sommario.")
        else:
            print("Il modello non è stato ancora addestrato. Chiamare il metodo `fit` prima.")

    def plot_residuals(self):
        """
        Genera un pannello di grafici dei residui rispetto a ogni variabile del modello.
        """
        if not self._is_fitted or not self.final_model_:
            print("Il modello deve essere addestrato con `fit` prima di poter plottare i residui.")
            return

        residuals = self.final_model_.resid
        X_final = self._build_design_matrix(self.X_, self.selected_features_).drop('const', axis=1)

        n_features = len(self.selected_features_)
        ncols = min(n_features, 3)
        nrows = math.ceil(n_features / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
        axes = np.ravel(axes) # Rende l'array di assi 1D per un facile accesso

        for i, feature_name in enumerate(self.selected_features_):
            ax = axes[i]
            sns.residplot(x=X_final[feature_name], y=residuals, lowess=True, 
                          ax=ax, scatter_kws={'alpha': 0.6})
            ax.set_title(f'Residui vs {feature_name}')
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Residui')

        # Nasconde gli assi non utilizzati
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()
