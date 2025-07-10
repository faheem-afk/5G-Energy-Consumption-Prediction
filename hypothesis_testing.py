from settings import model_names
import pandas as pd
from scipy import stats
import os
from logger import logging
import json

class HypothesisTest():

    def __init__(self):
        os.makedirs('hypothesis_tests', exist_ok=True)
    
    def hyp_test(self):
        for i in model_names:
            
            logging.info(i)
            result = pd.read_csv(f'results/{i}.csv').values.reshape(-1)
            
            alpha = 0.05
            
            # Null hypothesis: mean(mae_scores) == 1.5
            # Perform two‐sided t‐test first
            t_stat, p_two_sided = stats.ttest_1samp(result, popmean=1.5)

            
            value_dic = {}
            
            # Convert to one‐sided p‐value for H1: mean < 1.5
            p_one_sided = p_two_sided / 2

            mean = result.mean()
            std_err = stats.sem(result)
            n = len(result)
            
            ci = stats.t.interval(1 - alpha, df=n - 1, loc=mean, scale=std_err)
            
            value_dic = {
            'significance_value': alpha,
            't-statistic': f"{t_stat:.4f}",
            'H0': 1.5,
            'one-sided_p-value': f"{p_one_sided:.2e}",
            'MAE_sample_mean': round(mean, 4),
            '95%_confidence_interval': [round(ci[0], 4), round(ci[1], 4)],
        }
            logging.info(f"significane_value:  {alpha}")
            logging.info(f"t-statistic:    {t_stat:.4f}")
            logging.info(f"H0:  {1.5}")
            logging.info(f"one-sided p-value: {p_one_sided:.2e}")
            logging.info(f"MAE_sample_mean: {round(mean, 4)}")
            logging.info(f"95%_confidence_interval: {[round(ci[0], 4), round(ci[1], 4)]}")
            
               
            if p_one_sided < alpha:
                logging.info(f"p = {p_one_sided:.2e} < {alpha:.2f}: reject H₀")
                value_dic[f'p = {p_one_sided:.2e} < {alpha:.2f}'] = 'reject H0'
                value_dic['model_significance'] = "Positive"
            else:
                logging.info(f"p = {p_one_sided:.2e} ≥ {alpha:.2f}: cannot reject H₀")
                value_dic[f'p = {p_one_sided:.2e} ≥ {alpha:.2f}'] = 'cannot reject H0'
                value_dic['model_significance'] = "Negative"
                
            with open(f'hypothesis_tests/{i}.json', 'w') as f:
                json.dump(value_dic, f, indent=2)
            
            print('\n')
            