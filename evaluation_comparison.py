import seaborn as sns  
import matplotlib.pyplot as plt  
import pandas as pd  
import Models  
import inspect  
import numpy as np 
import os, re  

class Eval_comparison():
    def __init__(self):
        pass  
        
    def comp_btw_rnns(self):
        # get all class objects from Models module
        classes = inspect.getmembers(Models, inspect.isclass)

        # regex pattern to match 'rnn', 'gru', or 'lstm' in class names
        pattern = re.compile(r"rnn|gru|lstm", flags=re.IGNORECASE)

        # filter class names that come from our Models module and match pattern
        self.module_names = [
            (re.search(pattern, name)).group() 
            for name, cls in classes 
            if cls.__module__ == Models.__name__
        ]

        self.ev_scores = []  # will store average MAE for each model
        
        for i in self.module_names:
         
            match = re.search(pattern, i)
       
            result = pd.read_csv(f'results/{match.group()}.csv').values.reshape(-1)
       
            avg_result = round(np.mean(result), 3)
            self.ev_scores.append(avg_result)  # add to score list
            
      
        scores = {'Model': self.module_names, 'MAE': self.ev_scores}
        df = pd.DataFrame(scores)

        plt.figure()  
        
        sns.barplot(x='Model', y='MAE', data=df, width=0.4)
        plt.title('MAE Comparison of Models')

    
        os.makedirs('comparison-graphs', exist_ok=True)
    
        plt.savefig('comparison-graphs/Prediction MAE Comparison btw rnn Models.jpg')
        
    def comp_btw_models(self):
        # also compare with baseline ANN
        self.module_names = self.module_names + ['baseline-ANN']
        self.ev_scores = self.ev_scores + [1.5]  # baseline MAE assumed

        scores = {'Model': self.module_names, 'MAE': self.ev_scores}
        df = pd.DataFrame(scores)

        # plot bar chart with all models
        sns.barplot(x='Model', y='MAE', data=df, width=0.4)
        plt.title('MAE Comparison of Models')

        os.makedirs('comparison-graphs', exist_ok=True)  

        plt.savefig('comparison-graphs/Prediction MAE Comparison btw Models.jpg')