import pandas as pd
import numpy as np
import warnings
import time
import copy

from Quant.factor_mining.formula_tree import formula_tree
from Quant.data.mysql_processor import mysql_processor
from Quant.utils.matrix_utils import matrix
from Quant.factor_mining.formula import run_function_for_signal_matrix, run_function_for_double_matrix
from Quant.factor_mining.preprocessor import standardize_dataframe, factor_preprocessor
from Quant.data.data_processor import fill_datanan
from Quant.factor_module.yield_calculator import yield_calculator
from Quant.factor_module.factor_test import factor_test, plot

from scipy.stats import spearmanr

# 忽略所有警告
warnings.filterwarnings("ignore")

class genetics_algo():
    def __init__(self) -> None:
        self.type_dict = {
            'add': 1,
            'subcontract': 1,
            'multiply': 1,
            'divide': 1,
            'max': 2,
            'min': 2,
            'sum': 2,
            'mean': 2,
            'std': 2,
            'delta_sub': 3,
            'delta_add': 3,
            'delta_multiply': 3,
            'delta_divide' : 3,
            'decay_linear': 3,
            'square': 4,
            'log': 4,
            'ln': 4,
            'abs': 4,
            'negative': 4,
            'sign': 4,
            'sqrt': 4,
            'cube': 4,
            'cbrt': 4,
            'inv': 4
        }
        self.type_list = [[],['add', 'subcontract', 'multiply', 'divide'],
                          ['max', 'min', 'sum', 'mean', 'std'], 
                          ['delta_sub','delta_add','delta_multiply','delta_divide','decay_linear'],
                          ['square', 'log', 'ln', 'abs', 'negative', 'sign', 'sqrt', 'cube', 'cbrt', 'inv']]
        self.formula_trees = []
        self.flat_list = [item for sublist in self.type_list for item in sublist]

    def create_formula_trees(self, num_trees):

        for num in range(num_trees):
            formula_tree = self.formula_genetor.build_formula_tree()
            self.formula_trees.append(formula_tree)
    
    def concat_array(self,formula_trees):
        indices_to_delete = []
        all_matrix = np.array([])

        for i in range(len(self.formula_trees)):
            factor_matrix_1 = self.formula_genetor.calculate_formula_tree(self.formula_trees[i])
            factor_matrix_1 = factor_preprocessor.signal_median_mad_processor(factor_matrix_1)

            if np.isnan(factor_matrix_1).any():
                indices_to_delete.append(i)
                continue
            if np.isinf(factor_matrix_1).any():
                indices_to_delete.append(i)
                continue

            if len(all_matrix) == 0:
                all_matrix = factor_matrix_1
            else:
                all_matrix = np.concatenate((all_matrix, factor_matrix_1), axis=1)
        
        indices_to_delete = list(set(indices_to_delete))
        new_formula_trees = [formula_trees[i] for i in range(len(formula_trees)) if i not in indices_to_delete]

        return all_matrix, new_formula_trees
    

    def selection(self, rank_threshold,corr_threshold):
        indices_to_delete = []
        data_matrix, self.formula_trees = self.concat_array(self.formula_trees)

        for i in range(len(self.formula_trees) - 1):
            factor_matrix_1 = data_matrix[:,i].reshape(len(data_matrix),1)
            factor_eva_1 = self.factor_test_module.signal_linear_factor_test(factor_matrix_1, self.yield_matrix, self.windows)
            
            if abs(factor_eva_1.loc[0,'RankIC']) < rank_threshold or np.isnan(factor_eva_1.loc[0,'RankIC']):
                indices_to_delete.append(i)
                continue

            for j in range(i + 1,len(self.formula_trees)):
                factor_matrix_2 = data_matrix[:,j].reshape(len(data_matrix),1)

                if np.corrcoef(factor_matrix_1.flatten(), factor_matrix_2.flatten())[0,1] > corr_threshold:
                    factor_matrix_2 = factor_preprocessor.signal_median_mad_processor(factor_matrix_2)
                    factor_eva_2 = self.factor_test_module.signal_linear_factor_test(factor_matrix_2, self.yield_matrix, self.windows)

                    if np.isnan(factor_eva_2.loc[0,'RankIC']):
                        indices_to_delete.append(j)
                        continue
                    
                    if factor_eva_1.loc[0,'RankIC'] <= factor_eva_2.loc[0,'RankIC']:
                        indices_to_delete.append(i)
                        break
                    elif factor_eva_1.loc[0,'RankIC'] > factor_eva_2.loc[0,'RankIC']:
                        indices_to_delete.append(j)

                    
        indices_to_delete = list(set(indices_to_delete))
        new_formula_list = [self.formula_trees[i] for i in range(len(self.formula_trees)) if i not in indices_to_delete]
        self.formula_trees = new_formula_list

        if len(self.formula_trees) != 0:
            self.performance_evaluation()

    def performance_evaluation(self):
        data_matrix, self.formula_trees = self.concat_array(self.formula_trees)
        eva = self.factor_test_module.multiple_linear_factor_eva(data_matrix, self.yield_matrix)
        print(eva)

        #self.plot_module.plot_heatmap(data_matrix)

        
    def get_cross_list(self, formula_list, target_index):
        i = 0
        cross_list = [target_index]
        while True:
            if cross_list[-1] * 2 + 1 > len(formula_list):
                return cross_list
                break
            left_cross_list = [cross_list[x] * 2 for x in range(-2**i, 0)]
            right_cross_list = [cross_list[x] * 2 + 1 for x in range(-2**i, 0)]
            cross_list += left_cross_list + right_cross_list
            cross_list = sorted(cross_list)
            i += 1
        
    def get_origin_list(self, cross_list, target_index):
        i = 0
        origin_cross_list = [target_index]

        while True:
            if len(origin_cross_list) == len(cross_list):
                return origin_cross_list
                break
            left_cross_list = [origin_cross_list[x] * 2 for x in range(-2**i, 0)]
            right_cross_list = [origin_cross_list[x] * 2 + 1 for x in range(-2**i, 0)]
            origin_cross_list += left_cross_list + right_cross_list
            origin_cross_list = sorted(origin_cross_list)
            i += 1

    def execute_crossover(self, formula_list_1, formula_list_2, origin_list, cross_list):

        length = len(formula_list_1) - 1

        while length < origin_list[-1]:

            length = (length + 1) * 2 - 1
        
        new_length = length - (len(formula_list_1) - 1)
        new_list = [0] * new_length
        new_formula_list = copy.deepcopy(formula_list_1)
        new_formula_list += new_list

        for i in range(len(origin_list)):
            if i <= len(cross_list) - 1:
                new_formula_list[origin_list[i]] = formula_list_2[cross_list[i]]
                #print( formula_list_1[origin_list[i]])
                #print(formula_list_2[cross_list[i]])
            elif i > len(cross_list) - 1:
                new_formula_list[origin_list[i]] = 0
        
        return new_formula_list
    
    def crossover(self):

        new_formula_trees = []

        for i in range(len(self.formula_trees) - 1):
            formula_list_1 = self.formula_trees[i]

            for j in range(i + 1,len(self.formula_trees)):
                formula_list_2 = self.formula_trees[j]

                for index_1 in range(len(formula_list_1)):
                    
                    for index_2 in range(len(formula_list_2)):

                        if formula_list_1[index_1] in self.type_dict and formula_list_2[index_2] in self.type_dict:

                            if self.type_dict[formula_list_1[index_1]] == self.type_dict[formula_list_2[index_2]]:

                                crossover_pro = np.random.rand(1)

                                if crossover_pro > 0.5 and formula_list_1[i] in self.flat_list and formula_list_2[j] in self.flat_list:

                                    cross_list_1 = self.get_cross_list(formula_list_1, index_1)
                                    cross_list_2 = self.get_cross_list(formula_list_2, index_2)

                                    origin_list_1 = self.get_origin_list(cross_list_2, index_1)
                                    origin_list_2 = self.get_origin_list(cross_list_1, index_2)

                                    if len(cross_list_1) > len(origin_list_1):
                                        origin_list_1 = cross_list_1
                                    
                                    if len(cross_list_2) > len(origin_list_2):
                                        origin_list_2 = cross_list_2
                                    
                                    new_formula_list_1 = self.execute_crossover(formula_list_1, formula_list_2, origin_list_1, cross_list_2)
                                    new_formula_list_2 = self.execute_crossover(formula_list_2, formula_list_1, origin_list_2, cross_list_1)
                                    new_formula_trees.append(new_formula_list_1)
                                    new_formula_trees.append(new_formula_list_2)

        
        self.formula_trees += new_formula_trees
                    
    def mutation(self):
        for i in range(len(self.formula_trees)):
            new_formula_list = self.formula_trees[i]
            mutation_exe = False

            for j in range(len(new_formula_list)):
                mutation_pro = np.random.rand(1)
                if mutation_pro > 0.5 and new_formula_list[j] in self.flat_list:
                    type_index = self.type_dict[new_formula_list[j]]
                    new_type = np.random.choice(self.type_list[type_index])
                    new_formula_list[j] = new_type
                    mutation_exe = True
            
            if mutation_exe:
                self.formula_trees.append(new_formula_list)

    def get_bitcoin_data(self, start_date:str, end_date:str):

        data_connector = mysql_processor()
        self.max_window = 20
        self.windows = [10]
        self.bitcoin_data = data_connector.get_data_for_factor('BTC',start_date,end_date,int(self.max_window * 2))
        self.bitcoin_data.to_csv('/Users/fu/Desktop/bitcoin_test.csv')
        self.bitcoin_data = fill_datanan(self.bitcoin_data)
        self.yield_data = data_connector.get_data_for_return('BTC',start_date,end_date,10)
        self.yield_data = fill_datanan(self.yield_data)
    
    def create_module(self, start_date, end_date):

        self.yield_matrix = yield_calculator.get_yield(self.yield_data, end_date, self.windows)
        self.formula_genetor = formula_tree(start_date,end_date,self.bitcoin_data, self.max_window)
        self.factor_test_module = factor_test()
        self.plot_module = plot()
    
    def get_factor_matrix(self,start_date, end_date, formula_trees):
        self.get_bitcoin_data(start_date, end_date)
        self.create_module(start_date, end_date)

        for i in range(len(formula_trees)):
            factor_matrix_1 = self.formula_genetor.calculate_formula_tree(formula_trees[i])
            factor_matrix_1 = factor_preprocessor.signal_median_mad_processor(factor_matrix_1)

            if i == 0:
                all_matrix = factor_matrix_1
            else:
                all_matrix = np.concatenate((all_matrix, factor_matrix_1), axis=1)

        return all_matrix
        
    def run(self, start_date, end_date, num_trees):

        self.get_bitcoin_data(start_date, end_date)
        self.create_module(start_date, end_date)
        self.create_formula_trees(num_trees)
        for i in range(3):
            self.selection(0.02,0.5)
            self.crossover()
            self.mutation()

        self.selection(0.02,0.5)

        return self.formula_trees
        


if __name__ == '__main__':
    genetics_modeule = genetics_algo()
    genetics_modeule.run()