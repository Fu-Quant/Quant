import pandas as pd
import numpy as np

from Quant.evaluation.plot import Plotter
from datetime import datetime

class evaluator():
    def eva(self, result_dict, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        datetime_list = self.get_datetime(result_dict)
        
        yield_list = self.get_yield_list(result_dict,'yield')
        yield_inc_slippage = self.get_yield_list(result_dict,'yield_including_slippage')
        yield_inc_commission = self.get_yield_list(result_dict,'yield_including_commission')
        yield_inc_slippage_commission = self.get_yield_list(result_dict,'yield_including_commission_and_slippage')

        cumulative_return_list = self.get_cumulative_return_list(yield_list)
        cumulative_return_list_inc_slippage = self.get_cumulative_return_list(yield_inc_slippage)
        cumulative_return_list_inc_commission = self.get_cumulative_return_list(yield_inc_commission)
        cumulative_return_list_inc_slippage_commission = self.get_cumulative_return_list(yield_inc_slippage_commission)

        win_rate = self.get_win_rate(yield_list)
        win_rate_inc_commission_slippage = self.get_win_rate(yield_inc_slippage_commission)

        sharpe_ratio = self.get_sharpe_ratio(yield_list)
        sharpe_ratio_inc_commission_slippage = self.get_sharpe_ratio(yield_inc_slippage_commission)

        sortino_ratio = self.get_sortino_ratio(yield_list)
        sortino_ratio_inc_commission_slippage = self.get_sortino_ratio(yield_inc_slippage_commission)

        max_drawdown = self.get_max_drawdown(yield_list)
        max_drawdown_inc_commission_slippage = self.get_max_drawdown(yield_inc_slippage_commission)

        profit_loss_ratio = self.get_profit_loss_ratio(yield_list)
        profit_loss_ratio_inc_commission_slippage = self.get_profit_loss_ratio(yield_inc_slippage_commission)

        calmar_ratio = self.get_calmar_ratio(yield_list,max_drawdown)
        calmar_ratio_inc_commission_slippage = self.get_calmar_ratio(yield_inc_slippage_commission,max_drawdown_inc_commission_slippage)

        ret_eva_dict = {
            'cumulative_return_ratio': cumulative_return_list[-1],
            'cumulative_return_ratio_inc_slippage': cumulative_return_list_inc_slippage[-1],
            'cumulative_return_ratio_inc_commission': cumulative_return_list_inc_commission[-1],
            'cumulative_return_ratio_inc_slippage_commission': cumulative_return_list_inc_slippage_commission[-1],
            'win_rate': win_rate,
            'win_rate_inc_commission_slippage': win_rate_inc_commission_slippage,
            'profit_loss_ratio': profit_loss_ratio,
            'profit_loss_ratio_inc_commission_slippage': profit_loss_ratio_inc_commission_slippage,
            'sharpe_ratio': sharpe_ratio,
            'sharpe_ratio_inc_commission_slippage': sharpe_ratio_inc_commission_slippage,
        }

        risk_eva_dict = {
            'sortino_ratio': sortino_ratio,
            'sortino_ratio_inc_commission_slippage': sortino_ratio_inc_commission_slippage,
            'max_drawdown': max_drawdown,
            'max_drawdown_inc_commission_slippage': max_drawdown_inc_commission_slippage,
            'calmar_ratio': calmar_ratio,
            'calmar_ratio_inc_commission_slippage': calmar_ratio_inc_commission_slippage
        }

        ret_eva_dataframe = pd.Series(ret_eva_dict)
        print(f'return relevant ecaluation: \n{ret_eva_dataframe}\n')

        risk_eva_dataframe = pd.Series(risk_eva_dict)
        print(f'risk relevant ecaluation: \n{risk_eva_dataframe}\n')

        Plotter.plot_cumulative_product(cumulative_return_list, cumulative_return_list_inc_slippage,cumulative_return_list_inc_commission, cumulative_return_list_inc_slippage_commission, datetime_list)
    
    def get_yield_list(self, result_dict, yield_type):
        yield_list = []
        for i in range(len(result_dict)):
            yield_list.append(result_dict[i][yield_type])

        return yield_list
    
    def get_cumulative_return_list(self, yield_list):
        return np.cumsum(yield_list)
    
    def get_win_rate(self, yield_list):
        """计算胜率"""
        positive_yields = sum(1 for y in yield_list if y > 0)
        total_yields = len(yield_list)
        return positive_yields / total_yields if total_yields > 0 else 0
    
    def get_sharpe_ratio(self, yield_list):
        return np.mean(yield_list) / np.std(yield_list)
    
    def get_sortino_ratio(self, yield_list):
        """计算Sortino比率"""
        downside_returns = [y for y in yield_list if y < 0]
        return np.mean(yield_list) / np.std(downside_returns)
    
    def get_max_drawdown(self, yield_list):
        """计算最大回撤"""
        yield_list = np.array(yield_list)
        cum_rets = np.cumprod(1 + yield_list)
        peak = np.maximum.accumulate(cum_rets)
        drawdown = (peak - cum_rets) / peak

        return np.max(drawdown)
    
    def get_profit_loss_ratio(self, yield_list):
        """计算盈亏比"""
        profits = [y for y in yield_list if y > 0]
        losses = [-y for y in yield_list if y < 0]
        if not profits or not losses:
            return 0
        avg_profit = np.mean(profits)
        avg_loss = np.mean(losses)
        return avg_profit / avg_loss
    
    def get_annualized_return(self, yield_list, trading_days_per_year=240):
        """计算年化收益率"""
        yield_list = np.array(yield_list)
        cumulative_return = np.prod(1 + yield_list) - 1
        days = self.get_days_between_dates(self.start_date, self.end_date)
        num_years = days / trading_days_per_year
        annualized_return = (1 + cumulative_return) ** (1 / num_years) - 1
        return annualized_return
    
    def get_calmar_ratio(self, yield_list, max_drawdown, trading_days_per_year=240):
        """计算卡玛比率"""
        annualized_return = self.get_annualized_return(yield_list, trading_days_per_year)
        if max_drawdown == 0:
            return float('inf')  # 避免除以零的情况
        calmar_ratio = annualized_return / max_drawdown
        return calmar_ratio
    
    def get_days_between_dates(self, start_date, end_date):
        """计算两个日期之间的天数差"""
        start = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
        delta = end - start
        return delta.days
    
    def get_datetime(self, result_dict):
        datetime_list = []
        for i in range(len(result_dict)):
            datetime_list.append(result_dict[i]['datetime'])

        return datetime_list