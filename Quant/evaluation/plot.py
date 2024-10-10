import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Plotter:

    @staticmethod
    def plot_cumulative_product(cumulative_list, cumulative_return_list_inc_slippage,
                                   cumulative_return_list_inc_commission, cumulative_return_list_inc_slippage_commission,datetime_list):
        """绘制累积乘积的图表"""
        fig, ax = plt.subplots()

        # 将日期转换为数值
        num_dates = mdates.date2num(datetime_list)

        # 绘制曲线
        ax.plot_date(num_dates, cumulative_list, '-', label='Cumulative Product')
        ax.plot_date(num_dates, cumulative_return_list_inc_slippage, '-', label='Cumulative Product with Slippage')
        ax.plot_date(num_dates, cumulative_return_list_inc_commission, '-', label='Cumulative Product with Commission')
        ax.plot_date(num_dates, cumulative_return_list_inc_slippage_commission, '-', label='Cumulative Product with Slippage and Commission')

        # 设置日期格式器和定位器
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_minor_locator(mdates.DayLocator())

        plt.xlabel('Date')
        plt.ylabel('Cumulative Yield')
        plt.title('Cumulative Yield of Bitcoin Strategy')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_cumulative_product_3d(cumulative_return_list, cumulative_return_list_inc_slippage,
                                   cumulative_return_list_inc_commission, cumulative_return_list_inc_slippage_commission):
        """绘制四个累积收益的三维图表"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制四条线
        ax.plot(range(len(cumulative_return_list)), cumulative_return_list, zs=0, zdir='y', label='No Slippage/Commission')
        ax.plot(range(len(cumulative_return_list_inc_slippage)), cumulative_return_list_inc_slippage, zs=1, zdir='y', label='With Slippage')
        ax.plot(range(len(cumulative_return_list_inc_commission)), cumulative_return_list_inc_commission, zs=2, zdir='y', label='With Commission')
        ax.plot(range(len(cumulative_return_list_inc_slippage_commission)), cumulative_return_list_inc_slippage_commission, zs=3, zdir='y', label='With Slippage and Commission')

        ax.set_xlabel('Index')
        ax.set_ylabel('Scenario')
        ax.set_zlabel('Cumulative Return')
        ax.set_title('Cumulative Returns with Different Scenarios')
        ax.legend()

        plt.show()