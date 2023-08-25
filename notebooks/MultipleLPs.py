import copy
import random
from collections import defaultdict
from mbt_gym.stochastic_processes.midprice_models import GeometricBrownianMotionMidpriceModel

import PPO_class
from mbt_gym.agents import Agent
from mbt_gym.agents.BaselineAgents import PoolInterpGEtaMmAgent
import matplotlib.colors as mcolors

class ETAFunctionError(Exception):
    pass
class PriceMismatch(Exception):
    pass
class UnconsideredCase(Exception):
    pass
class ZValueMismatch(Exception):
    pass
class MarketImpactFunctionLogicalError(Exception):
    pass
class CaseError(Exception):
    pass
class LT_Action_Error(Exception):
    pass
class ImapctFnExcludeLPError (Exception):
    pass
class BidAskError(Exception):
    pass
class NumberOfTradesLargerThanSteps(Exception):
    pass
class TotalMarketImpactCaseError(Exception):
    pass

class WealthSlippage(Exception):
    pass
from nb_utils import rescale_plot, mtick, md, run_simulation,\
                     getSimulationData, get_pool_agent, get_arb_env,\
                     get_LT_LP_Binance_data, pd, plt, np, ASSET_PRICE_INDEX,\
                     datetime, get_binance_month_data, plot_impact_curves,\
                     runOneSimulation, getOneSimulationData, plot_one_sim_result

RL_SETTING = False
DOUBLE_CHECK = False
External_Market_Arbitrage_Flag = False

# Common Parameters Fixed
max_depth              = 0
terminal_time          = 30/60/24 #/24/ # 30 minutes
n_steps                = 1000
step_size              = terminal_time/n_steps
seed                   = 1
num_trajectories       = 1
verbose                = False


# Common Standard Parameters

# LP_1 strategy parameters
############################
LP_1 = {
'jump_size_L'            : 0.01,
'phi'                    : 1e-4,
'alpha'                  : 1e-4,
'fill_exponent'          : 2,
'initial_inventory_pool' : -4,
'initial_price'          : 10,
'unit_size'              : 1,
'arrival_rate'           : 100,
'impact_fn_set':2
}
LP_1['target_inventory' ] = 0#LP_1['initial_inventory_pool']
LP_1['min_inventory_pool']  = LP_1['initial_inventory_pool'] - 99.
LP_1['max_inventory_pool'] = LP_1['initial_inventory_pool'] + 99.

# LP_2 strategy parameters
LP_2_significant_inv_penelties = {
'jump_size_L'            : 0.01,
'phi'                    : 5e-3,
'alpha'                  : 5e-3,
'fill_exponent'          : 2,
'initial_inventory_pool' : 0,
'initial_price'          : 10,
'unit_size'              : 1,
'arrival_rate'           : 100,
'impact_fn_set':2
}
LP_2_significant_inv_penelties['target_inventory' ] = 0
LP_2_significant_inv_penelties['min_inventory_pool']  = LP_2_significant_inv_penelties['initial_inventory_pool'] - 502.
LP_2_significant_inv_penelties['max_inventory_pool'] = LP_2_significant_inv_penelties['initial_inventory_pool'] + 502.

# LP_2 strategy parameters
LP_2_no_inv_penelties = {
'jump_size_L'            : 0.01,
'phi'                    : 0,
'alpha'                  : 0,
'fill_exponent'          : 2,
'initial_inventory_pool' : 0,
'initial_price'          : 10,
'unit_size'              : 1,
'arrival_rate'           : 100,
'impact_fn_set':2
}
LP_2_no_inv_penelties['target_inventory' ] = 0#LP_2['initial_inventory_pool']
LP_2_no_inv_penelties['min_inventory_pool']  = LP_2_no_inv_penelties['initial_inventory_pool'] - 502.
LP_2_no_inv_penelties['max_inventory_pool'] = LP_2_no_inv_penelties['initial_inventory_pool'] + 502.

# LP_3 strategy parameters
LP_3_kappa_2 = {
'jump_size_L'            : 0.01,
'phi'                    : 1e-4,
'alpha'                  : 1e-4,
'fill_exponent'          : 2,
'initial_inventory_pool' : 0,
'initial_price'          : 10,
'unit_size'              : 1,
'arrival_rate'           : 100,
'impact_fn_set':2
}
LP_3_kappa_2['target_inventory' ] = 0#LP_2['initial_inventory_pool']
LP_3_kappa_2['min_inventory_pool']  = LP_3_kappa_2['initial_inventory_pool'] - 502.
LP_3_kappa_2['max_inventory_pool'] = LP_3_kappa_2['initial_inventory_pool'] + 502.

# LP_3 strategy parameters
LP_3_kappa_1_8 = {
'jump_size_L'            : 0.01,
'phi'                    : 1e-4,
'alpha'                  : 1e-4,
'fill_exponent'          : 1.8,
'initial_inventory_pool' : 0,
'initial_price'          : 10,
'unit_size'              : 1,
'arrival_rate'           : 100,
'impact_fn_set':2
}
LP_3_kappa_1_8['target_inventory' ] = 0#LP_2['initial_inventory_pool']
LP_3_kappa_1_8['min_inventory_pool']  = LP_3_kappa_1_8['initial_inventory_pool'] - 502.
LP_3_kappa_1_8['max_inventory_pool'] = LP_3_kappa_1_8['initial_inventory_pool'] + 502.

# LP_4 strategy parameters
LP_4 = {
'jump_size_L'            : 0.2,
'phi'                    : 0.1,
'alpha'                  : 1e-9,
'fill_exponent'          : 0.5,
'initial_inventory_pool' : 0,
'initial_price'          : 2600,
'unit_size'              : 1,
'arrival_rate'           : 100,
'impact_fn_set':0
}
LP_4['target_inventory' ] = -100#LP_2['initial_inventory_pool']
LP_4['min_inventory_pool']  = LP_4['initial_inventory_pool'] - 100.
LP_4['max_inventory_pool'] = LP_4['initial_inventory_pool'] + 100.

# LP_5 strategy parameters
LP_5 = {
'jump_size_L'            : 0.2,
'phi'                    : 0,#0.1,
'alpha'                  : 0,#1e-9,
'fill_exponent'          : 2,
'initial_inventory_pool' : 0,
'initial_price'          : 2600,
'unit_size'              : 1,
'arrival_rate'           : 100,
'impact_fn_set':0
}
LP_5['target_inventory' ] = 0#LP_2['initial_inventory_pool']
LP_5['min_inventory_pool']  = LP_5['initial_inventory_pool'] - 100.
LP_5['max_inventory_pool'] = LP_5['initial_inventory_pool'] + 100.

LT_1 = {'starting_inv': 0,
        'starting_cash': 0,
        'probs':(0.45,0.1,0.45),
        'arbitrageur':False
        }

LT_2 = {'starting_inv': 0,
        'starting_cash': 0,
        'probs':(0.25,0.25,0.5),
        'arbitrageur':False
        } #[LT_take_bid, LT_hold, LT_take_ask]

LT_3 = {'starting_inv': 0,
        'starting_cash': 0,
        'probs':(0,1,0),
        'arbitrageur':True
        }

LT_take_bid = -1  # -1 LT Sells (BID)
LT_hold = 0
LT_take_ask = 1  # 1 LT Buys (ASK)
LT_liquidate = 2

# Paper eta functions
def assign_impact_fn(eta_function_used):
    if eta_function_used == 0: #my fun
        def eta_func_ask(y, Delta, L):
            if y==Delta: return 0
            if y-Delta>0: return L * Delta  / (y - Delta)
            if y-Delta<0: return -L * Delta / (y - Delta)

        def eta_func_bid(y, Delta, L):
            if y+Delta==0: return 0
            if y+Delta>0: return L * Delta  / (y + Delta)
            if y+Delta<0: return -L * Delta / (y + Delta)
    elif eta_function_used == 1:#paper fn
        def eta_func_ask(y, Delta, L):
            if y-Delta==0: return -L
            if y-Delta>0: return L * Delta  / (y - Delta)
            if y-Delta<0: return -L * Delta / (y - Delta)

        def eta_func_bid(y, Delta, L):
            if y+Delta==0: return -L
            if y+Delta>0: return L * Delta  / (y + Delta)
            if y+Delta<0: return -L * Delta / (y + Delta)
    elif eta_function_used == 2:
        def eta_func_ask(y, Delta, L): #improved fn
            if y == Delta: return L
            if y - Delta > 0: return L * Delta / (y - Delta)
            if y - Delta < 0: return -L * Delta / (y - Delta)

        def eta_func_bid(y, Delta, L):
            if y + Delta == 0: return L
            if y + Delta > 0: return L * Delta / (y + Delta)
            if y + Delta < 0: return -L * Delta / (y + Delta)
    else:
        raise ETAFunctionError

    return eta_func_bid, eta_func_ask

# -------------------------Plotting Functions-------------------
# Plot 1
def plot_bid_ask_spreads_and_impact_for_LPs(router):
    fig = plt.figure(figsize=(13, len(router.LPs)*5))
    for (lp,i)  in zip(router.LPs,range(0,len(router.LPs)*2,2)):
        plt.subplot(len(router.LPs), 2, i+1)
        router.LPs[lp].plot_impact_fn(show_plot=False)
        plt.subplot(len(router.LPs), 2, i + 2)
        router.LPs[lp].plot_bid_and_ask_spread(show_plot=False)
    plt.show()

# Last Plot
def plot_different_trade_executions(sims): #Last plot
    import matplotlib.colors as mcolors
    sim_colours = [z for z in mcolors.BASE_COLORS][::-1]
    for sim, colour in zip(sims, sim_colours):
        trades_executed = sim['trade_log']

        plt.plot([z[1] for z in trades_executed], [z[0] for z in trades_executed], 'o'.format(colour),
                     label="Case {}".format(sims.index(sim)+1), linewidth=1.0, markerfacecolor='none')
    plt.title("Trade Prices for Different Simulations")
    plt.ylabel("Trade Price")
    plt.xlabel("Time step")
    plt.grid()
    plt.legend(loc="upper left")
    plt.show()

#Simulation Plot
def plot_simulation_results(router,sim_stat, case):
    fig = plt.figure(figsize=(12,11))
    st = fig.suptitle(
        "Market Simulation - {}".format(case),
        fontsize="xx-large")
    lp_colours = mcolors.BASE_COLORS
    lt_colours = mcolors.TABLEAU_COLORS

    plt.subplot(3, 2, 1)
    x_axis = sim_stat['time']

    x_axis_ticks = x_axis[::2]
    plt.xticks(x_axis_ticks)

    for lt, colour in zip(router.LTs,lt_colours):
        plt.plot(x_axis, sim_stat['LT'][lt]['inv'], colour, label="LT {}".format(lt), linewidth=2.0)
    for lp, colour in zip(router.LPs, lp_colours):
        plt.plot(x_axis, sim_stat['LP'][lp]['inv'], '--{}'.format(colour), label="LP {}".format(lp), linewidth=2.0)

    plt.ylabel("Inventory of LPs and LTs")
    plt.xlabel("Time step")
    plt.grid()
    plt.xticks(x_axis_ticks)
    plt.legend(loc="best")

    plt.subplot(3, 2, 2)
    for lt, colour in zip(router.LTs, lt_colours):
        plt.plot(x_axis, sim_stat['LT'][lt]['w'], colour, label="LT {}".format(lt), linewidth=2.0)
    for lp, colour in zip(router.LPs, lp_colours):
        plt.plot(x_axis, sim_stat['LP'][lp]['w'], '--{}'.format(colour), label="LP {}".format(lp), linewidth=2.0)
    plt.ylabel("Cash of LPs and LTs")
    plt.xlabel("Time step")
    plt.grid()
    plt.xticks(x_axis_ticks)
    plt.legend(loc="best")

    plt.subplot(3, 2, 3)
    for lp, colour in zip(router.LPs,lp_colours):
        plt.plot(x_axis, sim_stat['LP'][lp]['z'], '--{}'.format(colour), label="LP {}".format(lp), linewidth=1.0)
    plt.ylabel("LP's Marginal Rate")
    plt.xlabel("Time step")
    plt.grid()
    plt.xticks(x_axis_ticks)
    plt.legend(loc="best")

    plt.subplot(3, 2, 4)
    for lp, colour in zip(router.LPs,lp_colours):
        trades_exec_by_lp = [z for z in sim_stat['trade_log'] if z[2] == lp]
        plt.plot([z[1] for z in trades_exec_by_lp], [z[0] for z in trades_exec_by_lp], 'o'.format(colour), label="LP_{}".format(lp), linewidth=1.0)
    plt.ylabel("Trade Price")
    plt.xlabel("Time step")
    plt.grid()
    plt.xticks(list(set([z[1] for z in sim_stat['trade_log']][::2])))
    plt.legend(loc="best")

    plt.subplot(3, 2, 5)
    for lp, colour in zip(router.LPs, lp_colours):
        plt.plot(x_axis, sim_stat['LP'][lp]['bid'], colour, label="LP {} Bid".format(lp), linewidth=1.0)
    plt.ylabel("LP's posting bid")
    plt.xlabel("Time step")
    y_ticks, y_tick_labels = plt.yticks()
    plt.ylim(min(y_ticks) - 0.01, max(y_ticks))
    plt.grid()
    plt.xticks(x_axis_ticks)
    plt.legend(loc="best")

    plt.subplot(3, 2, 6)

    for lp, colour in zip(router.LPs, lp_colours):
        plt.plot(x_axis, sim_stat['LP'][lp]['ask'], colour, label="LP {} Ask".format(lp), linewidth=1.0)
    plt.ylabel("LP's posting ask")
    plt.xlabel("Time step")
    y_ticks, y_tick_labels = plt.yticks()
    plt.ylim(min(y_ticks) - 0.01, max(y_ticks))
    plt.grid()
    plt.xticks(x_axis_ticks)
    plt.legend(loc="best")

    plt.show()


def plot_multiple_simulation_results(simulation_results, title=''):
    fig = plt.figure(figsize=(11,10))

    lps = list(simulation_results['LP'].keys())
    lts = list(simulation_results['LT'].keys())
    lp_colours = [x for x in mcolors.TABLEAU_COLORS][:len(lps)]
    lt_colours = [x for x in mcolors.TABLEAU_COLORS][len(lps):]

    x_axis_time_steps= simulation_results['time']

    plt.subplot(3, 2, 1)
    for lp, colour in zip(lps, lp_colours):
        data = simulation_results['LP'][lp]['inv']
        mean = data['mean']
        std_dev = data['std']

        plt.plot(x_axis_time_steps, mean, linestyle='--',lw=2, color=colour, label='LP {}'.format(lp))
        plt.fill_between(x_axis_time_steps, mean - std_dev, mean + std_dev, color=colour, alpha=0.1, label='LP {}'.format(lp))

    plt.ylabel("LP's Inventory")
    plt.xlabel("Time step")
    # plt.legend(loc="lower left")

    plt.subplot(3, 2, 2)
    for lt, colour in zip(lts, lt_colours):
        data = simulation_results['LT'][lt]['inv']
        mean = data['mean']
        std_dev = data['std']

        plt.plot(x_axis_time_steps, mean, linestyle='--',lw=2, color=colour, label='LT {}'.format(lt))
        plt.fill_between(x_axis_time_steps, mean - std_dev, mean + std_dev, color=colour, alpha=0.1,
                         label='LT {}'.format(lt))

    plt.ylabel("LT's Inventory")
    plt.xlabel("Time step")
    # plt.legend(loc="lower left")

    plt.subplot(3, 2, 3)
    for lp, colour in zip(lps, lp_colours):
        data = simulation_results['LP'][lp]['w']
        mean = data['mean']
        std_dev = data['std']

        plt.plot(x_axis_time_steps, mean, linestyle='--', lw=2, color=colour, label='LP {}'.format(lp))
        plt.fill_between(x_axis_time_steps, mean - std_dev, mean + std_dev, color=colour, alpha=0.1,
                         label='LP {}'.format(lp))

    plt.ylabel("LP's Cash")
    plt.xlabel("Time step")
    # plt.legend(loc="lower left")

    plt.subplot(3, 2, 4)
    for lt, colour in zip(lts, lt_colours):
        data = simulation_results['LT'][lt]['w']
        mean = data['mean']
        std_dev = data['std']

        plt.plot(x_axis_time_steps, mean, linestyle='--', lw=2, color=colour, label='LT {} Mean'.format(lt))
        plt.fill_between(x_axis_time_steps, mean - std_dev, mean + std_dev, color=colour, alpha=0.1,
                         label='LT {} - 1 Stdv'.format(lt))

    plt.ylabel("LT's Cash")
    plt.xlabel("Time step")
    # plt.legend(loc="lower left")

    plt.subplot(3, 2, 5)
    for lp, colour in zip(lps, lp_colours):
        data = simulation_results['LP'][lp]['z']#no_of_exec_trades
        mean = data['mean']
        std_dev = data['std']

        bid = simulation_results['LP'][lp]['bid']['mean']
        ask = simulation_results['LP'][lp]['ask']['mean']

        plt.plot(x_axis_time_steps, mean, linestyle='--', lw=2, color=colour, label='LP {}'.format(lp))
        plt.fill_between(x_axis_time_steps, mean - std_dev, mean + std_dev, color=colour, alpha=0.1,
                         label='LP {}'.format(lp))

    plt.ylabel("LP's Marginal Rate")
    plt.xlabel("Time step")
    # plt.legend(loc="best")

    plt.subplot(3, 2, 6)

    for lp, colour in zip(lps, lp_colours):
        data = simulation_results['LP'][lp]['fees_collected']
        mean = data['mean']
        std_dev = data['std']

        plt.plot(x_axis_time_steps, mean, linestyle='--', lw=2, color=colour, label='LP {}'.format(lp))
        plt.fill_between(x_axis_time_steps, mean - std_dev, mean + std_dev, color=colour, alpha=0.1,
                         label='LP {}'.format(lp))

    plt.ylabel("Fees collected by LP")
    plt.xlabel("Time step")
    # plt.legend(loc="best")
    labels = []
    for l in lps:
        labels.append('LP {} Mean'.format(l))
    for l in lps:
        labels.append('LP {} Mean ± Std Dev'.format(l))
    for l in lts:
        labels.append('LT {} Mean'.format(l))
    for l in lts:
        labels.append('LT {} Mean ± Std Dev'.format(l))
    fig.legend(ncol=int(len(lps)),labels=labels,
               loc="upper center") #len(labels)/

    # Adjusting the sub-plots
    plt.subplots_adjust(top=0.92)
    plt.show()

def plot_external_market_maker_simulation_results(external_market_pricing, sim_stat, title=''):
    fig = plt.figure(figsize=(12,6))

    lps = list(sim_stat['LP'].keys())
    lts = list(sim_stat['LT'].keys())
    lp_colours = [x for x in mcolors.TABLEAU_COLORS][:len(lps)]
    lt_colours = ['brown','pink','gray','olive','crimson','navy','peru','black','darkred']
    lp_colours = ['olive' , lp_colours[0]]
    x_axis_time_steps= sim_stat['time']

    plt.subplot(1, 2, 1)
    plt.plot(x_axis_time_steps, external_market_pricing, linestyle='--', lw=2, color='black', label='Binance')
    for lp, colour in zip(lps, lp_colours):
        data = simulation_results['LP'][lp]['z']#no_of_exec_trades
        mean = data['mean']
        std_dev = data['std']

        bid = simulation_results['LP'][lp]['bid']['mean']
        ask = simulation_results['LP'][lp]['ask']['mean']
        if lp == 1:
            latex_legend = r"$Z^{(1)}_t$"
            latex_legend_2 = r"$[Z^{(1)}_t - \delta^{{(1)},b}_t,Z^{(1)}_t + \delta^{{(1)},a}_t] $"
        else:
            latex_legend = r"$Z^{(2)}_t$"
            latex_legend_2 = r"$Z^{(2)}_t - \delta^{{(2)},b}_t, Z^{(2)}_t + \delta^{{(2)},a}_t$"
        # message = f"This is a LaTeX fraction: {latex_code}"

        plt.plot(x_axis_time_steps, mean, linestyle='--', lw=2, color=colour, label=latex_legend)
        plt.fill_between(x_axis_time_steps, bid, ask, color=colour, alpha=0.2,
                         label=latex_legend_2)

    plt.ylabel("Price")
    plt.xlabel("Time step")
    plt.legend(loc="best")

    plt.subplot(1, 2, 2)

    for lp, colour in zip(lps, lp_colours):
        data = simulation_results['LP'][lp]['fees_collected']['mean']
        plt.plot(x_axis_time_steps, data, linestyle='--', lw=2, color=colour, label='LP {} Earnings'.format(lp))

        data = simulation_results['LP'][lp]['pool_value']['mean']
        plt.plot(x_axis_time_steps, data, linestyle='-', lw=2, color=colour, label='LP {} Pool Value'.format(lp))


    plt.ylabel("Performance")
    plt.xlabel("Time step")
    plt.legend(loc="best")

    plt.show()
# -------------------------End Plotting Functions-------------------
class external_market_maker:
    def __init__(self, Z, bid_ask_spread, drift, volatility):
        self.price_model = GeometricBrownianMotionMidpriceModel(
            drift = drift, volatility  = volatility, initial_price = Z, terminal_time = terminal_time, step_size = step_size,
        )#            drift = drift, volatility  = volatility, initial_price = Z, terminal_time = 10, step_size = 0.01,

        self.Z = self.price_model.current_state[0][0]
        self.bid_and_ask = (self.Z-bid_ask_spread,self.Z+bid_ask_spread)
        self.wealth = 0
        self.current_inv = 0
        self.bid_ask_spread = bid_ask_spread

    def LP_reset(self):
        self.price_model = GeometricBrownianMotionMidpriceModel(
            drift = self.price_model.drift, volatility  = self.price_model.volatility, initial_price = self.price_model.initial_state[0][0], terminal_time = self.price_model.terminal_time, step_size = self.price_model.step_size,
        )
        self.Z = self.price_model.current_state[0][0]
        self.wealth = 0
        self.current_inv = 0
        self.bid_and_ask = (self.Z-self.bid_ask_spread,self.Z+self.bid_ask_spread)

    def observe_pool_attributes(self, id):
        print('LP External Market {}, Z: {}, Current (BID,ASK):{}'.format(id, round(self.Z,5),  self.get_pool_bid_and_ask(None)))
    def get_pool_bid_and_ask(self,_):
        self.bid_and_ask = (self.Z - self.bid_ask_spread, self.Z + self.bid_ask_spread)
        return self.bid_and_ask
    def observe_pool_bid_and_ask(self):
        return self.bid_and_ask


    def trade(self, bid_takes_place, execution_time=0.):
        if bid_takes_place:
            self.wealth -=self.observe_pool_bid_and_ask()[0]
            self.current_inv += 1
        else:
            self.wealth += self.observe_pool_bid_and_ask()[1]
            self.current_inv -= 1


    def update(self):
        self.price_model.current_state = (
                self.price_model.current_state
                + self.price_model.drift * self.price_model.current_state * self.price_model.step_size
                + self.price_model.volatility
                * self.price_model.current_state
                * np.sqrt(self.price_model.step_size)
                * self.price_model.rng.normal(size=(self.price_model.num_trajectories, 1))
        )
        self.Z = self.price_model.current_state[0][0]
        self.get_pool_bid_and_ask(None)

class Binance_external_pricing:
    def __init__(self):
        self.BinanceData = self.get_data()
        self.price_counter = 0
        self.bid_ask_spread = 0
        self.Z = self.BinanceData.iloc[self.price_counter].Binance
        self.bid_and_ask = self.get_pool_bid_and_ask(None)


    def get_pool_bid_and_ask(self,_):
        self.bid_and_ask = (self.BinanceData.iloc[self.price_counter].Binance - self.bid_ask_spread, self.BinanceData.iloc[self.price_counter].Binance + self.bid_ask_spread)
        return self.bid_and_ask

    def update(self):
        self.price_counter+=1
        self.Z = self.BinanceData.iloc[self.price_counter].Binance
        self.get_pool_bid_and_ask(None)
        return self.Z

    def get_data(self):
        LPdata = pd.read_csv("data/LPdata")
        LTdata = pd.read_csv("data/LTdata")
        BinanceData = pd.read_csv("data/BinanceData")

        LPdata['timestamp'] = pd.to_datetime(LPdata['timestamp'])
        LTdata['timestamp'] = pd.to_datetime(LTdata['timestamp'])
        BinanceData['time'] = pd.to_datetime(BinanceData['time'])

        LPdata = LPdata.set_index('LPIndex')
        BinanceData = BinanceData.set_index('time')

        from_datetime = '2021-08-01 09:00'
        to_datetime = '2021-08-01 09:30'
        trade_date = '2021-08-01'

        oneDayLTdata, oneDayLPdata, oneDaybinanceLTdata, \
        fill_exponents, pool_sizes, hist_prices, \
        initial_convexity, trade_sizes, bothPrices = get_LT_LP_Binance_data(LTdata, LPdata, BinanceData, trade_date,
                                                                            from_datetime, to_datetime)
        bothPrices.columns = ['Binance', 'Uniswap']
        return  bothPrices



class LP_pool_agent:
    def __init__(self, LP_prop_specific):
        impact_function_set = assign_impact_fn(LP_prop_specific['impact_fn_set'])
        self.pool_agent =  get_pool_agent(LP_prop_specific['arrival_rate'], LP_prop_specific['phi'], LP_prop_specific['alpha'], LP_prop_specific['fill_exponent'],
                                LP_prop_specific['initial_inventory_pool'], LP_prop_specific['target_inventory'],
                                LP_prop_specific['jump_size_L'], LP_prop_specific['unit_size'], LP_prop_specific['min_inventory_pool'], LP_prop_specific['max_inventory_pool'],
                                LP_prop_specific['initial_price'], max_depth,
                                terminal_time, step_size,
                                seed, n_steps, num_trajectories, impact_function_set[0], impact_function_set[1],
                                verbose)
        self.pool_agent_copy = copy.deepcopy(get_pool_agent(LP_prop_specific['arrival_rate'], LP_prop_specific['phi'], LP_prop_specific['alpha'], LP_prop_specific['fill_exponent'],
                                LP_prop_specific['initial_inventory_pool'], LP_prop_specific['target_inventory'],
                                LP_prop_specific['jump_size_L'], LP_prop_specific['unit_size'], LP_prop_specific['min_inventory_pool'], LP_prop_specific['max_inventory_pool'],
                                LP_prop_specific['initial_price'], max_depth,
                                terminal_time, step_size,
                                seed, n_steps, num_trajectories, impact_function_set[0], impact_function_set[1],
                                verbose))

        # self.ensure_correct_bid_ask_spread()


        self.wealth = 0
        self.Z = LP_prop_specific['initial_price']
        self.current_inv = LP_prop_specific['initial_inventory_pool']
        self.bid_and_ask = self.get_pool_bid_and_ask(0)
        self.impact_function_set = LP_prop_specific['impact_fn_set']

        self.initial_Z = LP_prop_specific['initial_price']
        self.initial_inv = LP_prop_specific['initial_inventory_pool']

        # PLOT BID ASK SPREAD
        # self.plot_bid_and_ask_spread()

    # -----------------Helper Plotting Functions-----------------
    def plot_bid_and_ask_spread(self,show_plot=True):
        pool_agent = self.pool_agent
        # plt.title("H_t = O(t,y)/g")
        # plt.plot([y for y in pool_agent.inventory_space], pool_agent._calculate_ht(0.))
        # plt.grid()
        # plt.xticks(np.arange(pool_agent.inventory_space[0], pool_agent.inventory_space[-1], 1.0))
        # plt.show()
        #
        # plt.title("1/G(O(t,y+G) - O(t,y))")
        # plt.plot([y for y in pool_agent.inventory_space[:-1]],
        #          (pool_agent._calculate_ht(0.)[1:] - pool_agent._calculate_ht(0.)[:-1]))
        # plt.grid()
        # plt.xticks(np.arange(pool_agent.inventory_space[0], pool_agent.inventory_space[-1], 1.0))

        # plt.show()
        bid_list = []
        ask_list = []
        y_inv = []
        for y in range(-8,8):
            delta_bid, delta_ask = pool_agent._calculate_deltas(current_time=0., inventories=y)[0]
            bid_list.append(delta_bid)
            ask_list.append(delta_ask)
            y_inv.append(y)

        plt.title(r"Optimal control for $\eta(y)$ with (L={})".format(self.pool_agent.env.midprice_model.jump_size_L))
        plt.plot(y_inv, bid_list, "-b", label=r"$\delta^b(0,y)$", linewidth=2.0)
        plt.plot(y_inv, ask_list, "-r", label=r"$\delta^a(0,y)$", linewidth=2.0)
        # plt.plot(y_inv, [a + b for a, b in zip(bid_list, ask_list)], "-y", label="Sum", linewidth=2.0)
        plt.xlabel('LP inventory y')
        plt.ylabel('Optimal Control $\delta(0,y)$')
        plt.grid()
        plt.xticks(y_inv)
        plt.legend(loc="upper left")
        if not show_plot:
            return plt
        plt.show()

    def plot_impact_fn(self, show_plot=True):
        y_inv = []
        impact_bid = []
        impact_ask = []
        for y in range(-8, 8):
            eta_bid = self.pool_agent.env.midprice_model.eta_bid(y,
                                                            self.pool_agent.env.trader.unit_size,
                                                            self.pool_agent.env.midprice_model.jump_size_L)
            eta_ask = self.pool_agent.env.midprice_model.eta_ask(y,
                                                            self.pool_agent.env.trader.unit_size,
                                                            self.pool_agent.env.midprice_model.jump_size_L)

            impact_bid.append(eta_bid)
            impact_ask.append(eta_ask)

            y_inv.append(y)

        plt.title("Impact $\eta(y)$ with (L={})".format(self.pool_agent.env.midprice_model.jump_size_L))
        plt.xlabel('LP inventory y')
        plt.ylabel(r'Impact $\eta(y)$')
        plt.plot(y_inv, impact_bid, "-b", label=r"$\eta^b(y)$", linewidth=2.0)
        plt.plot(y_inv, impact_ask, "-r", label=r"$\eta^a(y)$", linewidth=2.0)
        plt.grid()
        plt.xticks(y_inv)
        plt.yticks(np.linspace(round((min(impact_bid)), 2), self.pool_agent.env.midprice_model.jump_size_L, 9))
        plt.legend(loc="upper left")
        if not show_plot:
            return plt
        plt.show()

    # -----------------Observe LP attributes-----------------
    def LP_reset(self):
        self.pool_agent = copy.deepcopy(self.pool_agent_copy)
        self.wealth = 0
        self.Z = self.initial_Z
        self.current_inv = self.initial_inv
        self.bid_and_ask = self.get_pool_bid_and_ask(0)

    def observe_pool_attributes(self, id):
        print('LP id: {}, Wealth: {} , Z: {}, Unit_trade: {}, Current inv: {}, Current (BID,ASK):{}'.format(id, round(self.wealth,5), round(self.Z,5),self.pool_agent.env.trader.unit_size, self.current_inv, self.observe_pool_bid_and_ask()))

    def get_pool_bid_and_ask(self, execution_time):
        delta_bid, delta_ask =  self.pool_agent._calculate_deltas(current_time=execution_time, inventories=self.current_inv)[0]
        # delta_bid, delta_ask = np.round(self.pool_agent._calculate_deltas(current_time=execution_time, inventories=self.current_inv),3)[0]
        if DOUBLE_CHECK:
            self.ensure_correct_bid_ask_spread(current_time=execution_time)
            if np.round(delta_bid - self.bid_ask_test[self.current_inv][0],9) != 0  or np.round(delta_ask- self.bid_ask_test[self.current_inv][
                1],9)!=0: raise BidAskError
        self.bid_and_ask = (self.Z - delta_bid, self.Z + delta_ask)

        if delta_bid > 9000 or delta_ask > 9000:
            print('Bid at max capacity')
        return self.bid_and_ask

    def observe_pool_bid_and_ask(self):
        return self.bid_and_ask
    # -----------------End Observe LP attributes-----------------

    # ----------------- Data integrity helper functions -----------------
    def ensure_correct_impact(self, new_Z, bid_takes_place,z_0,y,L,G):
        # ensure_correct_calc mathematically
        eta_function_used = self.impact_function_set
        if bid_takes_place:
            if y >= 0:
                Z_math = z_0 - np.divide(L * G , G + y)

            elif y == -G:
                if eta_function_used == 0:
                    Z_math = z_0
                elif eta_function_used == 1:
                    Z_math = z_0 + L
                elif eta_function_used == 2:
                    Z_math = z_0 - L
            elif y < 0 and y + G > 0:
                Z_math = z_0 - np.divide(L * G , G + y)
            elif y < 0 and y + G < 0:
                Z_math = z_0 + np.divide(L * G , G + y)
            else:
                raise UnconsideredCase
        else:
            if y <= 0:
                Z_math = z_0 + np.divide(L * G , G - y)
            elif y == G:
                if eta_function_used == 0:
                    Z_math = z_0
                elif eta_function_used == 1:
                    Z_math = z_0 - L
                elif eta_function_used == 2:
                    Z_math = z_0 + L
            elif y > 0 and y - G < 0:
                Z_math = z_0 - np.divide(L * G , G - y)
            elif y > 0 and y - G > 0:
                Z_math =  z_0 - np.divide(L * G , G - y)
            else:
                raise UnconsideredCase
        if np.abs(np.round(new_Z - Z_math, 10)) != 0:
            raise ZValueMismatch

    def ensure_correct_bid_ask_spread(self, current_time = 0.):
        inv_dict = {}
        for inv in self.pool_agent.inventory_space:
            delta_bid, delta_ask = self.pool_agent._calculate_deltas(current_time=current_time, inventories=inv)[0]
            inv_dict[inv] = [delta_bid, delta_ask]

        self.bid_ask_test = inv_dict

    # -----------------End Data integrity helper functions -----------------

    #  -----------------Main LP Trading Function -----------------
    def trade(self, bid_takes_place, execution_time=0.):
        impact_bid = self.pool_agent.env.midprice_model.eta_bid(self.current_inv, self.pool_agent.env.trader.unit_size,
                                                           self.pool_agent.env.midprice_model.jump_size_L)
        impact_ask = self.pool_agent.env.midprice_model.eta_ask(self.current_inv, self.pool_agent.env.trader.unit_size,
                                                           self.pool_agent.env.midprice_model.jump_size_L)

        price_before_impact = self.Z
        inv_before_impact = self.current_inv
        if bid_takes_place:
            self.wealth -= self.observe_pool_bid_and_ask()[0] * self.pool_agent.env.trader.unit_size
            self.current_inv += self.pool_agent.env.trader.unit_size
            self.Z -= impact_bid
            if DOUBLE_CHECK:
                self.ensure_correct_impact(self.Z,bid_takes_place,price_before_impact, inv_before_impact,self.pool_agent.env.midprice_model.jump_size_L, self.pool_agent.env.trader.unit_size )

        else:
            self.wealth += self.observe_pool_bid_and_ask()[1] * self.pool_agent.env.trader.unit_size
            self.current_inv -= self.pool_agent.env.trader.unit_size
            self.Z += impact_ask
            if DOUBLE_CHECK:
                self.ensure_correct_impact(self.Z, bid_takes_place, price_before_impact, inv_before_impact,
                                           self.pool_agent.env.midprice_model.jump_size_L,
                                           self.pool_agent.env.trader.unit_size)

        # update posting bid and ask to reflect new pricing
        self.get_pool_bid_and_ask(execution_time)

class LT_agent:
    def __init__(self, LT_prop):
        self.inv = LT_prop['starting_inv']
        self.wealth = LT_prop['starting_cash']
        self.initial_inv = LT_prop['starting_inv'] #for book keeping
        self.initial_wealth = LT_prop['starting_cash'] #for book keeping
        self.probs = LT_prop['probs']
        self.arbitrageur = LT_prop['arbitrageur']

        self.next_action_in_arbitrage = LT_hold


        # self.action_space = 3
        # self.input_dim = 6
        # self.rl_agent = PPO_class.Agent(n_actions=self.action_space, input_dims=[self.input_dim],gamma=0.99, alpha_actor=0.0003,alpha_critic= 0.0003, gae_lambda=0.95,
        #          policy_clip=0.05, batch_size=64, n_epochs=10, ent_coef=0.01 ,vf_coef = 0.5,  nn_actor=128, nn_critic=128)

    def LT_reset(self):
        self.inv = self.initial_inv
        self.wealth = self.initial_wealth
        self.next_action_in_arbitrage = LT_hold

    def observe_LT_attributes(self, id):
        print('LT id:{} , Current Wealth: {} , Current inv: {}, Initial Wealth: {}, Initial Inv: {}'.format(id,round(self.wealth,5),
                                                                                            self.inv,
                                                                                            round(self.initial_wealth,5),
                                                                                            self.initial_inv))

    def trade(self, bid_takes_place, price, number_of_units):
        if bid_takes_place: # LP buys LT Sells
            self.inv -= number_of_units
            self.wealth += price*number_of_units
        else:
            self.inv += number_of_units
            self.wealth -= price * number_of_units

    def action_trade_direction(self,obs=None):
        if RL_SETTING:
            action, prob, val, entropy = self.rl_agent.choose_action(obs)
            return action, prob, val, entropy
        else:
            # return np.random.choice([LT_take_bid, LT_hold, LT_take_ask], p=self.probs)
            if not self.arbitrageur:
                return np.random.choice([LT_take_bid, LT_hold, LT_take_ask],p=self.probs) ,False
            else:
                if External_Market_Arbitrage_Flag:
                    external_market_bid_and_ask = External_Market.get_pool_bid_and_ask(None)
                    CQV_best_bid = obs[0]
                    CQV_best_ask = obs[1]

                    if external_market_bid_and_ask [0] > CQV_best_ask:
                        self.trade(bid_takes_place = True, price=external_market_bid_and_ask [0], number_of_units=LP_4['unit_size']) #trade with external
                        action = LT_take_ask
                        # print('arb LT Take Ask')
                        return action, True
                    elif external_market_bid_and_ask[1] < CQV_best_bid:
                        self.trade(bid_takes_place=False, price=external_market_bid_and_ask[1],
                                   number_of_units=LP_4['unit_size'])
                        action = LT_take_bid
                        print('arb LT Take Bid')
                        return action, True
                    else:
                        action = np.random.choice([LT_take_bid, LT_hold, LT_take_ask], p=self.probs)
                        return action,False

                else:
                    return np.random.choice([LT_take_bid, LT_hold, LT_take_ask], p=self.probs),False


class Router_env:
    def __init__(self, LPs, LTs):
        self.LPs = LPs
        self.LTs = LTs
        self.best_bid = None
        self.best_ask = None
        self.best_bidder = None
        self.best_asker = None
        self.update_quotes(0)
        self.current_market_liquidity = self.calc_current_market_liquidity()
        self.market_impact_fns = assign_impact_fn(0)
        self.jump_size_L = 0.01 # market jumpsize
        self.total_wealth = 0
        self.observation = None  # RL [inv, wealth, initial_inv, initial_wealth,current_bid, current_ask, current_Z, price_change, pool_inv]

    def router_reset_env(self):
        for lp in self.LPs:
            self.LPs[lp].LP_reset()
        for lt in self.LTs:
            self.LTs[lt].LT_reset()
        self.best_bid = None
        self.best_ask = None
        self.best_bidder = None
        self.best_asker = None
        self.update_quotes(0)
        self.current_market_liquidity = self.calc_current_market_liquidity()
        self.market_impact_fns = assign_impact_fn(0)
        self.jump_size_L = 0.01
        self.observation = None

    # -----------------Helper Plotting Functions-----------------
    def append_statistics(self, simulation_statistics, execution_time=0.):
        simulation_statistics['time'].append(execution_time)
        for lt in self.LTs:
            simulation_statistics['LT'][lt]['inv'].append(self.LTs[lt].inv)
            simulation_statistics['LT'][lt]['w'].append(self.LTs[lt].wealth)
        for lp in self.LPs:
            simulation_statistics['LP'][lp]['inv'].append(self.LPs[lp].current_inv)
            simulation_statistics['LP'][lp]['w'].append(self.LPs[lp].wealth)
            simulation_statistics['LP'][lp]['z'].append(self.LPs[lp].Z)
            simulation_statistics['LP'][lp]['pool_value'].append(self.LPs[lp].Z * self.LPs[lp].current_inv + self.LPs[lp].wealth)
            if self.LPs[lp].bid_and_ask[0] < self.LPs[lp].Z - 5000:
                simulation_statistics['LP'][lp]['bid'].append(self.LPs[lp].Z)
            else:
                simulation_statistics['LP'][lp]['bid'].append(self.LPs[lp].bid_and_ask[0])
            if self.LPs[lp].bid_and_ask[1] > self.LPs[lp].Z + 5000:
                simulation_statistics['LP'][lp]['ask'].append( self.LPs[lp].Z)
            else:
                simulation_statistics['LP'][lp]['ask'].append(self.LPs[lp].bid_and_ask[1])

        return simulation_statistics
    # -----------------End Helper Plotting Functions-----------------

    # ----------------- Data integrity helper functions -----------------
    def ensure_no_wealth_slippage(self):
        wealth_calculation_current = 0
        wealth_calculation_initial = 0
        for lp in self.LPs:
            wealth_calculation_current += self.LPs[lp].wealth
            # note initial_lp_wealth = 0
        for lt in self.LTs:
            wealth_calculation_current += self.LTs[lt].wealth
            wealth_calculation_initial += self.LTs[lt].initial_wealth

        if np.round(wealth_calculation_current - wealth_calculation_initial,10) != 0:
            raise WealthSlippage
    # ----------------- End Data integrity helper functions -----------------

    # -----------------Observe Market attributes-----------------


    def observe_market_participants(self, include_bid_and_ask=True):
        if include_bid_and_ask:print("--------------------------Observe Market:--------------------------\nLPs:")
        print('LPs:')
        for lp in self.LPs:
            self.LPs[lp].observe_pool_attributes(lp)
        print('LTs:')
        for lt in self.LTs:
            self.LTs[lt].observe_LT_attributes(lt)
        if include_bid_and_ask:
            print("\n--------------------------Best Bid And Ask------------------------------")
            print('Best Bid: {}, Best Ask: {}'.format(self.best_bid ,self.best_ask))
            print("-------------------------------------------------------------------------\n")

    # -----------------End Market attributes-----------------

    # -----------------Router Market impact -----------------


    def calc_current_market_liquidity(self):
        total_market_inventory = 0
        for lp in self.LPs:
            total_market_inventory += self.LPs[lp].current_inv
        return total_market_inventory

    def impact_all_LPs_non_trading_LPs_as_if_they_traded(self,execution_time, exclude_LP, executed_trade_bid_takes_place, verbose=False):
        if exclude_LP not in  self.LPs:
            raise ImapctFnExcludeLPError
        for lp in self.LPs:
            if lp == exclude_LP:
                continue
            old_z = self.LPs[lp].Z
            if executed_trade_bid_takes_place:
                impact_factor = self.LPs[lp].pool_agent.env.midprice_model.eta_bid(self.LPs[lp].current_inv, self.LPs[lp].pool_agent.env.trader.unit_size,
                                                           self.LPs[lp].pool_agent.env.midprice_model.jump_size_L)
                self.LPs[lp].Z -= impact_factor
            else:
                impact_factor = self.LPs[lp].pool_agent.env.midprice_model.eta_ask(self.LPs[lp].current_inv, self.LPs[lp].pool_agent.env.trader.unit_size,
                                                           self.LPs[lp].pool_agent.env.midprice_model.jump_size_L)
                self.LPs[lp].Z += impact_factor

            # update posting bid and ask to reflect new pricing
            self.LPs[lp].get_pool_bid_and_ask(execution_time)
            if verbose: print('LP {} new Z:{} , old Z:{}'.format(lp, self.LPs[lp].Z, old_z))
            if impact_factor < 0:
                raise MarketImpactFunctionLogicalError

    def impact_all_LPs_with_trading_LP_marginal_change(self,execution_time, exclude_LP, marginal_change, verbose=False):
        if exclude_LP not in  self.LPs:
            raise ImapctFnExcludeLPError
        for lp in self.LPs:
            if lp == exclude_LP:
                continue
            old_z = self.LPs[lp].Z
            self.LPs[lp].Z += marginal_change

            # update posting bid and ask to reflect new pricing
            self.LPs[lp].get_pool_bid_and_ask(execution_time)
            if verbose: print('LP {} new Z:{} , old Z:{}'.format(lp, self.LPs[lp].Z, old_z))


    def impact_fn_all_LPs(self, execution_time, exclude_LP = -1, verbose=False):
        if exclude_LP != -1 and exclude_LP not in  self.LPs:
            raise ImapctFnExcludeLPError
        old_liquidity = self.current_market_liquidity
        self.current_market_liquidity = self.calc_current_market_liquidity()
        if old_liquidity == self.current_market_liquidity:
            if verbose: print('Impact: 0')
            return
        elif old_liquidity > self.current_market_liquidity: #less liquid -> Z higher price
            impact_factor = self.market_impact_fns[1](old_liquidity, old_liquidity-self.current_market_liquidity,self.jump_size_L)
            if verbose:print('Impact: +{}'.format(impact_factor))
            for lp in self.LPs:
                if lp == exclude_LP:
                    continue
                old_z = self.LPs[lp].Z
                self.LPs[lp].Z += impact_factor
                # update posting bid and ask to reflect new pricing
                self.LPs[lp].get_pool_bid_and_ask(execution_time)
                if verbose:print('LP {} new Z:{} , old Z:{}'.format(lp,self.LPs[lp].Z,old_z))

        elif old_liquidity < self.current_market_liquidity:
            impact_factor = self.market_impact_fns[0](old_liquidity, self.current_market_liquidity - old_liquidity, self.jump_size_L)
            if verbose:print('Impact: -{}'.format(impact_factor))
            for lp in self.LPs:
                if lp == exclude_LP:
                    continue
                old_z = self.LPs[lp].Z
                self.LPs[lp].Z -= impact_factor
                # update posting bid and ask to reflect new pricing
                self.LPs[lp].get_pool_bid_and_ask(execution_time)
                if verbose:print('LP {} new Z:{} , old Z:{}'.format(lp,self.LPs[lp].Z,old_z))
        else:
            raise MarketImpactFunctionLogicalError

        if impact_factor < 0:
            raise MarketImpactFunctionLogicalError

    # -----------------End Router Market impact -----------------

    # -----------------Main Trading Functions-----------------
    def update_quotes(self, execution_time, verbose = False):
        # get all bids and asks
        # get best bid (largest) and ask (smallest)
        best_bid = -1e10
        best_bidder = None
        best_ask = 1e10
        best_asker = None

        for key, lp in self.LPs.items():
            bid, ask = lp.get_pool_bid_and_ask(execution_time)

            if best_bid < bid:
                best_bid = bid
                best_bidder = key

            if best_ask > ask:
                best_ask = ask
                best_asker = key

        self.best_bid = best_bid
        self.best_ask = best_ask
        self.best_bidder = best_bidder
        self.best_asker = best_asker
        if verbose:
            print('Best Bid: {}, Best Ask: {}'.format(best_bid,best_ask))

    def trade(self, trading_LT:int, lt_action, execution_time, impact_all_LP_case=0, verbose=False):
        price_impact_track = {}

        if lt_action == LT_take_bid: bid_takes_place = True
        elif lt_action == LT_take_ask: bid_takes_place = False
        elif lt_action == LT_hold: return [np.nan,np.nan,np.nan]
        else:raise LT_Action_Error

        if bid_takes_place:
            trade_price = self.best_bid
            trading_LP = self.best_bidder
            if trade_price != self.LPs[trading_LP].observe_pool_bid_and_ask()[0]:
                raise PriceMismatch
        else:
            trade_price = self.best_ask
            trading_LP = self.best_asker
            if trade_price != self.LPs[trading_LP].observe_pool_bid_and_ask()[1]:
                raise PriceMismatch

        if verbose:
            print('----Trade Details LP: {}, LT: {}, is_Bid:{}, Price:{}----'.format(trading_LP,trading_LT,bid_takes_place, round(trade_price,4)))
        trading_LP_fee = np.abs(self.LPs[trading_LP].Z - trade_price)
        trading_LP_Z_before_trade = self.LPs[trading_LP].Z

        # Trade LP and LT and observe LP price impact
        self.LPs[trading_LP].trade(bid_takes_place, execution_time)
        self.LTs[trading_LT].trade(bid_takes_place, trade_price, 1 if isinstance(self.LPs[trading_LP],external_market_maker) else self.LPs[trading_LP].pool_agent.env.midprice_model.unit_size)

        if impact_all_LP_case != 0:
            if impact_all_LP_case == 1: # update non trading lps with total market impact
                executing_LP = trading_LP
                if verbose:
                    self.observe_market_participants()
                self.impact_fn_all_LPs(execution_time,exclude_LP=executing_LP, verbose= verbose)
            elif impact_all_LP_case == 2: # update all lps with market impact
                executing_LP = -1
                if verbose:
                    self.observe_market_participants()
                self.impact_fn_all_LPs(execution_time,exclude_LP=executing_LP, verbose= verbose)
            elif impact_all_LP_case == 3: # update non trading lps as if they traded
                self.impact_all_LPs_non_trading_LPs_as_if_they_traded(execution_time, exclude_LP=trading_LP,
                                                                 executed_trade_bid_takes_place = bid_takes_place, verbose=verbose)

            elif impact_all_LP_case == 4:  # update non trading lps with same change in marginal rate
                executing_LP = trading_LP
                delta_Z = self.LPs[trading_LP].Z - trading_LP_Z_before_trade
                self.impact_all_LPs_with_trading_LP_marginal_change(execution_time, exclude_LP=executing_LP,  marginal_change = delta_Z,
                                                               verbose=False)
            else:
                raise TotalMarketImpactCaseError
        self.update_quotes(execution_time=execution_time, verbose=False)
        return [trade_price, trading_LP, trading_LP_fee]

    # -----------------End Main Trading Functions-----------------

    # -----------------Market Simulation Function-----------------
    def simulate(self, starting_time= 0., number_of_trades=1, trade_case_evaluation = 0, impact_all_LP_case=0 ,sim_title='',verbose=False):
        if number_of_trades > n_steps:
            raise NumberOfTradesLargerThanSteps

        self.update_quotes(execution_time=0., verbose=False)
        print('Sim Number {}'.format(sim_title))
        # print('--------------------------Starting Simulation:{}---------------------'.format(sim_title))
        # print('--------------------------Simulation Initial Attributes---------------------')
        # self.observe_market_participants(include_bid_and_ask=False)

        simulation_statistics = {'LP': {}, 'LT': {}, 'time': [], 'trades': [], 'fees_collected': []}
        for lt in self.LTs:
            simulation_statistics['LT'][lt] = {'inv' : [], 'w':[]}
        for lp in self.LPs:
            simulation_statistics['LP'][lp] = {'inv': [], 'w': [], 'z':[],'bid':[], 'ask':[], 'pool_value' : []}
        simulation_statistics = self.append_statistics(simulation_statistics,execution_time=0)

        execution_time = starting_time
        counter = 0
        trade_log = []

        while execution_time <= terminal_time and counter<number_of_trades:
            self.update_quotes(execution_time=execution_time, verbose=False)
            if verbose:
                print('\n--------------------------------Trading Cycle {} Trading time {}--------------------------------'.format(counter, execution_time))
                self.observe_market_participants()

            # simulate different LTs submitting orders in different times
            lt_list = [lt for lt in self.LTs]
            random.shuffle(lt_list)

            if not External_Market_Arbitrage_Flag: #Normal environment where traders take turns to trade once
                for lt in lt_list:
                    if trade_case_evaluation == 0:
                        lt_action,_ = self.LTs[lt].action_trade_direction((self.best_bid,self.best_ask))
                    elif trade_case_evaluation == 1:
                        if 2*counter < number_of_trades:
                            lt_action = LT_take_bid
                        else:
                            lt_action = LT_take_ask
                    else:
                        raise CaseError

                    traded_price, trading_LP, trading_LP_fee = self.trade(trading_LT=lt, lt_action=lt_action, impact_all_LP_case= impact_all_LP_case, execution_time=execution_time,
                               verbose=verbose)
                    trade_log.append((traded_price,counter, trading_LP, lt, trading_LP_fee))

            else: #arbitrage environment where arbitragues can submit multiple trades in one timestep
                arbitrage_flag = True
                for lt in lt_list:
                    while arbitrage_flag:
                        # if LP hit their invenotry limit, recalibrate at market price
                        for lp in self.LPs:
                            if self.LPs[lp].current_inv == self.LPs[lp].pool_agent.max_inventory-1:
                                #sell all evtra inv in open market
                                self.LPs[lp].wealth += (self.LPs[lp].current_inv - self.LPs[lp].initial_inv) * External_Market.get_pool_bid_and_ask(None)[0]
                                self.LPs[lp].current_inv = self.LPs[lp].initial_inv
                            elif self.LPs[lp].current_inv == self.LPs[lp].pool_agent.min_inventory:
                                #buy inv in open market
                                self.LPs[lp].wealth -= (self.LPs[lp].initial_inv - self.LPs[lp].current_inv) * External_Market.get_pool_bid_and_ask(None)[1]
                                self.LPs[lp].current_inv = self.LPs[lp].initial_inv
                        lt_action, arbitrage_flag = self.LTs[lt].action_trade_direction((self.best_bid, self.best_ask))
                        if self.best_ask > 9000 and lt_action == LT_take_ask:
                            arbitrage_flag = False
                            continue
                        if self.best_bid < -9000 and lt_action == LT_take_bid:
                            arbitrage_flag = False
                            continue

                        traded_price, trading_LP, trading_LP_fee = self.trade(trading_LT=lt, lt_action=lt_action,
                                                                              impact_all_LP_case=impact_all_LP_case,
                                                                              execution_time=execution_time,
                                                                              verbose=verbose)

                        trade_log.append((traded_price, counter, trading_LP, lt, trading_LP_fee))
                        if lt_action != LT_hold:
                            self.LPs[trading_LP].wealth -= trading_LP_fee



            execution_time += step_size
            counter+=1
            # note for presentation purposes we do += stepsize therefore results are starting inv at new timestep
            simulation_statistics = self.append_statistics(simulation_statistics, execution_time=counter)
            # self.ensure_no_wealth_slippage()

            if External_Market_Arbitrage_Flag:

                External_Market.update()
                external_market_price_tracking.append(External_Market.Z)


        # print('--------------------------Simulation Final Attributes---------------------')
        # self.observe_market_participants(include_bid_and_ask=False)

        total_trading_cost = 0
        for lt in simulation_statistics['LT']:
            total_trading_cost += simulation_statistics['LT'][lt]['w'][-1]
        # print('Total Loss for LTs; {}'.format(total_trading_cost))

        # print('--------------------------Finish Simulation:{}---------------------\n\n'.format(sim_title))
        simulation_statistics['trade_log'] = trade_log
        return simulation_statistics


    # RL
    def populate_trader_observation(self, LT_id, previous_Z=-1):
        self.update_quotes(execution_time=0.)
        # if previous_Z == -1:
        previous_Z = np.log(self.LPs[self.best_bidder].Z)

        return [np.divide(self.LTs[LT_id].inv - LP_1['min_inventory_pool'],LP_1['max_inventory_pool']-LP_1['min_inventory_pool']),
                self.LTs[LT_id].wealth,
                np.divide(self.LTs[LT_id].initial_inv- LP_1['min_inventory_pool'],LP_1['max_inventory_pool']-LP_1['min_inventory_pool']),
                # self.LTs[LT_id].initial_wealth,
                np.log(self.best_bid),
                np.log(self.best_ask),
                # self.LPs[self.best_bidder].Z,
                # np.log(self.LPs[self.best_bidder].Z) - previous_Z,
                np.divide(self.LPs[self.best_bidder].current_inv- LP_1['min_inventory_pool'],LP_1['max_inventory_pool']-LP_1['min_inventory_pool'])]




# ------------------------------------------------------------------------------------------------------------------------

# Simulation of different market design
def simulate_sequence_of_bids_then_asks_for_different_market_conditions(number_of_trades, LT_bid_ask_sequence, plot=False):
    pool_agent_1 = LP_pool_agent(LP_1)
    pool_agent_2 = LP_pool_agent(LP_1)

    lt_agent_1 = LT_agent(LT_1)

    LP_dict = {1: pool_agent_1, 2: pool_agent_2}
    LT_dict = {1: lt_agent_1}

    router = Router_env(LP_dict, LT_dict)

    router.update_quotes(execution_time=0.)
    if plot:plot_bid_ask_spreads_and_impact_for_LPs(router)

    sim_stat_1 = router.simulate(starting_time=0., number_of_trades=number_of_trades, trade_case_evaluation=LT_bid_ask_sequence, impact_all_LP_case =0,sim_title='', verbose=False)
    if plot:plot_simulation_results(router,sim_stat_1,'Case 1 - Only Trading LP Updates - No External Impact.')

    router.router_reset_env()
    sim_stat_2 = router.simulate(starting_time=0., number_of_trades=number_of_trades, trade_case_evaluation=LT_bid_ask_sequence, impact_all_LP_case =3,sim_title='', verbose=False)
    if plot:plot_simulation_results(router, sim_stat_2, 'Case 2 - Non-trading LP updates as if it traded.')

    router.router_reset_env()
    sim_stat_3 = router.simulate(starting_time=0., number_of_trades=number_of_trades, trade_case_evaluation=LT_bid_ask_sequence, impact_all_LP_case=4,
                                 sim_title='', verbose=False)
    if plot:plot_simulation_results(router, sim_stat_3, 'Case 3 - Non-trading LP copies trading LP\'s marginal rate change')

    router.router_reset_env()
    sim_stat_4 = router.simulate(starting_time=0., number_of_trades=number_of_trades, trade_case_evaluation=LT_bid_ask_sequence, impact_all_LP_case=2, sim_title='',
                                 verbose=False)
    if plot: plot_simulation_results(router, sim_stat_4, 'Case 4 - All LPs also update w.r.t General Market Impact.')

    if plot:plot_different_trade_executions([sim_stat_1,sim_stat_2,sim_stat_3,sim_stat_4])

    return [sim_stat_1,sim_stat_2,sim_stat_3,sim_stat_4]

simulate_sequence_of_bids_then_asks_for_different_market_conditions(number_of_trades=32, LT_bid_ask_sequence=1,plot=True)

# ------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------

# Parallel simulations of Market Models
from concurrent.futures import ThreadPoolExecutor
def wrapper(instance, starting_time, number_of_trades, trade_case_evaluation, impact_all_LP_case, sim_index):
    instance_copy = copy.deepcopy(instance)
    instance_copy.router_reset_env()
    return instance_copy.simulate(starting_time, number_of_trades, trade_case_evaluation, impact_all_LP_case,
                                 sim_title=sim_index, verbose=False)

def simulate_markets(number_of_simulations, LPs_settings, LTs_settings, sim_args):
    LP_dict = {}
    for index, value in enumerate(LPs_settings, start=1):
        LP_dict[index] =  LP_pool_agent(value)

    LT_dict = {}
    for index, value in enumerate(LTs_settings, start=1):
        LT_dict[index] =  LT_agent(value)

    router = Router_env(LP_dict, LT_dict)
    router.update_quotes(execution_time=0.)

    # Sim 1
    arguments = [(router, sim_args[0], sim_args[1],sim_args[2],sim_args[3], i) for i in range(0,number_of_simulations)]
    with ThreadPoolExecutor() as executor:
        simulation = [executor.submit(wrapper, *arg) for arg in arguments]
        simulation_results = [simulation.result() for simulation in simulation]

    # averaging statistics
    mult_sim_stats = {'LP': {}, 'LT': {}}
    for actor in mult_sim_stats:
        for lp_or_lt in simulation_results[0][actor]:
            mult_sim_stats[actor][lp_or_lt] = {}
            for attribute in simulation_results[0][actor][lp_or_lt]:
                attribute_stats = np.array([sim[actor][lp_or_lt][attribute] for sim in simulation_results])
                attribute_mean = np.mean(attribute_stats, axis=0)
                attribute_std = np.std(attribute_stats, axis=0)
                mult_sim_stats[actor][lp_or_lt][attribute] = {}
                mult_sim_stats[actor][lp_or_lt][attribute]['mean'] = attribute_mean
                mult_sim_stats[actor][lp_or_lt][attribute]['std'] = attribute_std

    for lp in simulation_results[0]['LP']:
        fees_collected_in_sim = []
        no_of_exec_trades_in_sim = []
        for sim in simulation_results:
            trade_log = sim['trade_log']
            trades_exec_by_lp = [z for z in trade_log if z[2] == lp]
            timestep_and_fees_collected =[(z[1],z[4]) for z in trades_exec_by_lp]
            no_of_executed_trades_dict = defaultdict(int)
            fees_dict = defaultdict(int)
            for timestep, fees in timestep_and_fees_collected:
                fees_dict[timestep] = fees_dict[timestep] + fees
                no_of_executed_trades_dict[timestep] = no_of_executed_trades_dict[timestep] + 1

            min_timestep = min(trade[1] for trade in trade_log)
            max_timestep = max(trade[1] for trade in trade_log)

            cumulative_sum = 0
            result = []

            no_of_executed_trades = []
            cum_sum_trades = 0

            result.append(0.) #before trading starts
            no_of_executed_trades.append(0)

            # Iterate through the range of timesteps and calculate the cumulative sum
            for timestep in range(min_timestep, max_timestep + 1):
                fees = fees_dict[timestep]
                cumulative_sum += fees
                result.append(cumulative_sum)


                exec_trades = no_of_executed_trades_dict[timestep]
                cum_sum_trades += exec_trades
                no_of_executed_trades.append(cum_sum_trades)


            fees_collected_in_sim.append(result)
            no_of_exec_trades_in_sim.append(no_of_executed_trades)

        attribute_stats = np.array([z for z in fees_collected_in_sim])
        attribute_mean = np.mean(attribute_stats, axis=0)
        attribute_std = np.std(attribute_stats, axis=0)
        mult_sim_stats['LP'][lp]['fees_collected'] = {}
        mult_sim_stats['LP'][lp]['fees_collected']['mean'] = attribute_mean
        mult_sim_stats['LP'][lp]['fees_collected']['std'] = attribute_std

        attribute_stats = np.array([z for z in no_of_exec_trades_in_sim])
        attribute_mean = np.mean(attribute_stats, axis=0)
        attribute_std = np.std(attribute_stats, axis=0)
        mult_sim_stats['LP'][lp]['no_of_exec_trades'] = {}
        mult_sim_stats['LP'][lp]['no_of_exec_trades']['mean'] = attribute_mean
        mult_sim_stats['LP'][lp]['no_of_exec_trades']['std'] = attribute_std

    mult_sim_stats['time'] = simulation_results[0]['time']

    return mult_sim_stats

number_of_simulations = 100
number_of_trades= 100
random_LT_trading = True
market_impact_case=0
#
# # Parallel simulations of Market Models with diff invenotry penalties
simulation_results = simulate_markets(number_of_simulations, [LP_2_no_inv_penelties,LP_2_significant_inv_penelties],[LT_1,LT_1], [0., number_of_trades,0 if random_LT_trading else 1,market_impact_case])

plot_multiple_simulation_results(simulation_results, title='')
# # ------------------------------------------------------------------------------------------------------------------------
# # Parallel simulations of Market Models with diff kappa
simulation_results = simulate_markets(number_of_simulations, [LP_3_kappa_2,LP_3_kappa_1_8],[LT_1,LT_1], [0., number_of_trades,0 if random_LT_trading else 1,market_impact_case])

plot_multiple_simulation_results(simulation_results, title='')

External_Market_Arbitrage_Flag = True
External_Market = Binance_external_pricing()
External_Market_price_counter = 0
price_Z           = External_Market.Z
price_S           = External_Market.Z
external_market_price_tracking = [price_Z]
LP_4['initial_price'] = price_Z
LP_5['initial_price'] = price_Z
number_of_trades = len(External_Market.BinanceData.index)-1

number_of_simulations = 1
simulation_results = simulate_markets(number_of_simulations, [LP_4, LP_5],[LT_3], [0., number_of_trades,0 if random_LT_trading else 1,market_impact_case])

plot_external_market_maker_simulation_results(external_market_price_tracking, simulation_results, title='')
#
















