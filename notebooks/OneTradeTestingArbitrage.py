import math

import matplotlib.pyplot


class UnconsideredCase(Exception):
    pass

class ZValueMismatch(Exception):
    pass

class LTCashMismatch (Exception):
    pass

class IncorrectStepInTimeValue(Exception):
    pass

class ETAFunctionError(Exception):
    pass

from nb_utils import rescale_plot, mtick, md, run_simulation,\
                     getSimulationData, get_pool_agent, get_arb_env,\
                     get_LT_LP_Binance_data, pd, plt, np, ASSET_PRICE_INDEX,\
                     datetime, get_binance_month_data, plot_impact_curves,\
                     runOneSimulation, getOneSimulationData, plot_one_sim_result

import numpy as np
############################
# LP strategy parameters
############################
jump_size_L            = 0.01
phi                    = 1e-4
alpha                  = 1e-4
fill_exponent          = 2
initial_inventory_pool = 0
arrival_rate           = 100

############################
# Pool liquidity parameters
############################
initial_price          = 100
target_inventory       = initial_inventory_pool
min_inventory_pool     = initial_inventory_pool - 1500.
max_inventory_pool     = initial_inventory_pool + 1500.
unit_size              = 1

############################
# Others
############################
max_depth              = 0
terminal_time          = 30/60/24 #/24/ # 30 minutes
n_steps                = 100
step_size              = terminal_time/n_steps
seed                   = 1
num_trajectories       = 1
verbose                = False

############################
# Create agent
############################

#Paper eta functions
eta_function_used =0
if eta_function_used == 0:
    def eta_func_ask(y, Delta, L):
        if y==Delta: return 0
        if y-Delta>0: return L * Delta  / (y - Delta)
        if y-Delta<0: return -L * Delta / (y - Delta)

    def eta_func_bid(y, Delta, L):
        if y+Delta==0: return 0
        if y+Delta>0: return L * Delta  / (y + Delta)
        if y+Delta<0: return -L * Delta / (y + Delta)
elif eta_function_used == 1:
    def eta_func_ask(y, Delta, L):
        if y-Delta==0: return -L
        if y-Delta>0: return L * Delta  / (y - Delta)
        if y-Delta<0: return -L * Delta / (y - Delta)

    def eta_func_bid(y, Delta, L):
        if y+Delta==0: return -L
        if y+Delta>0: return L * Delta  / (y + Delta)
        if y+Delta<0: return -L * Delta / (y + Delta)
elif eta_function_used == 2:
    def eta_func_ask(y, Delta, L):
        if y == Delta: return L
        if y - Delta > 0: return L * Delta / (y - Delta)
        if y - Delta < 0: return -L * Delta / (y - Delta)

    def eta_func_bid(y, Delta, L):
        if y + Delta == 0: return L
        if y + Delta > 0: return L * Delta / (y + Delta)
        if y + Delta < 0: return -L * Delta / (y + Delta)
else:raise ETAFunctionError


# Plotting functions-------
def plot_simulation_results(sim_stat, number_of_rt):
    fig = plt.figure(figsize=(8,12))
    st = fig.suptitle("Simulation for LP initial inventory y={} and executing k={} round-trip trades".format(sim_stat['LP']['inv'][0],number_of_rt), fontsize="xx-large")
    import matplotlib.colors as mcolors
    lp_colours = mcolors.BASE_COLORS
    lt_colours = mcolors.TABLEAU_COLORS

    plt.subplot(3, 1, 1)
    x_axis = sim_stat['time']

    x_axis_ticks = np.linspace(sim_stat['time'][0],round(sim_stat['time'][-1]+1,-3), 11)
    plt.xticks(x_axis_ticks)

    plt.plot(x_axis, sim_stat['LT']['inv'], 'b', label="LT", linewidth=2.0)
    plt.plot(x_axis, sim_stat['LP']['inv'], '--{}'.format('r'), label="LP", linewidth=2.0)

    plt.ylabel("LP's and LT's Inventory")
    plt.xlabel("Time step")
    plt.grid()
    plt.xticks(x_axis_ticks)
    plt.legend(loc="best")

    plt.subplot(3, 1, 2)
    plt.plot(x_axis, sim_stat['LT']['w'], 'b', label="LT", linewidth=2.0)
    plt.plot(x_axis, sim_stat['LP']['w'], '--{}'.format('r'), label="LP", linewidth=2.0)
    plt.ylabel("LP's and LT's Cash Position")
    plt.xlabel("Time step")
    plt.grid()
    plt.xticks(x_axis_ticks)
    plt.legend(loc="best")

    plt.subplot(3, 1, 3)
    plt.plot(x_axis, sim_stat['z'], '--{}'.format('r'), label="Z", linewidth=1.0)
    plt.plot(x_axis, sim_stat['bid'], 'g', label=r"$Z-\delta^b(y)$", linewidth=1.0)
    plt.plot(x_axis, sim_stat['ask'], 'y', label=r"$Z+\delta^a(y)$", linewidth=1.0)
    plt.ylabel("Marginal Rate Z, Bid and Ask")
    plt.xlabel("Time step")
    plt.grid()
    plt.xticks(x_axis_ticks)
    plt.legend(loc="lower left")

    # plt.subplot(2, 2, 4)
    #
    # plt.ylabel("LP Bid and Ask")
    # plt.xlabel("Time step")
    # plt.grid()
    # plt.xticks(x_axis_ticks)
    # plt.legend(loc="lower left")
    plt.show()

def plot_new_Z_difference_round_trade(initial_price):
    pool_agent = get_pool_agent(arrival_rate, phi, alpha, fill_exponent,
                                initial_inventory_pool, target_inventory,
                                jump_size_L, unit_size, min_inventory_pool, max_inventory_pool,
                                initial_price, max_depth,
                                terminal_time, step_size,
                                seed, n_steps, num_trajectories, eta_func_bid, eta_func_ask,
                                verbose)
    fig = plt.figure(figsize=(8, 6))
    for LP_buy_then_sell in [True,False]:
        new_price_list = []
        for y in pool_agent.inventory_space:
            Z,current_inventory,current_wealth, sim_stats = evaluate_scenario_round_trade(LP_buy_then_sell=LP_buy_then_sell, agent_wealth = 1000, initial_price=initial_price, initial_inventory_pool=y,
                                          first_trade_time=0, second_trade_time=pool_agent.terminal_time, verbose=False,
                                          pool_agent=pool_agent)

            new_price_list.append(Z-initial_price)#np.divide(Z,initial_price))
        plt.plot(pool_agent.inventory_space, new_price_list, "-b" if LP_buy_then_sell else "-r", label="Bid-then-Ask" if LP_buy_then_sell else "Ask-then-Bid", linewidth=2.0)
    plt.grid()
    plt.ylabel(r"$\Delta$ in Z caused by round-trip trade")
    plt.xlabel("LP's inventory before round-trip execution.")

    plt.title(r"$\Delta$ in Z caused by different round-trip trades for Impact Set {} ".format(eta_function_used+1))
    plt.xticks(np.arange(pool_agent.inventory_space[0], pool_agent.inventory_space[-1]+1, 1.0))
    if eta_function_used == 2:
        plt.yticks(np.linspace(min(new_price_list), max(new_price_list),11))
    plt.legend(loc="upper left")
    plt.show()


def plot_bid_and_ask_spread(initial_price, min_inv, max_inv):
    pool_agent = get_pool_agent(arrival_rate, phi, alpha, fill_exponent,
                                initial_inventory_pool, target_inventory,
                                jump_size_L, unit_size, min_inv, max_inv,
                                initial_price, max_depth,
                                terminal_time, step_size,
                                seed, n_steps, num_trajectories, eta_func_bid, eta_func_ask,
                                verbose)

    plt.title("H_t = O(t,y)/g")
    plt.plot([y for y in pool_agent.inventory_space], pool_agent._calculate_ht(0.))
    plt.grid()
    plt.xticks(np.arange(pool_agent.inventory_space[0], pool_agent.inventory_space[-1], 1.0))
    plt.show()

    plt.title(r'Bid and Ask with $\eta^b(y)=\eta^a(y) = 0$')
    plt.xlabel('LP Inventory y')
    plt.plot([y for y in pool_agent.inventory_space[:-1]],
             (1/ pool_agent.kappa) -(pool_agent._calculate_ht(0.)[1:] - pool_agent._calculate_ht(0.)[:-1]),label=r'$\delta^b$')

    plt.plot([y for y in pool_agent.inventory_space[1:]],
             (1 / pool_agent.kappa) - (pool_agent._calculate_ht(0.)[:-1] - pool_agent._calculate_ht(0.)[1:]),
             label=r'$\delta^a$')
    plt.grid()
    plt.xticks(pool_agent.inventory_space)
    plt.legend(loc="lower center")
    plt.show()

    bid_list = []
    ask_list = []
    y_inv = []
    for y in pool_agent.inventory_space[2:-2]:
        delta_bid, delta_ask = pool_agent._calculate_deltas(current_time=0., inventories=y)[0]
        bid_list.append(delta_bid)
        ask_list.append(delta_ask)
        y_inv.append(y)

    plt.title("Bid/Spread Quoted at Current in y")
    plt.plot(y_inv, bid_list, "-b" , label="Bid", linewidth=2.0)
    plt.plot(y_inv, ask_list, "-r", label="Ask", linewidth=2.0)
    plt.plot(y_inv,[a+b for a,b in zip(bid_list,ask_list)], "-r", label="Sum", linewidth=2.0)
    plt.grid()
    plt.xticks(y_inv)
    plt.legend(loc="upper left")
    plt.show()


def populate_stats(simulation_statistics, Z, current_inventory,current_wealth, pool_agent):
    trade_counter = simulation_statistics['time'][-1] + 1
    simulation_statistics['LP']['inv'].append(current_inventory)
    simulation_statistics['LT']['w'].append(current_wealth)

    simulation_statistics['LP']['w'].append(simulation_statistics['LP']['w'][-1]+(simulation_statistics['LT']['w'][-2]-simulation_statistics['LT']['w'][-1]))
    simulation_statistics['LT']['inv'].append(simulation_statistics['LT']['inv'][-1] + simulation_statistics['LP']['inv'][-2] - simulation_statistics['LP']['inv'][-1])

    simulation_statistics['z'].append(Z)
    delta_bid, delta_ask = pool_agent._calculate_deltas(current_time=0., inventories=current_inventory)[0]
    simulation_statistics['bid'].append(Z-delta_bid)
    simulation_statistics['ask'].append(Z+delta_ask)
    simulation_statistics['time'].append(trade_counter)
    if  simulation_statistics['LP']['inv'][-1] >  simulation_statistics['LP']['inv'][-2]:
        simulation_statistics['trades'].append((simulation_statistics['bid'][-2],True))
    else:simulation_statistics['trades'].append((simulation_statistics['ask'][-2], False))
    return simulation_statistics

# End Plotting functions-------

def get_trade_results(LP_buys,pool_agent,price_Z, current_inventory,current_wealth, execution_time):
    delta_bid,delta_ask = pool_agent._calculate_deltas(current_time=execution_time, inventories=current_inventory)[0]

    impact_bid = pool_agent.env.midprice_model.eta_bid(current_inventory,pool_agent.env.trader.unit_size, pool_agent.env.midprice_model.jump_size_L)
    impact_ask = pool_agent.env.midprice_model.eta_ask(current_inventory,pool_agent.env.trader.unit_size, pool_agent.env.midprice_model.jump_size_L)

    if LP_buys:
        current_wealth += (price_Z - delta_bid) * pool_agent.env.trader.unit_size # LT current Wealth
        current_inventory += pool_agent.env.trader.unit_size # LP inventory
        Z = price_Z - impact_bid

    else:
        current_wealth -= (price_Z + delta_ask) * pool_agent.env.trader.unit_size
        current_inventory -= pool_agent.env.trader.unit_size
        Z = price_Z + impact_ask

    return Z, current_inventory, current_wealth


def evaluate_scenario_round_trade(LP_buy_then_sell,agent_wealth, initial_price,initial_inventory_pool, first_trade_time, second_trade_time, verbose =False,
                                  pool_agent=None, simulation_statistics= None):
    if pool_agent is None:
        pool_agent = get_pool_agent(arrival_rate, phi, alpha, fill_exponent,
                                    initial_inventory_pool, target_inventory,
                                    jump_size_L, unit_size, min_inventory_pool, max_inventory_pool,
                                    initial_price, max_depth,
                                    terminal_time, step_size,
                                    seed, n_steps, num_trajectories, eta_func_bid, eta_func_ask,
                                    verbose)

    if second_trade_time < first_trade_time or second_trade_time > terminal_time:
        raise IncorrectStepInTimeValue

    current_inventory = initial_inventory_pool
    current_wealth = agent_wealth
    Z = initial_price

    z_0 = Z
    y = current_inventory
    G = pool_agent.env.trader.unit_size
    L = pool_agent.env.midprice_model.jump_size_L

    # ensure_correct_calc mathematically
    if LP_buy_then_sell:
        if y>0:
            Z_math = z_0 + np.divide(L*G*G, y*(G+y))# np.divide(z_0*y*y + z_0*y*G + L*G*G , y*(G+y))
            lt_cash = agent_wealth + G*(np.divide(-2,pool_agent.kappa)+np.divide(G*L, y+G))
        elif y==0:
            if eta_function_used == 0:
                Z_math = z_0 - L
                lt_cash = agent_wealth+ G * (np.divide(-2, pool_agent.kappa) + 2*L)
            elif eta_function_used == 1:
                Z_math = z_0 - 2*L
                lt_cash = agent_wealth+ G * (np.divide(-2, pool_agent.kappa) + 2 * L)
            elif eta_function_used == 2:
                Z_math = z_0
                lt_cash = agent_wealth+ G * (np.divide(-2, pool_agent.kappa) + 2 * L)
        elif y == -G:
            if eta_function_used == 0:
                Z_math = z_0 + L
                lt_cash = agent_wealth+ G * (np.divide(-2,  pool_agent.kappa) +  L)
            elif eta_function_used == 1:
                Z_math = z_0 +2*L
                lt_cash = agent_wealth+ G * (np.divide(-2,  pool_agent.kappa))
            elif eta_function_used == 2:
                Z_math = z_0
                lt_cash = agent_wealth+ G * (np.divide(-2,  pool_agent.kappa) + 2 * L)
        elif y < 0 and y+G > 0:
            Z_math = np.divide(z_0*y*y + z_0*y*G - L*G*G - 2*G*L*y, y*(G+y))
        elif y<0 and y+G < 0:
            Z_math = z_0 - np.divide(L*G*G, y*(G+y))#np.divide(z_0*y*y + z_0*y*G - L*G*G, y * (G + y))
            lt_cash = agent_wealth+ G * (np.divide(-2, pool_agent.kappa) - np.divide(G * L, y + G))
        else:
            raise UnconsideredCase
    else:
        if y<0:
            Z_math = z_0 - np.divide(L*G*G, y*(y-G))#np.divide(z_0*y*y - z_0*y*G - L*G*G , y*(y-G))
            lt_cash = agent_wealth + G * (np.divide(-2, pool_agent.kappa) - np.divide(G * L, y - G))
        elif y==0:
            if eta_function_used == 0:
                Z_math = z_0 + L
                lt_cash = agent_wealth + G * (np.divide(-2, pool_agent.kappa) + 2 * L)
            elif eta_function_used == 1:
                Z_math = z_0 +2*L
                lt_cash = agent_wealth + G * (np.divide(-2, pool_agent.kappa) + 2 * L)
            elif eta_function_used == 2:
                Z_math = z_0
                lt_cash = agent_wealth + G * (np.divide(-2, pool_agent.kappa) + 2 * L)
        elif y == G:
            if eta_function_used == 0:
                Z_math = z_0 - L
                lt_cash = agent_wealth + G * (np.divide(-2, pool_agent.kappa) + L)
            elif eta_function_used == 1:
                Z_math = z_0 - 2*L
                lt_cash = agent_wealth + G * (np.divide(-2, pool_agent.kappa))
            elif eta_function_used == 2:
                Z_math = z_0
                lt_cash = agent_wealth + G * (np.divide(-2, pool_agent.kappa) + 2 * L)
        elif y > 0 and y-G < 0:
            Z_math = np.divide(z_0*y*y - z_0*y*G + L*G*G - 2*G*L*y, y*(y-G))
        elif y>0 and y-G > 0:
            Z_math = z_0 + np.divide(L*G*G, y*(y-G))#np.divide(z_0*y*y - z_0*y*G + L*G*G, y * (y-G))
            lt_cash = agent_wealth + G * (np.divide(-2, pool_agent.kappa) + np.divide(G * L, y - G))
        else:
            raise UnconsideredCase

    execution_time = 0. + first_trade_time
    Z,current_inventory,current_wealth = get_trade_results(LP_buy_then_sell,pool_agent,Z,current_inventory,current_wealth, execution_time=execution_time)
    if simulation_statistics is not None:
        simulation_statistics = populate_stats(simulation_statistics, Z, current_inventory,
                                           current_wealth,  pool_agent)

    execution_time = 0.+ second_trade_time
    Z,current_inventory,current_wealth = get_trade_results(not LP_buy_then_sell,pool_agent,Z,current_inventory,current_wealth,execution_time=execution_time)
    if simulation_statistics is not None:
        simulation_statistics = populate_stats(simulation_statistics, Z, current_inventory,
                                           current_wealth, pool_agent)
    if verbose:
        print('New Price Z: {} , Current Inv: {} , Current agent wealth: {}'.format(Z,current_inventory,current_wealth))


    if np.abs(np.round(Z-Z_math,10)) != 0:
        raise ZValueMismatch

    if first_trade_time == second_trade_time:
        accuracy = 10
    else:
        accuracy = 4
    if y > pool_agent.min_inventory + 2 and y < pool_agent.max_inventory - 2 and np.abs(np.round(current_wealth-lt_cash,accuracy)) != 0:
        raise LTCashMismatch

    return Z,current_inventory,current_wealth, simulation_statistics



def get_optimal_trading_times(agent_wealth, initial_price,initial_inventory_pool,total_number_of_steps, start_time_restriction = 0.):
    opt_buy_sell = None
    opt_sell_buy = None
    pool_agent = get_pool_agent(arrival_rate, phi, alpha, fill_exponent,
                                initial_inventory_pool, target_inventory,
                                jump_size_L, unit_size, min_inventory_pool, max_inventory_pool,
                                initial_price, max_depth,
                                terminal_time, step_size,
                                seed, n_steps, num_trajectories, eta_func_bid, eta_func_ask,
                                verbose)

    for buy_then_sell in [True,False]:
        max_wealth = - np.abs(agent_wealth)
        time_space =  np.linspace(start_time_restriction,pool_agent.terminal_time,total_number_of_steps)
        for i in range(0,len(time_space)-1):
            for j in range(i+1,len(time_space)):
                Z, current_inventory, current_wealth, sim_stats = evaluate_scenario_round_trade(buy_then_sell, agent_wealth,
                                                                                     initial_price,initial_inventory_pool,
                                                                                     first_trade_time=time_space[i],
                                                                                     second_trade_time=time_space[j],
                                                                                     verbose=False, pool_agent=pool_agent)
                # print(current_wealth)
                if current_wealth > max_wealth:
                    max_wealth=current_wealth
                    opt_params = (time_space[i],time_space[j])

        if buy_then_sell:
            print('Optimal timing LT maximize wealth for LP Buy then LP Sell.\n'
                  'LP Bid execution time step : {}, LP Ask execution time step : {},\n'
                  'LT wealth: {}'.format(opt_params[0],opt_params[1], max_wealth)  )
            opt_buy_sell = opt_params
        else:
            print('Optimal timing LT maximize wealth for LP Sell then LP Buy.\n'
                  'LP Ask execution time step : {}, LP Bid execution time step : {},\n'
                  'LT wealth: {}'.format(opt_params[0], opt_params[1], max_wealth))
            opt_sell_buy = opt_params
    return opt_buy_sell, opt_sell_buy
# opt = get_optimal_trading_times(agent_wealth=1000, initial_price=12.30,initial_inventory_pool=15,total_number_of_steps=100,start_time_restriction= 0)



def is_arbitrage_possible(initial_price):
    pool_agent = get_pool_agent(arrival_rate, phi, alpha, fill_exponent,
                                initial_inventory_pool, target_inventory,
                                jump_size_L, unit_size, min_inventory_pool, max_inventory_pool,
                                initial_price, max_depth,
                                terminal_time, step_size,
                                seed, n_steps, num_trajectories, eta_func_bid, eta_func_ask,
                                verbose)
    if eta_function_used == 0 or eta_function_used == 1:
        round_trip_pos_inv_target = 0
        round_trip_neg_inv_target = 0
        round_trip_into_positive_inv = False
    elif eta_function_used == 2:
        round_trip_pos_inv_target = 1
        round_trip_into_positive_inv = True  # LP buy then sell (blue)
        round_trip_neg_inv_target = -1
    else:
        raise ETAFunctionError

    # calc_number_of_round_trades_needed to cover deltas  not considering price drift
    inventory_space = pool_agent.inventory_space
    round_trip_pos_inv_target_index = np.where(pool_agent.inventory_space == round_trip_pos_inv_target)[0][0]
    round_trip_neg_inv_target_index = np.where(pool_agent.inventory_space == round_trip_neg_inv_target)[0][0]

    exec_cost_bid = np.array([np.round(pool_agent._calculate_deltas(current_time=0.,
                                                                    inventories=inv
                                                                    ), 3)[0][0]
                              for inv in inventory_space])

    exec_cost_ask = np.array([np.round(pool_agent._calculate_deltas(current_time=0.,
                                                                    inventories=inv
                                                                    ), 3)[0][1]
                              for inv in inventory_space])

    # array range -1 and 1 because at that inventory we will not ask or bid
    agents_profit_LP_neg_inv = - np.cumsum(
        (exec_cost_bid[:round_trip_neg_inv_target_index] + exec_cost_ask[1:round_trip_neg_inv_target_index + 1])[::-1])[::-1]
    corresponding_neg_inv = inventory_space[:round_trip_neg_inv_target_index]


    agents_profit_LP_pos_inv = -np.cumsum(
        (exec_cost_bid[round_trip_pos_inv_target_index: - 2] + exec_cost_ask[
                                                               round_trip_pos_inv_target_index + 1:-1]))  # bid posted for second to last is also very high penalty
    corresponding_pos_inv = inventory_space[1 + round_trip_pos_inv_target_index:-1]

    # Calculate difference in price of Z cause by n trades to one side and then the other
    if round_trip_pos_inv_target ==0:
        add_round_trip_to_zero_off_set = True
    else:
        add_round_trip_to_zero_off_set = False


    #pos inv case
    number_of_trades_in_one_dir = corresponding_pos_inv - max(1,round_trip_pos_inv_target)
    # New Z = Old Z + nLG^2 / init_inv(iniy-nG) up till inv 1 , -L to inv 0 and back
    pos_inv_LP_sell_and_eventual_LP_buy_price_off_set = np.divide(number_of_trades_in_one_dir*jump_size_L*(pool_agent.env.midprice_model.unit_size**2),corresponding_pos_inv*(corresponding_pos_inv-number_of_trades_in_one_dir*pool_agent.env.midprice_model.unit_size))

    # -ve inv case
    # New Z = Old Z - nLG^2 / init_inv(iniy+nG) up till inv 1 , +L to inv 0 and back
    number_of_trades_in_one_dir = abs(corresponding_neg_inv - min(-1, round_trip_neg_inv_target))
    neg_inv_LP_buy_and_eventual_LP_sell_price_off_set = -np.divide(
        number_of_trades_in_one_dir * jump_size_L * (pool_agent.env.midprice_model.unit_size ** 2),
        corresponding_neg_inv * (
                corresponding_neg_inv + number_of_trades_in_one_dir * pool_agent.env.midprice_model.unit_size))

    if add_round_trip_to_zero_off_set:
        if eta_function_used == 0:
            pos_inv_LP_sell_and_eventual_LP_buy_price_off_set -= jump_size_L
            neg_inv_LP_buy_and_eventual_LP_sell_price_off_set += jump_size_L
        elif eta_function_used == 1:
            pos_inv_LP_sell_and_eventual_LP_buy_price_off_set -= 2*jump_size_L
            neg_inv_LP_buy_and_eventual_LP_sell_price_off_set += 2*jump_size_L
        elif eta_function_used == 2:
            pos_inv_LP_sell_and_eventual_LP_buy_price_off_set -= 0
            neg_inv_LP_buy_and_eventual_LP_sell_price_off_set += 0

    neg_inv_LP_buy_and_eventual_LP_sell_price_off_set = agents_profit_LP_neg_inv + neg_inv_LP_buy_and_eventual_LP_sell_price_off_set
    pos_inv_LP_sell_and_eventual_LP_buy_price_off_set = agents_profit_LP_pos_inv + pos_inv_LP_sell_and_eventual_LP_buy_price_off_set

    # for i in range(0,len(corresponding_neg_inv)):
    #     inc = int(corresponding_neg_inv[i])
    #     calc_ = neg_inv_LP_buy_and_eventual_LP_sell_price_off_set[i]
    #
    #     sim = test_n_round_trades_vs_n_trades_one_way_n_trades_other(buy_then_sell=True, agent_wealth=1000, initial_price=10,
    #                                                        initial_inventory_pool=inc, n_trades=abs(inc)-1)
    #     if round(sim - (calc_+initial_price),10) != 0 :
    #         print('error')

    # Evaluate round trip price and wealth effects
    # Case 0,1 - Ask then Bid - ie LP decrease and increase inv - red
    # Case 2 Bid then Ask - ie LP increase and decrease inv - blue

    # obtain round trip change in marginal rate and costs

    pos_inv_new_Z, pos_inv_current_inventory, pos_inv_current_wealth, _ = evaluate_scenario_round_trade(
        LP_buy_then_sell=round_trip_into_positive_inv, agent_wealth=0,
        initial_price=1,
        initial_inventory_pool=round_trip_pos_inv_target, first_trade_time=0,
        second_trade_time=0. + pool_agent.env.midprice_model.step_size, verbose=False, pool_agent=pool_agent)
    # pos inv we want Z to increase so LT sell back at a higher price
    change_in_marginal_rate = pos_inv_new_Z - 1
    cost_from_trade = 0 - pos_inv_current_wealth # note 0- to get cost

    pos_inv_number_of_round_trades_required = np.divide(-pos_inv_LP_sell_and_eventual_LP_buy_price_off_set,
                                                (corresponding_pos_inv- round_trip_pos_inv_target) * change_in_marginal_rate - cost_from_trade)

    # Case 0,1 - Bid then Ask - ie LP increase and decrease inv - blue
    # Case 2 - Ask then Bid - ie LP decrease and increase inv - red
    neg_inv_new_Z, neg_inv_current_inventory, neg_inv_current_wealth, _ = evaluate_scenario_round_trade(
        LP_buy_then_sell=not round_trip_into_positive_inv, agent_wealth=0,
        initial_price=1,
        initial_inventory_pool=round_trip_neg_inv_target, first_trade_time=0,
        second_trade_time=0. + pool_agent.env.midprice_model.step_size, verbose=False, pool_agent=pool_agent)

    change_in_marginal_rate =  1 - neg_inv_new_Z
    cost_from_trade = 0 - neg_inv_current_wealth  # note 0- to get cost

    neg_inv_number_of_round_trades_required = np.divide(-neg_inv_LP_buy_and_eventual_LP_sell_price_off_set,
                                                            abs(corresponding_neg_inv - round_trip_neg_inv_target) * change_in_marginal_rate - cost_from_trade)


    # neg_inv_Z_cutoff
    if round_trip_neg_inv_target == -1:
        z_offset_at_target_y = np.cumsum(np.divide(unit_size * jump_size_L, corresponding_neg_inv + unit_size)[::-1])[
                               ::-1]
    elif round_trip_neg_inv_target == 0:
        # impact on Z from inv 0 to 1 is Zero, but y+G = 0 will result in div by zero. Thus, remove from calc and append value after
        z_offset_at_target_y = np.cumsum(
            np.divide(unit_size * jump_size_L, corresponding_neg_inv[:-1] + unit_size)[::-1])[
                               ::-1]
        z_offset_at_target_y= np.append(z_offset_at_target_y,0)
    else:
        raise ETAFunctionError
    number_of_possible_round_trades = np.divide(initial_price + z_offset_at_target_y,  change_in_marginal_rate)

    plt.plot(corresponding_neg_inv, np.clip(neg_inv_number_of_round_trades_required, -2000,10000), "-y", label=r"y_0 $\leq y_{ni}$", linewidth=2.0)
    plt.plot(corresponding_pos_inv, np.clip(pos_inv_number_of_round_trades_required, -2000,10000), "-c", label=r"y_0 $\geq y_{pi}$",
             linewidth=2.0)

    plt.plot(corresponding_neg_inv,number_of_possible_round_trades,
             "--g", label=r"Max k for y_0 $\leq y_{ni}$",
             linewidth=2.0)

    plt.title(r'Min k to be performed at $y_{ni}$ or $y_{pi}$ for LP initial inv $y_0$')
    plt.xlabel(r'LP initial inv $y_0$')
    plt.ylabel('Minimum number of round-trip trades k')
    plt.grid()
    plt.legend(loc="upper left")
    plt.show()

    return corresponding_neg_inv , corresponding_pos_inv, neg_inv_number_of_round_trades_required , pos_inv_number_of_round_trades_required, number_of_possible_round_trades


def arbitrage_strategy(agent_wealth,agent_inv, initial_inventory_pool, initial_price,number_of_round_trades, verbose=False):
    pool_agent = get_pool_agent(arrival_rate, phi, alpha, fill_exponent,
                                initial_inventory_pool, target_inventory,
                                jump_size_L, unit_size, min_inventory_pool, max_inventory_pool,
                                initial_price, max_depth,
                                terminal_time, step_size,
                                seed, n_steps, num_trajectories, eta_func_bid, eta_func_ask,
                                verbose)

    current_inventory = initial_inventory_pool
    current_wealth = agent_wealth
    agent_current_inventory = agent_inv
    Z = initial_price
    delta_bid, delta_ask = pool_agent._calculate_deltas(current_time=0., inventories=current_inventory)[0]
    simulation_statistics = {'LP': {'inv': [current_inventory], 'w': [0]},
                             'LT': {'inv': [], 'w': [current_wealth]},
                             'z': [Z], 'bid': [Z-delta_bid], 'ask': [Z+delta_ask],
                             'time': [0],
                             'trades': []}
    trade_counter = 0


    # agent has money - will buy up inventory of pool , push prices up , sell

    # Note if eta fn used is 0 or 1 (paper) we move to zero and then select either Blue or Red depending if we want price to go up or down
    # Eta = 2 is different. If we want to increase price we either move inv to 1 and do (bid-ask) or 2 and don(ask-bid). To decrease price
    # we move inv to -2 (bid-ask) or -1 (ask-bid) Note this strategy also works for eta=0 or 1 but slower
    if eta_function_used == 0 or eta_function_used == 1:
        # move inv to zero
        if initial_inventory_pool > 0 : #LT buys up all inv, then push price up by trading round(-1,0), then sell
            LP_buys= False
            round_trip_into_positive_inv = False
            agent_inv_increase = 1
            simulation_statistics['LT']['inv'].append(0)
        elif initial_inventory_pool<0:  #LT sell so LP inv = 0, then push price down by trading round(0,1), then buy
            LP_buys= True
            round_trip_into_positive_inv = True
            agent_inv_increase = -1
            simulation_statistics['LT']['inv'].append(-initial_inventory_pool)
        else:
            raise UnconsideredCase

        print('Initial Setup Z: {} , LP Inv: {} , LT wealth: {}, LT inv: {}'.format(Z, current_inventory,
                                                                                                 current_wealth,
                                                                                                 agent_current_inventory))

        while current_inventory != 0:
            Z, current_inventory, current_wealth = get_trade_results(LP_buys=LP_buys, pool_agent=pool_agent, price_Z=Z, current_inventory=current_inventory,
                                                                     current_wealth=current_wealth, execution_time=0)
            agent_current_inventory += agent_inv_increase
            if verbose:
                print('Trade Executed-> New Price Z: {} , LP Inv: {} , LT wealth: {}, LT inv: {}'.format(Z, current_inventory, current_wealth,agent_current_inventory))
            trade_counter+=1
            simulation_statistics = populate_stats(simulation_statistics, Z, current_inventory,current_wealth, pool_agent)

        print('Round Trade arbitrage starting Z: {} , LP Inv: {} , LT wealth: {}, LT inv: {}'.format(Z, current_inventory,
                                                                                    current_wealth,
                                                                                    agent_current_inventory))
        for i in range(0,number_of_round_trades):
            Z, current_inventory, current_wealth, simulation_statistics = evaluate_scenario_round_trade(LP_buy_then_sell=round_trip_into_positive_inv, agent_wealth=current_wealth, initial_price=Z,
                                                                                 initial_inventory_pool=current_inventory, first_trade_time=0.,
                                                                                 second_trade_time=0.+pool_agent.env.step_size, verbose=verbose, pool_agent=pool_agent, simulation_statistics = simulation_statistics)

        print(
            'Round Trade arbitrage Finished Z: {} , LP Inv: {} , LT wealth: {}, LT inv: {}'.format(Z, current_inventory,
                                                                                                   current_wealth,
                                                                                                   agent_current_inventory))


        while agent_current_inventory != agent_inv:
            Z, current_inventory, current_wealth = get_trade_results(LP_buys=not LP_buys, pool_agent=pool_agent, price_Z=Z,
                                                                     current_inventory=current_inventory,
                                                                     current_wealth=current_wealth, execution_time=0)
            agent_current_inventory -= agent_inv_increase
            if verbose:
                print('Midway Price Z: {} , Inv: {} , agent wealth: {}, agent_inv: {}'.format(Z, current_inventory, current_wealth,agent_current_inventory))
            trade_counter += 1
            simulation_statistics = populate_stats(simulation_statistics, Z, current_inventory,current_wealth, pool_agent)

        plot_simulation_results(simulation_statistics, number_of_round_trades)

        print('Final Values Price Z: {} , Inv: {} , agent wealth: {}, agent_inv: {}\n\n'.format(Z, current_inventory, current_wealth,agent_current_inventory))
    elif eta_function_used ==2:
        round_trip_pos_inv_target = 1
        round_trip_neg_inv_target = -1
        if initial_inventory_pool > round_trip_pos_inv_target:  # LT buys up all inv, then push price up by trading round(1,2), then sell
            LP_buys = False
            round_trip_into_positive_inv = True #LP buy then sell (blue)
            agent_inv_increase = 1
        elif initial_inventory_pool < round_trip_neg_inv_target:  # LT sell so LP inv = 0, then push price down by trading round(-1,-2), then buy
            LP_buys = True
            round_trip_into_positive_inv = False #LP sell then buy (red)
            agent_inv_increase = -1
        else:
            raise UnconsideredCase

        print('Initial Setup Z: {} , LP Inv: {} , LT wealth: {}, LT inv: {}'.format(Z, current_inventory,
                                                                                    current_wealth,
                                                                                    agent_current_inventory))
        while current_inventory != -1 and current_inventory != 1:
            Z, current_inventory, current_wealth = get_trade_results(LP_buys=LP_buys, pool_agent=pool_agent, price_Z=Z,
                                                                     current_inventory=current_inventory,
                                                                     current_wealth=current_wealth, execution_time=0)
            agent_current_inventory += agent_inv_increase
            if verbose: print(
                'Trade Executed-> New Price Z: {} , LP Inv: {} , LT wealth: {}, LT inv: {}'.format(Z, current_inventory,
                                                                                                   current_wealth,
                                                                                                   agent_current_inventory))

        print(
            'Round Trade arbitrage starting Z: {} , LP Inv: {} , LT wealth: {}, LT inv: {}'.format(Z, current_inventory,
                                                                                                   current_wealth,
                                                                                                   agent_current_inventory))
        for i in range(0, number_of_round_trades):
            Z, current_inventory, current_wealth = evaluate_scenario_round_trade(
                LP_buy_then_sell=round_trip_into_positive_inv, agent_wealth=current_wealth, initial_price=Z,
                initial_inventory_pool=current_inventory, first_trade_time=0.,
                second_trade_time=0.+pool_agent.env.step_size, verbose=verbose, pool_agent=pool_agent)

        print(
            'Round Trade arbitrage Finished Z: {} , LP Inv: {} , LT wealth: {}, LT inv: {}'.format(Z, current_inventory,
                                                                                                   current_wealth,
                                                                                                   agent_current_inventory))
        while agent_current_inventory != agent_inv:
            Z, current_inventory, current_wealth = get_trade_results(LP_buys=not LP_buys, pool_agent=pool_agent,
                                                                     price_Z=Z,
                                                                     current_inventory=current_inventory,
                                                                     current_wealth=current_wealth,
                                                                     execution_time=0)
            agent_current_inventory -= agent_inv_increase
            if verbose:print('Midway Price Z: {} , Inv: {} , agent wealth: {}, agent_inv: {}'.format(Z, current_inventory,
                                                                                          current_wealth,
                                                                                          agent_current_inventory))
        print('Final Values Price Z: {} , Inv: {} , agent wealth: {}, agent_inv: {}\n\n'.format(Z, current_inventory,
                                                                                            current_wealth,
                                                                                            agent_current_inventory))
    else:
        raise ETAFunctionError


#Test 1 and 2
# Test 1
# def test_stationary_n_round_trades_vs_n_trades_one_way_n_trades_other(buy_then_sell, agent_wealth, initial_price, initial_inventory_pool, n_trades):
#     pool_agent_rt = get_pool_agent(arrival_rate, phi, alpha, fill_exponent,
#                                 initial_inventory_pool, target_inventory,
#                                 jump_size_L, unit_size, min_inventory_pool, max_inventory_pool,
#                                 initial_price, max_depth,
#                                 terminal_time, step_size,
#                                 seed, n_steps, num_trajectories, eta_func_bid, eta_func_ask,
#                                 verbose)
#
#     pool_agent_n_one_way = get_pool_agent(arrival_rate, phi, alpha, fill_exponent,
#                                    initial_inventory_pool, target_inventory,
#                                    jump_size_L, unit_size, min_inventory_pool, max_inventory_pool,
#                                    initial_price, max_depth,
#                                    terminal_time, step_size,
#                                    seed, n_steps, num_trajectories, eta_func_bid, eta_func_ask,
#                                    verbose)
#
#     current_inventory = initial_inventory_pool
#     current_wealth = agent_wealth
#     Z = initial_price
#
#     execution_times = np.linspace(0,pool_agent_rt.terminal_time,2*n_trades)
#     for i ,exec_time in zip(range(0,n_trades), execution_times[0:n_trades]):
#         Z, current_inventory, current_wealth = get_trade_results(buy_then_sell, pool_agent_rt, Z, current_inventory,
#                                                                  current_wealth, execution_time=exec_time)
#     print('Midway Price Z: {} , Inv: {} , agent wealth: {}'.format(Z, current_inventory, current_wealth))
#
#     for i, exec_time in zip(range(0,n_trades),execution_times[n_trades:]):
#         Z, current_inventory, current_wealth = get_trade_results(not buy_then_sell, pool_agent_rt, Z, current_inventory,
#                                                                  current_wealth, execution_time=exec_time)
#     print('Final Price Z: {} , Inv: {} , agent wealth: {}'.format(Z, current_inventory, current_wealth))
#
#     print('\n\n')
#
#     #testing n round trades
#     current_inventory = initial_inventory_pool
#     current_wealth = agent_wealth
#     Z = initial_price
#     for i in range(0,len(execution_times)-1,2):
#         Z, current_inventory, current_wealth,_ = evaluate_scenario_round_trade(buy_then_sell, current_wealth, Z, current_inventory, execution_times[i],
#                                                                              execution_times[i+1], verbose=verbose, pool_agent=pool_agent_n_one_way)
#
#     print('Final Price Z: {} , Inv: {} , agent wealth: {}\n\n'.format(Z, current_inventory, current_wealth))
#     return Z
# test_stationary_n_round_trades_vs_n_trades_one_way_n_trades_other(buy_then_sell=True, agent_wealth=1000, initial_price=10, initial_inventory_pool=-800, n_trades=800)

#Test 2
# def cost_analysis_for_n_shifting_round_trades_vs_n_onw_way_then_n_other_way(buy_then_sell, agent_wealth, initial_price, initial_inventory_pool, n_trades):
#     pool_agent_rt = get_pool_agent(arrival_rate, phi, alpha, fill_exponent,
#                                 initial_inventory_pool, target_inventory,
#                                 jump_size_L, unit_size, min_inventory_pool, max_inventory_pool,
#                                 initial_price, max_depth,
#                                 terminal_time, step_size,
#                                 seed, n_steps, num_trajectories, eta_func_bid, eta_func_ask,
#                                 verbose)
#
#     pool_agent_n_one_way = get_pool_agent(arrival_rate, phi, alpha, fill_exponent,
#                                    initial_inventory_pool, target_inventory,
#                                    jump_size_L, unit_size, min_inventory_pool, max_inventory_pool,
#                                    initial_price, max_depth,
#                                    terminal_time, step_size,
#                                    seed, n_steps, num_trajectories, eta_func_bid, eta_func_ask,
#                                    verbose)
#
#     original_cash = agent_wealth
#     current_inventory = initial_inventory_pool
#     current_wealth = agent_wealth
#     Z = initial_price
#
#     shifting_inv_for_round_trade = []
#
#     execution_times = np.linspace(0,pool_agent_rt.terminal_time,2*n_trades)
#     for i ,exec_time in zip(range(0,n_trades), execution_times[0:n_trades]):
#         shifting_inv_for_round_trade.append(current_inventory)
#         Z, current_inventory, current_wealth = get_trade_results(buy_then_sell, pool_agent_rt, Z, current_inventory,
#                                                                  current_wealth, execution_time=0.)
#     print('Midway Price Z: {} , Inv: {} , agent wealth: {}'.format(Z, current_inventory, current_wealth))
#
#     for i, exec_time in zip(range(0,n_trades),execution_times[n_trades:]):
#         Z, current_inventory, current_wealth = get_trade_results(not buy_then_sell, pool_agent_rt, Z, current_inventory,
#                                                                  current_wealth, execution_time=0.)
#     print('Final Price Z: {} , Inv: {} , agent wealth: {}'.format(Z, current_inventory, current_wealth))
#
#     print('Agent Original Cash = {}, Agent Final Cash = {}, Cash Change = {}'.format(original_cash,current_wealth, original_cash - current_wealth))
#     print('\n\n')
#
#     #testing n round trades
#     wealth_change = 0
#     marginal_rate_change = 0
#     Z = initial_price
#     for i, shifting_inv in zip(range(0,len(execution_times)-1,2),shifting_inv_for_round_trade):
#         Z, _, current_wealth, _ = evaluate_scenario_round_trade(buy_then_sell, agent_wealth, initial_price, shifting_inv, 0.,
#                                                                              0., verbose=verbose, pool_agent=pool_agent_n_one_way)
#         wealth_change += (current_wealth - agent_wealth)
#         marginal_rate_change +=  Z - initial_price
#
#     print('Agent Original Cash = {}, Agent Final Cash = {}, Cash Change = {}'.format(original_cash,original_cash + wealth_change ,wealth_change))
#     print('Final Price Z: {}'.format(initial_price + marginal_rate_change))
#     return
# cost_analysis_for_n_shifting_round_trades_vs_n_onw_way_then_n_other_way(buy_then_sell=True, agent_wealth=1000, initial_price=10, initial_inventory_pool=-805, n_trades=805)
#


# plot_bid_and_ask_spread(10., -10,10)
# #change min max pool inv for plot
# plot_new_Z_difference_round_trade(initial_price)
#


corresponding_neg_inv , corresponding_pos_inv, neg_inv_number_of_round_trades_required ,\
pos_inv_number_of_round_trades_required, neg_inv_number_of_possible_round_trades = is_arbitrage_possible(initial_price=12.30)

# neg pricing example
i = 1396
print('First example might not hit arbitrage since price cannot fall below zero')
arbitrage_strategy(agent_wealth=100000,agent_inv=0, initial_inventory_pool=corresponding_neg_inv[i],
                   initial_price=12.30, number_of_round_trades = math.floor(neg_inv_number_of_round_trades_required[i])+1)

print(neg_inv_number_of_round_trades_required[i])
arbitrage_strategy(agent_wealth=0,agent_inv=0, initial_inventory_pool=corresponding_neg_inv[i],
                   initial_price=12.30, number_of_round_trades = 0)


arbitrage_strategy(agent_wealth=0,agent_inv=0, initial_inventory_pool=corresponding_neg_inv[i],
                   initial_price=12.30, number_of_round_trades = math.floor(neg_inv_number_of_round_trades_required[i])+1)

arbitrage_strategy(agent_wealth=0,agent_inv=0, initial_inventory_pool=corresponding_neg_inv[i],
                   initial_price=12.30, number_of_round_trades = math.floor(neg_inv_number_of_round_trades_required[i])+400)




i = 350
arbitrage_strategy(agent_wealth=100000,agent_inv=0, initial_inventory_pool=corresponding_pos_inv[i],
                   initial_price=100, number_of_round_trades = math.floor(pos_inv_number_of_round_trades_required[i])+1)
#







# # EQNS for Z 1 Z 2 sec 2.4 adn 2.53
# def ensure_z1_z2_prices_are_correct(buy_then_sell, agent_wealth, initial_price, initial_inventory_pool, n_trades):
#     pool_agent_n_one_way = get_pool_agent(arrival_rate, phi, alpha, fill_exponent,
#                                    initial_inventory_pool, target_inventory,
#                                    jump_size_L, unit_size, min_inventory_pool, max_inventory_pool,
#                                    initial_price, max_depth,
#                                    terminal_time, step_size,
#                                    seed, n_steps, num_trajectories, eta_func_bid, eta_func_ask,
#                                    verbose)
#
#     current_inventory = initial_inventory_pool
#     current_wealth = agent_wealth
#     Z = initial_price
#
#     inv_tracker_y_i = [current_inventory]
#     inv_tracker_y_i_plus_1 = []
#     z_prices_bids = []
#     z_prices_ask = []
#
#     marg_rate_final = []
#     execution_times = np.linspace(0,pool_agent_n_one_way.terminal_time,2*n_trades)
#     z_prices_bids.append(Z)
#     for i ,exec_time in zip(range(0,n_trades), execution_times[0:n_trades]):
#         Z, current_inventory, current_wealth = get_trade_results(buy_then_sell, pool_agent_n_one_way, Z, current_inventory,
#                                                                  current_wealth, execution_time=0.)
#         z_prices_bids.append(Z)
#         inv_tracker_y_i.append(current_inventory)
#         inv_tracker_y_i_plus_1.append(current_inventory)
#
#
#     z_prices_ask.append(z_prices_bids[-1]) #The last Z is the one on which the first as is calculated
#     z_prices_bids = z_prices_bids[:-1]
#     inv_tracker_y_i = inv_tracker_y_i[:-1]
#     print('Midway Price Z: {} , Inv: {} , agent wealth: {}'.format(Z, current_inventory, current_wealth))
#
#     for i, exec_time in zip(range(0,n_trades),execution_times[n_trades:]):
#         Z, current_inventory, current_wealth = get_trade_results(not buy_then_sell, pool_agent_n_one_way, Z, current_inventory,
#                                                                  current_wealth, execution_time=0.)
#         z_prices_ask.append(Z)
#         marg_rate_final.append(Z)
#     print('Final Price Z: {} , Inv: {} , agent wealth: {}'.format(Z, current_inventory, current_wealth))
#
#     marg_rate_final = marg_rate_final[::-1]
#
#     z_prices_ask = z_prices_ask[:-1]
#     reversed_z_prices_ask = z_prices_ask[::-1]
#
#     #Eqns 2.43 and 2.44
#     for i in range(0, len(z_prices_bids)):
#         n  = n_trades - i
#         z_og = z_prices_bids[i]
#         if eta_function_used == 2:
#             new_marg_rate_after_n_trades = z_og - np.divide(n*jump_size_L*unit_size*unit_size, inv_tracker_y_i[i]*(inv_tracker_y_i[i]+n*unit_size))
#         else:
#             new_marg_rate_after_n_trades = z_og - np.divide((n-1) * jump_size_L * unit_size * unit_size,
#                                                             inv_tracker_y_i[i] * (inv_tracker_y_i[i] + (n-1) * unit_size))
#             if eta_function_used == 0:
#                 new_marg_rate_after_n_trades += jump_size_L
#             elif eta_function_used == 1:
#                 new_marg_rate_after_n_trades += 2*jump_size_L
#         z_targ = marg_rate_final[i]
#
#         if round(new_marg_rate_after_n_trades-z_targ,10) != 0:
#             print('Not Matching')
#
#
#     # Eqns 2.46 2.47
#     y_ni = 0
#     if eta_function_used == 2:
#         y_ni = -1
#
#     profit_list = []
#     for i in range (0,len(z_prices_bids)):
#         y_i = inv_tracker_y_i[i]
#         z_1 = z_prices_bids[i]
#         z_2 = reversed_z_prices_ask[i]
#         ind = i+1
#
#         if eta_function_used == 2:
#             z_offset = - np.divide((n_trades - ind) * jump_size_L,(y_i+1)*y_ni)
#             computed_z_2 = z_1 - eta_func_bid(inv_tracker_y_i[i],unit_size,jump_size_L)  + z_offset
#         else:
#             if ind < len(z_prices_bids):
#                 z_offset = -np.divide((n_trades - ind - 1) * jump_size_L, (y_i + 1) * (y_ni - unit_size))
#                 if eta_function_used == 0:
#                     z_offset += jump_size_L
#                 elif eta_function_used == 1:
#                     z_offset += jump_size_L * 2
#             else:
#                 z_offset = 0
#
#             computed_z_2 = z_1 - eta_func_bid(inv_tracker_y_i[i],unit_size,jump_size_L) +z_offset
#
#         profit_list.append(-z_offset + np.divide(-2,pool_agent_n_one_way.kappa) + 2*eta_func_bid(inv_tracker_y_i[i],unit_size,jump_size_L) +
#                            inv_tracker_y_i[i]*(eta_func_bid(inv_tracker_y_i[i],unit_size,jump_size_L) - eta_func_ask(inv_tracker_y_i_plus_1[i],unit_size,jump_size_L)))
#         if round(z_2-computed_z_2,10) == 0:
#             print('match')
#         else:
#             print('error')
#
#     print('donf')
#
#
#
# n_trades = 799
# if eta_function_used == 0 or eta_function_used == 1:
#     n_trades= 800
# ensure_z1_z2_prices_are_correct(buy_then_sell=True, agent_wealth=1000, initial_price=10, initial_inventory_pool=-800, n_trades=n_trades)
