import numpy as np
import pandas as pd
import pprint
from matplotlib import pyplot as plt

#basic functions
norm = np.linalg.norm
sqrt = np.sqrt
exp = np.exp
ln = np.log
inv = np.linalg.inv
matmul = np.dot

#dataframe
def load_data():
    df = pd.read_csv("./data/historical_yields.csv")
    df['rate'] = df['rate']/100.0
    return df

#Parameters
def get_params(df):
    mu = np.mean(df['rate'])
    sig = np.std(df['rate'])
    x = df['rate'] - mu
    phi = np.correlate(x[1:], x[:-1]) / (norm(x[1:]) * norm(x[:-1]))
    phi = phi[0]

    mu = 0.048
    sig = 0.033
    phi = 0.95
    lambd = 0.010
    h = 1.0
    r = 0.0269
    step = calculate_step(sig, phi, h)
    step = 0.011

    params = {
            'mu':mu,
            'sig':sig,
            'phi': phi,
            'h': h,
            'r': r,
            'lambda': lambd,
            'step': step
            }
    return params

def calculate_step(sig, phi, h):
    step = sig*sqrt(-2 * ln(phi))*sqrt(h)
    return step

def calculate_up_prob(r, params):
    mu = params['mu']
    sig = params['sig']
    phi = params['phi']
    h = params['h']
    q_vt = 0.5 
    q_vt += (mu - r)*sqrt(h)*sqrt(-ln(phi))  / (sig*sqrt(8))
    return q_vt

#start = have  all values before start year
#T = years you have to calculate this for
#if start == 0, need to have the rate in the vasicek dict already
def get_future_rates_1y(params,
        vasicek = {},
        start = 0, 
        T=8):

    if start == T:
        return vasicek

    step = params['step']
    if start == 0:
        vasicek[(0,1)] = [{'rate':params['r']}]
        cur_rate = vasicek[(0,1)][0]['rate'] #must be initiliazed
        discount = exp(-cur_rate)
        up_prob = calculate_up_prob(cur_rate, params)
        down_prob = 1 - up_prob
        vasicek[(0,1)][0]['discount'] = discount
        vasicek[(0,1)][0]['up_prob'] = up_prob
        vasicek[(0,1)][0]['down_prob'] = down_prob
        vasicek = get_future_rates_1y(params, vasicek, start+1, T)
        return vasicek

    for i in range(0, start ):
        old_rate = vasicek[start-1,start][i]['rate']
        new_rate = old_rate + step
        discount = exp(-new_rate)
        up_prob = calculate_up_prob(new_rate, params)
        down_prob = 1 - up_prob

        if (start, start+1) not in vasicek:
            vasicek[(start, start+1)] = []
        vasicek[(start,start+1)].append({
                    'rate':new_rate,
                    'discount': discount,
                    'up_prob':up_prob,
                    'down_prob':down_prob
                    })

    rate = old_rate - step
    discount = exp(-rate)
    up_prob = calculate_up_prob(rate, params)
    down_prob = 1 - up_prob
    vasicek[(start, start + 1)].append({
            'rate': old_rate - step,
            'discount': discount,
            'up_prob': up_prob,
            'down_prob': down_prob
            }) 
    vasicek = get_future_rates_1y(params, vasicek, start+1, T)

    return vasicek


def calculate_risk_penalty(rate_up, rate_down, params):
    mu = params['mu']
    sig = params['sig']
    phi = params['phi']
    lambd = params['lambda']
    h = params['h']

    rp = 0.5*lambd*sqrt(h)

    tmp = exp(-rate_up*h) - exp(-rate_down*h)

    rp *= tmp
    return rp


#go reverse
def get_future_rates_2y(params, vasicek, cur):
    vasicek[cur, cur+2] = []
    for i in range(0, cur+1):
        rate = vasicek[cur, cur+1][i]['rate']
        rate_up = vasicek[cur+1,cur+2][i]['rate']
        rate_down = vasicek[cur+1,cur+2][i+1]['rate']
        rp = calculate_risk_penalty(rate_up, rate_down, params)

        up_prob = vasicek[cur,cur+1][i]['up_prob']
        down_prob = vasicek[cur,cur+1][i]['down_prob']

        discount_2h = exp(-rate_up)*up_prob 
        discount_2h += exp(-rate_down)*down_prob
        discount_2h -= rp
        discount_2h *= exp(-rate)
        
        rate_2h = -ln(discount_2h) / (2.0*params['h'])

        vasicek[cur, cur+2].append({
                    'rate':rate_2h,
                    'discount': discount_2h
                    })
    if cur > 0:
        vasicek = get_future_rates_2y(params, vasicek, cur - 1)
    return vasicek

def get_future_rates_long(params, vasicek, cur, T):
    h_range = range(3, T-cur+1)

    for h in h_range:
        vasicek[cur, cur+h] = []
        for i in range(0, cur+1):
            d1 = vasicek[cur,cur+h-2][i]['discount']
            d2 = vasicek[cur,cur+h-1][i]['discount']

            if h == 3:
                dup0 = 1.0
                ddn0 = 1.0
            else:
                dup0 =  vasicek[cur+1,cur+h-2][i]['discount']
                ddn0 =  vasicek[cur+1,cur+h-2][i+1]['discount']

            dup1 = vasicek[cur+1,cur+h-1][i]['discount']
            ddn1 = vasicek[cur+1,cur+h-1][i+1]['discount']

            dup2 = vasicek[cur+1,cur+h][i]['discount']
            ddn2 = vasicek[cur+1,cur+h][i+1]['discount']

            A = np.matrix([[dup0,dup1], [ddn0,ddn1]])
            b = np.matrix([[dup2], [ddn2]])

            x = matmul(inv(A), b)

            n1 = float(x[0][0])
            n2 = float(x[1][0])

            discount_cur_h = n1*d1+n2*d2
            rate_cur_h = -ln(discount_cur_h)/h

            vasicek[cur,cur+h].append({
                                'rate':rate_cur_h,
                                'discount':discount_cur_h,
                                })
    if cur > 0:
        vasicek = get_future_rates_long(params, vasicek, cur-1, T)

    return vasicek

def plot(vasicek, T):
    Y_act = np.array([2.690,2.810,2.830,2.840,2.850,2.895,2.940,2.9700,3.0000,3.030])/100.

    Y_act = Y_act[0:T]
    X = range(1, T+1)
    Y = []
    for i in range(1, T+1):
        Y.append(vasicek[0,i][0]['rate'])

    plt.plot(X, Y_act, '-')
    plt.plot(X,Y, '-')
    # plt.show()
    return X, Y, Y_act


def main(T=8, params = None):
    df = load_data()
    if params == None:
        params = get_params(df)
    else:
        params['step'] = calculate_step(params['sig'], params['phi'],params['h'])

    vasicek = get_future_rates_1y(params, {}, 0, T)
    vasicek = get_future_rates_2y(params, vasicek, T-2)
    vasicek = get_future_rates_long(params, vasicek, T-3, T)
    term_struct = plot(vasicek, T)
    return df, params, vasicek, term_struct

def set_params(sig, phi, lambd, mu = 0.0269):
    params = {
         'mu':mu,
         'sig':sig,
         'phi':phi,
         'lambda':lambd,
         'r':0.0269,
         'step': calculate_step(sig, phi, 1.0),
         'h':1.0
         }
    return params

def option_pricing(vasicek, cur, T, strike_price = 1.02):
    for i in range(0, cur+1):
        max_val = max(vasicek[cur, cur+1][i]['discount'] - strike_price, 0 ) 
        if cur == T-1:
            call = 0
        else:
            d1 = vasicek[cur, cur+1][i]['discount']
            d2 = vasicek[cur, cur+2][i]['discount']

            a1 = vasicek[cur+1, cur+2][i]['discount']
            a2 = vasicek[cur+1, cur+2][i+1]['discount']

            b1 = vasicek[cur+1, T][i]['call']
            b2 = vasicek[cur+1, T][i+1]['call']

            A = np.matrix([[1, a1],[1, a2]])
            b = np.matrix([[b1],[b2]])

            x = np.dot(inv(A), b)
            n1 = float(x[0])
            n2 = float(x[1])

            call = n1*d1 + n2*d2
        
        vasicek[cur, T][i]['call'] = call + max_val
    if cur > 0:
        vasicek = option_pricing(vasicek, cur-1, T, strike_price )
    return vasicek

def floater_pricing(vasicek, T, cap = 0.08, cap_below = -100, floater = {}):
    for i in range(0, T):
        floater[i] = []
        for j in range(0, 2**i):
            if i > 0:
                tmp = floater[i-1][j/2]['vasicek_coupon']

                start = tmp[0] + 1
                end = tmp[1] + 1

                if (j/2)%2 == 0:
                    node = tmp[2]
                else:
                    node = tmp[2]+1
                    
                coupon = min(vasicek[start, end][node]['rate'], cap)
                coupon = max(cap_below, coupon)
                floater[i-1][j/2]['rate'] = vasicek[start, end][node]['rate']
                floater[i-1][j/2]['vasicek_rate'] = [start, end, node]
            else:
                start = -1
                end = 0
                node = 0
                coupon = 0
                vasicek_rate = None

            floater[i].append({
                'vasicek_coupon': [start, end, node],
                'coupon':coupon,
                })

    for i in range(T-1,-1,-1):

        for j in range(0, 2**i):
            if i == T-1:
                floater[i][j]['price'] = 1#exp(floater[i][j]['coupon'] )
                continue

            f = floater[i][j]
            v = f['vasicek_rate']

            d1 = vasicek[v[0],v[1]][v[2]]['discount']
            d2 = vasicek[v[0],v[1]+1][v[2]]['discount']

            a1 = vasicek[v[0]+1,v[1]+1][v[2]]['discount']
            a2 = vasicek[v[0]+1,v[1]+1][v[2] + 1]['discount']
            
            # b1 = floater[i+1][2*j]['price'] + floater[i+1][2*j]['coupon']
            b1 = floater[i+1][2*j]['price'] * exp( floater[i+1][2*j]['coupon'])
            # b2 = floater[i+1][2*j+1]['price'] + floater[i+1][2*j+1]['coupon']
            b2 = floater[i+1][2*j+1]['price'] *exp( floater[i+1][2*j+1]['coupon'])

            A = np.matrix([[1,a1], [1,a2]])
            b = np.matrix([[b1],[b2]])

            x = np.dot(np.linalg.inv(A), b)
            n1 = float(x[0])
            n2 = float(x[1])
            price = n1*d1 + n2*d2

            floater[i][j]['price'] = price
                
    return floater



def get_call_graph(vasicek, T):
    options = {}
    for i in range(0, T):
        options[i,T] = []
        for j in range(0, i+1):
            options[i,T].append(vasicek[i,T][j]['call'])
    return options


def find_delta(T=8, strike = 1.02, cap = 0.08):
    df = load_data()
    params = get_params(df)
    

    bond, option, floater = get_prices(T, params, strike, cap, 0.0268)
    bond1, option1, floater1 = get_prices(T, params, strike, cap, 0.0268)
    bond2, option2, floater2 = get_prices(T, params, strike, cap, 0.027)

    #find delta
    bond_delta = (bond2 - bond1)/0.0002
    option_delta = (option2 - option1)/0.0002
    floater_delta = (floater2 - floater1)/0.0002

    return bond_delta, option_delta, floater_delta

def get_prices(T, params, strike, cap, r = None):
    df = load_data()
    params = get_params(df)

    if r:
        params['r'] = r
    df, params, vasicek, ts = main(T, params)
    options = option_pricing(vasicek, T-1, T, strike)
    floater = floater_pricing(vasicek, T, cap, cap_below = -100)

    pprint.pprint(params)
    
    return_dict = {
            'bond':vasicek[0,T][0]['discount'],
            'option': options[0,8][0]['call'],
            'floater': floater[0][0]['price']
            }

    pprint.pprint(return_dict)
    return vasicek[0,T][0]['discount'], options[0,8][0]['call'], floater[0][0]['price']



