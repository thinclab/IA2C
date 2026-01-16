import math

def theory_win_rate_calculator(p_T, r0, p_A, V):
    if not p_T < r0 * V:
        print('Not satisfy requirement p_T < r0 * V')
        return None 
    L_D = p_T/V - p_A/(1+V)
    rD = r0 - p_A/(1+V)
    print(1-(L_D**2/rD**2))
    print(math.acos(math.sqrt(1-(L_D**2/rD**2))))
    P_A = 1 - (math.acos(math.sqrt(1-(L_D**2/rD**2))))/math.pi

    return L_D, rD, P_A

L_D, rD, P_A = theory_win_rate_calculator(0.5, 0.2, 0.1, 2.6)
print('L_D is ' + str(L_D))
print('r*D is ' + str(rD))
print('Intruder Winning Rate is ' + str(P_A))
print('Defender Winning Rate is ' + str(1-P_A))