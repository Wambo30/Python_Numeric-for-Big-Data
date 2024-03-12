# -*- coding: utf-8 -*-
"""
Wahid Far,Okhtay 870485

"""
import numpy as np
import matplotlib.pyplot as plt


    
class ODEResult(dict):
    ''' Container object exposing keys as attributes.
    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`'''
    
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError
            
    def __setattr__(self, key, value):
        self[key] = value        
            
    
def euler_forward(f, t_span, y0, t_eval):
    """
    Approximate the solution of y'=f(y,t) by Euler's method.

    Parameters
    ----------
    f : function
        Right-hand side of the differential equation y'=f(t,y), y(t_0)=y_0
    t_span: array [t0 tf]
        Interval limits
    y0 : number
        Initial value y(t0)=y0 wher t0 is the entry at index 0 in the array t
    t_eval : array
        1D NumPy array of t values where we approximate y values.

    Returns
    -------
    ODEResult : {t, y, status} object
        t: 1D NumPy array of t values where we approximate y values.
        y: Approximation y[n] of the solution y(t_n) computed by Euler's method.
        status: int
            Reason for algorithm termination:
                -1: Integration step failed.
                0: The solver successfully reached the end of tspan.
                1: A termination event occurred.
    """

    t0 = t_span[0]
    tf = t_span[1]
    ys = np.zeros( (len(y0), len(t_eval) ), dtype = float )
    ts = t_eval
    ys[:, 0] = y0
    
    for k in range( len(t_eval) - 1 ):
        h = ts[k+1] - ts[k] # Zeitschritt
        ys[:, k+1] = ys[:, k] + h*f(ts[k], ys[:, k])
    
    return ODEResult(t=ts, y=ys, status=1)


def rungekutta(f, t_span, y0, h):
    """
    Approximate the solution of y'=f(y,t) by Runge-Kutta's method with constant step.

    Parameters
    ----------
    f : function
        Right-hand side of the differential equation y'=f(t,y), y(t_0)=y_0
    t_span: array [t0 tf]
        Interval limits
    y0 : number
        Initial value y(t0)=y0 wher t0 is the entry at index 0 in the array t
    h : number
        step size

    Returns
    -------
    ODEResult : {t, y, eval_number} object
        t :     1D NumPy array of t values where we approximate y values.
        y :     Approximation y[n] of the solution y(t_n) computed by Euler's method.
        eval_numbers :   number of function evaluations
        status: int
            Reason for algorithm termination:
                -1: Integration step failed.
                0: The solver successfully reached the end of tspan.
                1: A termination event occurred.
    """
    stat  = 1                 # Status der Program
    t0 = t_span[0]            # Startzeitpunkt
    tf = t_span[1]            # Endzeitpunkt
    ts = [t0]                 # Liste für die diskrete Zeitpunkte
    ys = [y0]                 # Liste für die diskrete Loesung

    t = t0
    y = y0

    k1 = f(t, y)

    eval_number = 1             # Anzahl der Auswertungen

    while t <= tf:
        t_neu = t + h

        # Berechnen k's
        k2 = f(t + h/2, y + h*k1/2)
        k3 = f(t + h/2, y + h*k2/2)
        k4 = f(t + h  , y + h*k3)
        k5 = f(t + h  , y + h*k1/6 + h*k2/3 + h*k3/3 + h*k4/6)

        eval_number = eval_number + 4

        y = y + h*(k1/6 + k2/3 + k3/3 + k4/6)
        k1 = k5      #  FSAL
        t  = t_neu
        ys.append(y)             #  an die Liste anhaengen
        ts.append(t)


    ys = np.array(ys).transpose()
    ts = np.array(ts)

    return ODEResult(t=ts, y=ys, eval_numbers=eval_number, status=stat)


def rungekutta_43(f, t_span, y0, tol=1.e-6):
    """
    Approximate the solution of y'=f(y,t) by Runge-Kutta's method with adaptive size.

    Parameters
    ----------
    f : function
        Right-hand side of the differential equation y'=f(t,y), y(t_0)=y_0
    t_span: array [t0 tf]
        Interval limits
    y0 : number
        Initial value y(t0)=y0 wher t0 is the entry at index 0 in the array t
    tol : number (optional)
        error tolerance

    Returns
    -------
    ODEResult : {t, y, eval_number} object
        t :     1D NumPy array of t values where we approximate y values.
        y :     Approximation y[n] of the solution y(t_n) computed by Euler's method.
        eval_numbers :   number of function evaluations
        status: int
            Reason for algorithm termination:
                -1: Integration step failed.
                0: The solver successfully reached the end of tspan.
                1: A termination event occurred.
    """
    stat  = 1                 # Status der Programm
    q    = 5.                 # Hochschaltbegrenzung
    rho  = 0.9                # Sicherheitsfaktor
    s_min = 1.e-7             # absoluter Schwellwert fuer die interne Skalierung
    t0 = t_span[0]            # Startzeitpunkt
    tf = t_span[1]            # Endzeitpunkt
    h_max = (tf - t0)/10.     # Maximale Schrittweite
    h_min = (tf - t0)*1.e-10  # Minimale Schrittweite
    ts = [t0]                 # Liste für die diskrete Zeitpunkte
    ys = [y0]                 # Liste für die diskrete Loesung

    t = t0
    y = y0
    h = h_max
    k1 = f(t, y)

    eval_number = 1             # Anzahl der Auswertungen

    while t < tf:
        t_neu = t + h

        # Berechnen k's
        k2 = f(t + h/2, y + h*k1/2)
        k3 = f(t + h/2, y + h*k2/2)
        k4 = f(t + h  , y + h*k3)
        k5 = f(t + h  , y + h*k1/6 + h*k2/3 + h*k3/3 + h*k4/6)

        eval_number = eval_number + 4

        y_neu = y + h*(k1/6 + k2/3 + k3/3 + k4/6)
        y_neu_dach = y + h*(k1/6 + k2/3 + k3/3 + k5/6)
                              
        fehler   = np.linalg.norm(y_neu-y_neu_dach)  # Fehlerschaetzer sj # |y_neu - y_neu_dach|
        h_opt =   h*(rho*tol/fehler)**(1/4)      # Schrittweitenvorschlag # Ordung von RK p=3

        h_neu     = np.min([h_opt, h_max, q*h])     # Begrenzung der Schrittweite

        if (h_neu < h_min) :          # Abbruch, falls Schrittweite zu klein
            stat = -1                 # Abbruch-Status
            break

    
        if (fehler <= tol):    # Schritt wird akzeptiert
            y  = y_neu
            k1 = k5      #  FSAL
            t  = t_neu
            h  = min(h_neu, tf - t)  # damit letzter Zeitschritt richtig
            ys.append(y)             #  an die Liste anhaengen
            ts.append(t)
        else:                # Schritt wird abgelehnt
            h = h_neu
    
    ys = np.array(ys).transpose()      
    ts = np.array(ts)
    
    return ODEResult(t=ts, y=ys, eval_numbers=eval_number, status=stat)

def test_skalar():
    import matplotlib.pyplot as plt
    lamb = 1.
    K    = 10
    y0 = np.array([0.1])    # Anfangswert
    t0 = 0.                 # Anfangszeitpunkt
    tf = 10.                # Endzeitpunkt
    t_span = [t0, tf]       # Zeitintervall
    #' flogistic'
    def f(t, y):
        return lamb*(1. - y/K)*y
    def y_log(t, t0, y0):
        return y0*np.exp(lamb*(t-t0))/((1. - y0/K) + y0*np.exp(lamb*(t-t0))/K)

    # Test Euler
    t_eval = np.linspace(t0,tf,50)
    sol1 = euler_forward(f, t_span, y0, t_eval)
    plt.plot(sol1.t, sol1.y[0,:], label='euler_forward')

    # Test Runge Kuta
    sol2 = rungekutta_43(f, t_span, y0)
    plt.plot(sol2.t, sol2.y[0,:], label='Runge Kutta 4(3)')

    # Exakte Lösung
    fk_exakt = y_log(t_eval, t0, y0)
    plt.plot(t_eval, fk_exakt, '*', label='exact')
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.title(r'Logistisches Wachstum: $y^\prime = (1 - \frac{y}{10})y$')
    plt.show()

def err_logistic():
    # Fehler-Analyse bei verschiedene Toleranz
    import matplotlib.pyplot as plt
    lamb = 1.
    K    = 10
    y0 = np.array([0.1])    # Anfangswert
    t0 = 0.                 # Anfangszeitpunkt
    tf = 10.                # Endzeitpunkt
    t_span = [t0, tf]       # Zeitintervall
    #' flogistic'
    def f(t, y):
        return lamb*(1. - y/K)*y
    def y_log(t, t0, y0):
        return y0*np.exp(lamb*(t-t0))/((1. - y0/K) + y0*np.exp(lamb*(t-t0))/K)

    m = 10
    TOLi = 1/10**(np.arange(m) + 1)         # Toleranz
    Ei = np.zeros(len(TOLi))                # globale Fehler
    eval_numbers_i = np.zeros(len(TOLi))    # Anzahl der Funktionsauswertungen


    for k in range(len(TOLi)):
        solution = rungekutta_43(f, t_span, y0, TOLi[k])
        yk_exakt = y_log(solution.t, t0, y0)           # Berechnung der Loesungswerte an den Gitterpunken
        yk       = solution.y
        Ei[k]    = np.linalg.norm( yk[0,:] - yk_exakt, np.inf)
        eval_numbers_i[k] = solution.eval_numbers

    plt.plot(TOLi, eval_numbers_i, '*', label = r'Funktionsauswertungen')
    plt.xlabel('TOL')
    plt.ylabel('Funktionsauswertungen')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('Fehler')
    plt.show()

def test_system():
    # ... auch vektorwertiges Problem testen, z.B. Lotka Volterra
    import matplotlib.pyplot as plt
    y0 = [40, 20] # Anfangswert des Systems von DGLen, vektorwertig (oder "2 Anfangswerte")
    t0 = 0.       # Anfangszeitpunkt
    tf = 60.      # Endzeitpunkt
    t_span = [t0, tf]       # Zeitintervall

    #  Räuber-Beute-Modell
    def lotka_volterra(t, y):
        return np.array( [-0.25*y[0] + 0.01*y[0]*y[1], y[1] - 0.05*y[0]*y[1]] )


    # Test Runge Kuta
    solution = rungekutta_43(lotka_volterra, t_span, y0)
    tk = solution.t
    yk = solution.y
    plt.figure(figsize=(8,6))
    plt.plot(tk, yk[0,:], '-', label=r'$y_1$')
    plt.plot(tk, yk[1,:], '-', label=r'$y_2$')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.title('Raeuber-Beute-Modell')
    plt.show()

if __name__ == "__main__":
    test_skalar()
    test_system()
