# -*- coding: utf-8 -*-
"""

Okhtay Wahid Far 870485


"""
import numpy as np
import matplotlib.pyplot as plt
import time


    
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

def euler_backward(f, t_span, y0, t_eval, jac):
    """
    implicit solver for first order initial value problem

    Parameters
    ----------
    f : function with signature dy = f(t, y)
        right hand side of ODE
    jac: jacobian of f with respect to y
    t_span : tupel (t0, tf)
        left and right boundary of interval on which the IVP is solved
    y0 : ndarray of shape (n,)
        initial value y(t_initial)
    t_eval : ndarray of shape (N+1, )


    Returns
    -------
    bunch Object ODEResult with members
        t:  ndarray of shape (N+1, )
            discrete time instances
        y:  ndarray of shape(N+1, n)
            approximated values of solution at time instances given in t
        its: ndarray of shape shape (N+1, )
            Newton iterations in each time step

    """

    tol1 = 1.e-10
    tol2 = 1.e-10
    maxIt = 10

    t0 = t_span[0]
    tf = t_span[1]
    ys = np.zeros((len(y0), len(t_eval)), dtype=float)
    ts = t_eval
    ys[:, 0] = y0

    id = np.eye(len(y0))
    iterations = np.zeros_like(ts, dtype=int)

    for k in range(len(t_eval) - 1):  # time loop

        # Newton Iteration
        h = ts[k + 1] - ts[k]  # Zeitschritt
        y = ys[:, k].copy()  # Startvektor für Newton-Iteration
        t = ts[k + 1]  # nächster Zeitpunkt
        it = 0
        err1 = tol1 + 1
        err2 = tol2 + 1
        while (err1 > tol1 and err2 > tol2 and it < maxIt):
            b = ys[:, k] + h * f(ts[k], y) - y
            A = h * jac(t, y) - id
            delta_y = np.linalg.solve(A, -b)

            y += delta_y
            it += 1
            err1 = np.linalg.norm(delta_y)
            err2 = np.linalg.norm(b)

        ys[:, k + 1] = y
        iterations[k] = it

    return ODEResult(t=ts, y=ys, its=iterations)
def trapez_backward(f, t_span, y0, t_eval, jac):
    """
    implicit solver for first order initial value problem

    Parameters
    ----------
    f : function with signature dy = f(t, y)
        right hand side of ODE
    jac: jacobian of f with respect to y
    t_span : tupel (t0, tf)
        left and right boundary of interval on which the IVP is solved
    y0 : ndarray of shape (n,)
        initial value y(t_initial)
    t_eval : ndarray of shape (N+1, )


    Returns
    -------
    bunch Object ODEResult with members
        t:  ndarray of shape (N+1, )
            discrete time instances
        y:  ndarray of shape(N+1, n)
            approximated values of solution at time instances given in t
        its: ndarray of shape shape (N+1, )
            Newton iterations in each time step

    """

    tol1 = 1.e-10
    tol2 = 1.e-10
    maxIt = 10

    t0 = t_span[0]
    tf = t_span[1]
    ys = np.zeros((len(y0), len(t_eval)), dtype=float)
    ts = t_eval
    ys[:, 0] = y0

    id = np.eye(len(y0))
    iterations = np.zeros_like(ts, dtype=int)

    for k in range(len(t_eval) - 1):  # time loop

        # Newton Iteration
        h = ts[k + 1] - ts[k]  # Zeitschritt
        y = ys[:, k].copy()  # Startvektor für Newton-Iteration
        t = ts[k + 1]  # nächster Zeitpunkt
        it = 0
        err1 = tol1 + 1
        err2 = tol2 + 1
        while (err1 > tol1 and err2 > tol2 and it < maxIt):
            #b = ys[:, k] + h * f(ts[k], y) - y
            #A = h * jac(t, y) - id
            # ys[:, k] = yn-1
            # y= yn
            # f(ts[k], y) = fn
            # newton
            # b: -f(x) = yn - yn-1 -0,5h(fn+fn-1)
            b = -y + ys[:, k] + h/2*(f(ts[k], y) + f(t0, y))
            A = h * jac(t, y) - id
            delta_y = np.linalg.solve(A, -b)

            y += delta_y
            it += 1
            err1 = np.linalg.norm(delta_y)
            err2 = np.linalg.norm(b)

        ys[:, k + 1] = y
        iterations[k+1] = it

    return ODEResult(t=ts, y=ys, its=iterations)

def bdf2(f, t_span, y0, t_eval=None,  jac=None):
    tol1 = 1.e-10
    tol2 = 1.e-10
    maxIt = 100             # maximale Iterationsanzahl fuer Newton-Verfahren

    t0 = t_span[0]
    tf = t_span[1]
    # Falls kein Vektor 't_eval' übergeben wird, wird das Zeitintervall [t0,tf] in 1000 Zeitschritte zerlegt.
    if t_eval is None:
        ts = 0.001 * np.arange(t0 / 1000, tf / 1000)
    else:
        ts = t_eval

    # Vorwärts-Differenzenquotienten appoiximiert
    hy = 1.e-6
    def finite_jac(t, y):
        df1 = np.zeros((len(y), len(y)))
        fy = f(t, y)  # nur einmal f auswerten in (t, y)
        for k in range(len(y)):
            yh = y.copy()
            yh[k] += hy
            df1[:, k] = (f(t, yh) - fy) / hy
        return df1

    if jac is None:
        jac = finite_jac

    ys = np.zeros((len(y0), len(ts)), dtype=float)
    ys[:, 0] = y0
    # Für erste Schritt berechnen wir mit implizite Trapezregel (Ordnung 2, wie BDF2)
    y1_trapez = trapez_backward(f, np.array([ts[0], ts[1]]), y0, np.array([ts[0], ts[1]]), jac)
    ys[:, 1] = y1_trapez.y[:, -1]

    id = np.eye(len(y0))
    iterations = np.zeros_like(ts, dtype=int)       # vektor für Newton-Iterationsanzahlen
    iterations[0] = y1_trapez.its[-1]

    for k in range(1, len(t_eval) - 1):  # time loop

        # Newton Iteration
        h = ts[k + 1] - ts[k]  # Zeitschritt
        y = ys[:, k].copy()  # Startvektor für Newton-Iteration
        t = ts[k + 1]  # nächster Zeitpunkt
        it = 0
        err1 = tol1 + 1
        err2 = tol2 + 1
        while (err1 > tol1 and err2 > tol2 and it < maxIt):
            b = 2*ys[:, k] - 0.5*ys[:, k-1] + h * f(ts[k], y) - 1.5*y    # Newton-Nullstellenproblem
            A = h * jac(t, y) - 1.5*id                                   # Ableitung der Newton-Nullstellenproblem
            delta_y = np.linalg.solve(A, -b)

            y += delta_y
            it += 1
            err1 = np.linalg.norm(delta_y)
            err2 = np.linalg.norm(b)

        ys[:, k + 1] = y
        iterations[k] = it

    return ODEResult(t=ts, y=ys, its=iterations)

def test_euler_reuber_beute():
    ## ------------- Spezifikation des AWPs ----------------------
    # rechte Seite der DGL
    def flotka(t, y):
        return np.array([-0.25 * y[0] + 0.01 * y[0] * y[1], y[1] - 0.05 * y[0] * y[1]])

    # Ableitung der rechten Seite der DGL nach der Zustandsvariablen y
    def flotka_jac(t, y):
        return np.array([[-0.25 + 0.01 * y[1], 0.01 * y[0]],
                         [-0.05 * y[1], 1. - 0.05 * y[0]]])

    # Anfangsbedingung
    y0 = [40, 20]  # Anfangswert des Systems von DGLen, vektorwertig
    t0 = 0.  # Anfangszeitpunkt
    tf = 50.  # Endzeitpunkt
    t_span = (t0, tf)  # Zeitintervall als Tupel

    ## ------------ Lösen des AWPs --------------------------------
    N = 10000  # Anzahl Zeitintervalle; Schrittweite h = (tf - t0)/N
    tk = np.linspace(t0, tf, N + 1)  # an diesen Zeitpunkten soll die Loesung berechnet werden

    solution_euler = euler_backward(flotka, t_span, y0, tk, flotka_jac) #euler
    solution_trapez = trapez_backward(flotka, t_span, y0, tk, flotka_jac) #euler
    print("bdf2")
    solution = bdf2(flotka, t_span, y0, tk, flotka_jac)
    tk = solution.t
    yk = solution.y
    its = solution.its

    ## ------------ Visualisierung der diskreten Lösung -----------------
    plt.figure(figsize=(16, 6))
    plt.rcParams.update({'font.size': 14})

    plt.subplot(1, 3, 1)
    plt.plot(tk, yk[0, :], '-', label=r'Räuber')
    plt.plot(tk, yk[1, :], '-', label=r'Räuber')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$y_i$')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(yk[0, :], yk[1, :], '-', label=r'Phasenraumtrajektorie $y(t)$')
    plt.plot(yk[0, 0], yk[1, 0], '*', markersize=15, label=r'$y(t_0)$')
    plt.xlabel(r'$y_1$')
    plt.ylabel(r'$y_2$')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(tk[:-1], its[:-1], '*')
    plt.xlabel(r'$t$')
    plt.ylabel(r'Anzahl Iterationen')
def plot_vanderpol_oszillator(mu=0, tf=20):
    ## ------------- Spezifikation des AWPs ----------------------
    # y =[y y']   # y = y[0] y' = y[1]
    # f(y) = [y' y''] = [y' mu*(1-y^2)*y'-y]
    # rechte Seite der DGL
    def fvanderpol(t, y):
        return np.array([y[1] , mu*(1-y[0]**2)*y[1]-y[0]])

    # Ableitung der rechten Seite der DGL nach der Zustandsvariablen y
    def fvanderpol_jac(t, y):
        return np.array([[0, 1],
                         [-mu*2*y[0]*y[1]-1, mu*(1-y[0]**2)]])

    # Anfangsbedingung
    y0 = [2, 0]  # Anfangswert des Systems von DGLen, vektorwertig
    t0 = 0.  # Anfangszeitpunkt
    #tf = 50.  # Endzeitpunkt
    t_span = (t0, tf)  # Zeitintervall als Tupel

    ## ------------ Lösen des AWPs --------------------------------
    N = 10000  # Anzahl Zeitintervalle; Schrittweite h = (tf - t0)/N
    tk = np.linspace(t0, tf, N + 1)  # an diesen Zeitpunkten soll die Loesung berechnet werden

    solution = bdf2(fvanderpol, t_span, y0, tk)
    tk = solution.t
    yk = solution.y
    its = solution.its
    ## ------------ Visualisierung der diskreten Lösung -----------------
    plt.figure(figsize=(16, 6))
    plt.rcParams.update({'font.size': 14})

    plt.subplot(1, 3, 1)
    plt.plot(tk, yk[0, :])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$y_i$')

    plt.subplot(1, 3, 2)
    plt.plot(yk[0, :], yk[1, :])
    plt.xlabel(r'$y_1$')
    plt.ylabel(r'$y_2$')
    plt.title(r'BDF2')

    plt.subplot(1, 3, 3)
    plt.plot(tk[:-1], its[:-1], '*')
    plt.xlabel(r'$t$')
    plt.ylabel(r'Anzahl Iterationen')

    solution = rungekutta_43(fvanderpol, t_span, y0)
    tk = solution.t
    yk = solution.y
    ## ------------ Visualisierung der diskreten Lösung -----------------
    plt.figure(figsize=(16, 6))
    plt.rcParams.update({'font.size': 14})

    plt.subplot(1, 3, 1)
    plt.plot(tk, yk[0, :])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$y_i$')

    plt.subplot(1, 3, 2)
    plt.plot(yk[0, :], yk[1, :])
    plt.xlabel(r'$y_1$')
    plt.ylabel(r'$y_2$')
    plt.title(r'RK4(3)')


if __name__ == "__main__":
    # test_skalar()
    # test_system()
    #
    #test_euler_reuber_beute()
    #test_log()
    plot_vanderpol_oszillator(10,50)
