
import numpy as np


def quadSimpson(f, a, b, n):
    x = np.linspace(a,b,n)
    y = f(x)
    h = (b - a) / (n - 1)

    simpson = 0;
    for j in range(1,n - 1):
        simpson += y[j-1] + 4 * y[j] + y[j + 1] 
    return simpson * h / 6

def quadAdaptiv(f, a, b, tol, hmin = 1.e-6):
    nodes = [a, b]  # Liste der Auswertungsstellen
    fa = f(a);
    fb = f(b);
    Q = quadAdaptivRec(f, a, b, fa, fb, tol, hmin, nodes)
    return (nodes, Q )

def quadAdaptivRec(f, a, b, fa, fb, tol, hmin, nodes):
    m = (a + b) / 2 # Mittelpunkt
    nodes.append(m)   
    
    h = b - a;
    # Abbruch, fallbedingung für h < hmin
    if h < hmin:
        raise ValueError('Verfahren konvergiert nicht. Minimum größe von h ist erricht.')
        
    
    fm = f(m) # Funktionsauswertung an der Mittelpunkt
    # Quadrature des Simpons-Regel
    qSimpson = h * (fa + 4 * fm + fb) / 6;
    # Quadrature des Trapez-Regel
    qTrapez  = h * (fa + fb) / 2;
    # lokale Fehler
    err      = np.abs( qSimpson - qTrapez )
    
    if err <= tol:
        return  qSimpson # 
    else: 
        return  quadAdaptivRec(f, a, m, fa, fm, tol/2, hmin, nodes) + quadAdaptivRec(f, m, b, fm, fb, tol/2, hmin, nodes)
    # quadAdaptivRec( ... linker intervallteil ) + quadAdaptivRec( rechter intervallteil)
            # die Toleranz muss halbiert werden ... 



