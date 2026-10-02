# -*- coding: utf-8 -*-
#"""
#Created on Tue Sep 19 12:55:04 2023
#
#@author: petrip
#"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# =============================================================================
# SYSTEM AV FÖRSTA ORDNINGENS DIFFERENTIALEKVATIONER
# =============================================================================

def MittSystem(t, u, g, k, c, m1, m2, a, L):

    u1_dot = u[2]  # dz
    u2_dot = u[3]  # dtheta

    u3_dot = (
        m2*a*np.sin(u[1])*u[3]**2
        - c*u[2]
        - k*(u[0]-L)
    )

    u4_dot = (
        -m2*g*a*np.sin(u[1])
    )

    return np.array([
        u1_dot,
        u2_dot,
        u3_dot,
        u4_dot
    ])


# =============================================================================
# MASSMATRIS
# =============================================================================

def Massmatris(t, u, m1, m2, a):

    M = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, m1+m2, m2*a*np.cos(u[1])],
        [0, 0, m2*a*np.cos(u[1]), m2*a**2]
    ])

    return M


# =============================================================================
# SYSTEMET OCH MASSMATRISEN
# =============================================================================

def System_med_massmatris(t, u, g, k, c, m1, m2, a, L):

    # Beräkna systemet utan massmatris
    f = MittSystem(t, u, g, k, c, m1, m2, a, L)

    # Beräkna massmatrisen
    M = Massmatris(t, u, m1, m2, a)

    # Ta fram u_dot genom att multiplicera båda sidorna med M^{-1}
    u_dot = np.linalg.solve(M, f)

    return u_dot


# =============================================================================
# PARAMETRAR
# =============================================================================

g = 9.82
a = 1
L = 1
k = 5
c = 0
m1 = 1
m2 = 1


# =============================================================================
# LÖSNING AV DET DYNAMISKA SYSTEMET
# =============================================================================

# Begynnelsevillkor

z0 = 1.1
theta0 = np.pi/6
z_dot0 = 0
theta_dot0 = 0

u_0 = np.array([
    z0,
    theta0,
    z_dot0,
    theta_dot0
])


# Simuleringstid

t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)


# Integrationsparametrar

ode_args = (
    g,
    k,
    c,
    m1,
    m2,
    a,
    L
)


# ODE-lösaren

sol = solve_ivp(
    System_med_massmatris,
    t_span,
    u_0,
    method='RK45',
    t_eval=t_eval,
    args=ode_args,
    rtol=1e-4,
    atol=1e-4
)


# =============================================================================
# HÄMTA LÖSNINGARNA
# =============================================================================

z = sol.y[0]
theta = sol.y[1]

z_dot = sol.y[2]
theta_dot = sol.y[3]


# =============================================================================
# GRAF 1: z(t) OCH theta(t)
# =============================================================================

plt.figure()

plt.subplot(2, 1, 1)

plt.plot(sol.t, sol.y[0], 'k')

plt.xlabel('t [s]')
plt.ylabel('z [m]')

plt.grid()


plt.subplot(2, 1, 2)

plt.plot(sol.t, sol.y[1], 'k')

plt.xlabel('t [s]')
plt.ylabel('theta [rad]')

plt.grid()

plt.tight_layout()


# =============================================================================
# MASSORNAS POSITIONER
# =============================================================================

# m1

x1 = z
y1 = np.zeros_like(z)


# m2

x2 = z + a*np.sin(theta)
y2 = -a*np.cos(theta)


# =============================================================================
# GRAF 2: m1 OCH m2:s POSITIONER I xy-PLANET
# =============================================================================

plt.figure()

plt.plot(
    x1,
    y1,
    label='m1'
)

plt.plot(
    x2,
    y2,
    label='m2'
)

plt.xlabel('x [m]')
plt.ylabel('y [m]')

plt.title('Massornas positioner i xy-planet')

plt.legend()
plt.grid()
plt.axis('equal')


# =============================================================================
# SYSTEMETS TYNGDPUNKT
# =============================================================================

xG = (
    m1*x1
    + m2*x2
) / (m1+m2)

yG = (
    m1*y1
    + m2*y2
) / (m1+m2)


# =============================================================================
# GRAF 3: TYNGDPUNKTENS POSITION
# =============================================================================

plt.figure()

plt.plot(
    xG,
    yG,
    'k'
)

plt.xlabel('x [m]')
plt.ylabel('y [m]')

plt.title('Systemets tyngdpunkt i xy-planet')

plt.grid()
plt.axis('equal')


# =============================================================================
# AVSTÅND MELLAN MASSORNA
# =============================================================================

avstand = np.sqrt(
    (x2-x1)**2
    +
    (y2-y1)**2
)


# =============================================================================
# GRAF 4: AVSTÅND MELLAN m1 OCH m2
# =============================================================================

plt.figure()

plt.plot(
    sol.t,
    avstand,
    'k'
)

plt.xlabel('t [s]')
plt.ylabel('Avstånd [m]')

plt.title('Avståndet mellan m1 och m2')

plt.grid()


# =============================================================================
# ENERGI
# =============================================================================

# Hastighet för m1

v1_x = z_dot
v1_y = np.zeros_like(z_dot)


# Hastighet för m2

v2_x = (
    z_dot
    +
    a*np.cos(theta)*theta_dot
)

v2_y = (
    a*np.sin(theta)*theta_dot
)


# Kinetisk energi för m1

T1 = (
    0.5
    * m1
    * (v1_x**2 + v1_y**2)
)


# Kinetisk energi för m2

T2 = (
    0.5
    * m2
    * (v2_x**2 + v2_y**2)
)


# Total kinetisk energi

T = T1 + T2


# Potentiell energi i fjädern

V_fjader = (
    0.5
    * k
    * (z-L)**2
)


# Potentiell energi från gravitationen

V_gravitation = (
    m2
    * g
    * a
    * (1-np.cos(theta))
)


# Total potentiell energi

V = (
    V_fjader
    +
    V_gravitation
)


# Total mekanisk energi

E = T + V


# =============================================================================
# GRAF 5: KINETISK, POTENTIELL OCH TOTAL ENERGI
# =============================================================================

plt.figure()

plt.plot(
    sol.t,
    T,
    label='Kinetisk energi T'
)

plt.plot(
    sol.t,
    V,
    label='Potentiell energi V'
)

plt.plot(
    sol.t,
    E,
    label='Total energi E'
)

plt.xlabel('t [s]')
plt.ylabel('Energi [J]')

plt.title('Systemets energi')

plt.legend()
plt.grid()


# =============================================================================
# FUNKTION FÖR ATT TESTA OLIKA PARAMETRAR
# =============================================================================

def solve_case(
    g_case,
    k_case,
    c_case,
    m1_case,
    m2_case,
    a_case,
    L_case,
    u0_case
):

    ode_args_case = (
        g_case,
        k_case,
        c_case,
        m1_case,
        m2_case,
        a_case,
        L_case
    )

    sol_case = solve_ivp(
        System_med_massmatris,
        t_span,
        u0_case,
        method='RK45',
        t_eval=t_eval,
        args=ode_args_case,
        rtol=1e-4,
        atol=1e-4
    )

    return sol_case


# =============================================================================
# DÄMPNING
# =============================================================================

# Uppgiften säger att c ska ökas.
# Här jämförs c = 0, 1 och 2 Ns/m.

c_values = [
    0,
    1,
    2
]


# =============================================================================
# GRAF 6: DÄMPNINGENS PÅVERKAN
# =============================================================================

plt.figure()


# z

plt.subplot(2, 1, 1)

for c_test in c_values:

    sol_test = solve_case(
        g,
        k,
        c_test,
        m1,
        m2,
        a,
        L,
        u_0
    )

    plt.plot(
        sol_test.t,
        sol_test.y[0],
        label='c = ' + str(c_test)
    )


plt.xlabel('t [s]')
plt.ylabel('z [m]')

plt.title('Dämpningens påverkan på z')

plt.legend()
plt.grid()


# theta

plt.subplot(2, 1, 2)

for c_test in c_values:

    sol_test = solve_case(
        g,
        k,
        c_test,
        m1,
        m2,
        a,
        L,
        u_0
    )

    plt.plot(
        sol_test.t,
        sol_test.y[1],
        label='c = ' + str(c_test)
    )


plt.xlabel('t [s]')
plt.ylabel('theta [rad]')

plt.title('Dämpningens påverkan på theta')

plt.legend()
plt.grid()

plt.tight_layout()


# =============================================================================
# JÄMVIKTSLÄGE
# =============================================================================

# Jämviktsläget för systemet:
#
# z = L
# theta = 0
# z_dot = 0
# theta_dot = 0

u_jamvikt = np.array([
    L,
    0,
    0,
    0
])


sol_jamvikt = solve_case(
    g,
    k,
    c,
    m1,
    m2,
    a,
    L,
    u_jamvikt
)


# =============================================================================
# GRAF 7: NUMERISK KONTROLL AV JÄMVIKT
# =============================================================================

plt.figure()


plt.subplot(2, 1, 1)

plt.plot(
    sol_jamvikt.t,
    sol_jamvikt.y[0],
    'k'
)

plt.xlabel('t [s]')
plt.ylabel('z [m]')

plt.title('Jämviktskontroll: z')

plt.grid()


plt.subplot(2, 1, 2)

plt.plot(
    sol_jamvikt.t,
    sol_jamvikt.y[1],
    'k'
)

plt.xlabel('t [s]')
plt.ylabel('theta [rad]')

plt.title('Jämviktskontroll: theta')

plt.grid()

plt.tight_layout()


# =============================================================================
# MASSFÖRHÅLLANDE
# =============================================================================

# Tre olika värden på m2 jämförs.

m2_values = [
    0.5,
    1,
    2
]


# =============================================================================
# GRAFER 8-10: ÄNDRAT MASSFÖRHÅLLANDE
# =============================================================================

plt.figure()


for i, m2_test in enumerate(m2_values):

    sol_test = solve_case(
        g,
        k,
        c,
        m1,
        m2_test,
        a,
        L,
        u_0
    )

    plt.subplot(
        3,
        1,
        i+1
    )

    plt.plot(
        sol_test.t,
        sol_test.y[0],
        label='z'
    )

    plt.plot(
        sol_test.t,
        sol_test.y[1],
        label='theta'
    )

    plt.title(
        'm1 = '
        + str(m1)
        + ' kg, m2 = '
        + str(m2_test)
        + ' kg'
    )

    plt.xlabel('t [s]')

    plt.legend()
    plt.grid()


plt.tight_layout()


# =============================================================================
# FJÄDERKONSTANT
# =============================================================================

# Tre olika värden på k jämförs.

k_values = [
    2.5,
    5,
    10
]


# =============================================================================
# GRAFER 11-13: ÄNDRAD FJÄDERKONSTANT
# =============================================================================

plt.figure()


for i, k_test in enumerate(k_values):

    sol_test = solve_case(
        g,
        k_test,
        c,
        m1,
        m2,
        a,
        L,
        u_0
    )

    plt.subplot(
        3,
        1,
        i+1
    )

    plt.plot(
        sol_test.t,
        sol_test.y[0],
        label='z'
    )

    plt.plot(
        sol_test.t,
        sol_test.y[1],
        label='theta'
    )

    plt.title(
        'k = '
        + str(k_test)
        + ' N/m'
    )

    plt.xlabel('t [s]')

    plt.legend()
    plt.grid()


plt.tight_layout()


# =============================================================================
# STORT VÄRDE PÅ m2
# =============================================================================

# Uppgiften föreslår att man testar exempelvis m2 = 100 kg.

m2_stor = 100


sol_stor_m2 = solve_case(
    g,
    k,
    c,
    m1,
    m2_stor,
    a,
    L,
    u_0
)


# =============================================================================
# GRAF 14: m2 = 100 kg
# =============================================================================

plt.figure()


plt.subplot(2, 1, 1)

plt.plot(
    sol_stor_m2.t,
    sol_stor_m2.y[0],
    'k'
)

plt.xlabel('t [s]')
plt.ylabel('z [m]')

plt.title('z för m2 = 100 kg')

plt.grid()


plt.subplot(2, 1, 2)

plt.plot(
    sol_stor_m2.t,
    sol_stor_m2.y[1],
    'k'
)

plt.xlabel('t [s]')
plt.ylabel('theta [rad]')

plt.title('theta för m2 = 100 kg')

plt.grid()

plt.tight_layout()


# =============================================================================
# VISA ALLA GRAFER
# =============================================================================

plt.show()


# =============================================================================
# ANIMATION
# =============================================================================

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:43:33 2026

@author: petrip
"""

# Denna kod kan ni använda tillsammans med simuleringskoden för att animera
# pendelsystemet.
# Det är ett bra verktyg för att se om allt ser rätt ut samt att man ser direkt
# vad en förändring av parametrar leder till.

# Lägg till bibliotek

import matplotlib.animation as animation #animaation



# ----------------------------
# Kinematik
# ----------------------------

import numpy as np
import matplotlib.pyplot as plt


z = sol.y[0]
theta = sol.y[1]


# m1

x1 = z
y1 = np.zeros_like(z)


# m2

x2 = z + a*np.sin(theta)
y2 = -a*np.cos(theta)


# ----------------------------
# Fjäder
# ----------------------------

def spring_coords(x_end, n=12, amp=0.05):

    xs = np.linspace(
        0,
        x_end,
        2*n+1
    )

    ys = np.zeros_like(xs)

    ys[1:-1:2] = amp
    ys[2:-1:2] = -amp

    ys[0] = 0
    ys[-1] = 0

    return xs, ys


# ----------------------------
# Figur
# ----------------------------

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.set_aspect('equal')

ax.grid()


ax.set_xlim(
    np.min(x2)-0.5,
    np.max(x2)+0.5
)


ax.set_ylim(
    -1.2*a,
    0.4*a
)


ax.set_xlabel(
    r'$x$'
)

ax.set_ylabel(
    r'$y$'
)


ax.set_title(
    'Fjäder-pendelsystem'
)


# planet

ax.plot(
    [
        np.min(x1)-0.5,
        np.max(x1)+0.5
    ],
    [
        0,
        0
    ],
    'k--',
    lw=1
)


# objekt

spring, = ax.plot(
    [],
    [],
    'r',
    lw=2
)

pendulum, = ax.plot(
    [],
    [],
    'k',
    lw=2
)

mass1, = ax.plot(
    [],
    [],
    'ks',
    markersize=12
)

mass2, = ax.plot(
    [],
    [],
    'ro',
    markersize=8
)

trace, = ax.plot(
    [],
    [],
    'b:',
    lw=1
)


# ----------------------------
# Initiering
# ----------------------------

def init():

    spring.set_data(
        [],
        []
    )

    pendulum.set_data(
        [],
        []
    )

    mass1.set_data(
        [],
        []
    )

    mass2.set_data(
        [],
        []
    )

    trace.set_data(
        [],
        []
    )

    return (
        spring,
        pendulum,
        mass1,
        mass2,
        trace
    )


# ----------------------------
# Animation
# ----------------------------

def animate(i):


    xs, ys = spring_coords(
        x1[i]
    )


    spring.set_data(
        xs,
        ys
    )


    pendulum.set_data(
        [
            x1[i],
            x2[i]
        ],
        [
            y1[i],
            y2[i]
        ]
    )


    mass1.set_data(
        [
            x1[i]
        ],
        [
            y1[i]
        ]
    )


    mass2.set_data(
        [
            x2[i]
        ],
        [
            y2[i]
        ]
    )


    trace.set_data(
        x2[:i],
        y2[:i]
    )


    return (
        spring,
        pendulum,
        mass1,
        mass2,
        trace
    )


anim = animation.FuncAnimation(
    fig,
    animate,
    frames=range(
        0,
        len(sol.t),
        20
    ),
    init_func=init,
    interval=100,
    blit=False
)


plt.show()